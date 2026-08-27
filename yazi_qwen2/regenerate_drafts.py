"""
Qwen2 egitim seti icin 'draft' alanlarini DeepSeek API (deepseek-chat) ile
yeniden ureten script.

Ne yapar:
  - Her jsonl dosyasindaki (train/val/test) kayitlari okur.
  - Sistem promptunu 'canonical_system_prompt.txt' ile degistirir (tek sema).
  - assistant JSON'undaki 'responsible_unit' anahtarini 'target_unit' olarak yeniden adlandirir.
  - selected_legislation DOLU olan kayitlarda (~%85) 'draft' alanini DeepSeek ile
    yeniden ureterek verilen mevzuati dogal ve dogrudan atifla (halusinasyonsuz)
    kullanmasini saglar. Diger 6 assistant alani DEGISTIRILMEZ.
  - selected_legislation BOS olan (fallback) kayitlarda draft DEGISTIRILMEZ.
  - Sonuc her kaynak dosya icin '<isim>_updated.jsonl' olarak yazilir (orijinaller dokunulmaz).
  - Her kayit icin API cevabi scratchpad/deepseek_cache altinda cache'lenir; script
    kesintiye ugrarsa ayni kayitlar icin tekrar API cagrisi yapilmaz (resume).

Kullanim:
  python regenerate_drafts.py [--limit N] [--files train,val,test] [--workers 8]
  (DEEPSEEK_API_KEY ortam degiskeni gerekir)
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import threading
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get(
    "DEEPSEEK_CACHE_DIR",
    os.path.join(HERE, ".cache", "deepseek_cache"),
)
os.makedirs(CACHE_DIR, exist_ok=True)

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"

STOP_EVENT = threading.Event()


class InsufficientBalanceError(Exception):
    pass

with open(os.path.join(HERE, "canonical_system_prompt.txt"), encoding="utf-8") as f:
    CANONICAL_SYSTEM_PROMPT = f.read().rstrip("\n")

DEEPSEEK_SYSTEM_PROMPT = """Sen Türkçe T.C. resmî yazışma usul ve esaslarına tam hakim, belediye idari yazışma uzmanı bir metin yazarısın.
Görevin: sana verilen başvuru analizi, idari karar bilgileri ve mevzuat maddelerine dayanarak SADECE resmî yazı taslağını (draft) üretmektir.

KURALLAR:
1. Taslak; T.C. başlığı, belediye başkanlığı/müdürlük satırı, Sayı, Konu, (varsa İlgi), muhatap hitabı ("Sayın ..."), gövde paragrafları, kapanış ifadesi ve imza bloğundan oluşur. Minimum 250 kelime.
2. Sana 'selected_legislation' listesinde verilen mevzuat maddelerinden EN AZ BİRİNE, gövde paragrafında kanun adı ve madde numarasıyla DOĞRUDAN ATIF yap (örnek: "5393 sayılı Belediye Kanunu'nun 14. maddesi uyarınca..."). Bu atfı reference_text'in birebir kopyası olarak değil, somut olaya uyarlanmış doğal bir cümle içinde kullan. Amaç mevzuatı ezberlemek değil, olaya doğru şekilde uygulamaktır.
3. SADECE selected_legislation içinde verilen kanun/madde/yönetmelikleri kullan. Listede olmayan hiçbir kanun, madde veya yönetmelik adı UYDURMA (halüsinasyon kesinlikle yasak).
4. Kapanış ifadesi gönderici tipine göre ZORUNLU olarak şu şekilde olmalı:
   - sender_type VATANDAS veya OZEL_KURULUS ise: "Bilgilerinizi rica ederim." veya "Gereğini saygılarımla rica ederim." ("arz ederim" YASAK)
   - Üst/denk makam (Valilik, Kaymakamlık, Mahkeme, Savcılık, Bakanlık) ise: "Gereğini saygılarımla arz ederim." ("rica ederim" YASAK)
   - KAMU_KURUMU ise: "Gereğini bilgilerinize arz ve rica ederim."
5. Diğer idari karar alanları (target_unit, response_type, process_status, performed_actions, result_information, process_information) sana verilmiştir; bunlarla ÇELİŞMEYEN bir taslak yaz, bu alanları taslak içinde tekrar üretme.
6. Sadece geçerli JSON döndür: {"draft": "..."} - başka hiçbir açıklama, markdown ya da metin ekleme.
"""


def build_user_prompt(user_data: dict, assistant_data: dict) -> str:
    context = {
        "document_type": user_data.get("document_type"),
        "sender_type": user_data.get("sender_type"),
        "primary_topic": user_data.get("primary_topic"),
        "requested_action": user_data.get("requested_action"),
        "key_information": user_data.get("key_information"),
        "missing_information": user_data.get("missing_information"),
        "summary": user_data.get("summary"),
        "process_status": user_data.get("process_status"),
        "selected_legislation": user_data.get("selected_legislation"),
        "karar_bilgileri": {
            "target_unit": assistant_data.get("target_unit") or assistant_data.get("responsible_unit"),
            "response_type": assistant_data.get("response_type"),
            "process_status": assistant_data.get("process_status"),
            "performed_actions": assistant_data.get("performed_actions"),
            "result_information": assistant_data.get("result_information"),
            "process_information": assistant_data.get("process_information"),
        },
    }
    return json.dumps(context, ensure_ascii=False)


def cache_path(record_uid: str) -> str:
    return os.path.join(CACHE_DIR, f"{record_uid}.json")


def record_uid(fname: str, idx: int, user_content: str) -> str:
    h = hashlib.sha1(user_content.encode("utf-8")).hexdigest()[:10]
    return f"{fname}_{idx}_{h}"


def call_deepseek(api_key: str, user_prompt: str, timeout: int = 60) -> str:
    resp = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.4,
            "response_format": {"type": "json_object"},
        },
        timeout=timeout,
    )
    if resp.status_code in (402, 403):
        raise InsufficientBalanceError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    draft = parsed["draft"]
    if not isinstance(draft, str) or not draft.strip():
        raise ValueError("empty draft returned")
    return draft


CLOSING_RICA = ("rica ederim", "rica ederiz")
CLOSING_ARZ = ("arz ederim", "arz ederiz")
CLOSING_ARZ_RICA = ("arz ve rica ederim", "arz ve rica ederiz")


def validate_draft(draft: str, user_data: dict, assistant_data: dict) -> list:
    problems = []
    word_count = len(draft.split())
    if word_count < 180:
        problems.append(f"too_short({word_count}w)")

    sender_type = (user_data.get("sender_type") or "").upper()
    lower = draft.lower()
    has_rica = any(p in lower for p in CLOSING_RICA)
    has_arz = any(p in lower for p in CLOSING_ARZ)
    has_arz_rica_combo = any(p in lower for p in CLOSING_ARZ_RICA)
    if sender_type in ("VATANDAS", "OZEL_KURULUS"):
        if not has_rica or has_arz:
            problems.append("wrong_closing_expected_rica")
    elif sender_type == "KAMU_KURUMU":
        if not has_arz_rica_combo:
            problems.append("wrong_closing_expected_arz_and_rica")
    else:
        if not has_arz or has_rica:
            problems.append("wrong_closing_expected_arz")

    legislation = user_data.get("selected_legislation") or []
    if legislation:
        cited = False
        for leg in legislation:
            law_name = (leg.get("law_name") or "")
            law_number_match = re.search(r"\d+", law_name)
            article = (leg.get("article") or "")
            article_num_match = re.search(r"\d+", article)
            law_hit = law_number_match and law_number_match.group(0) in draft
            article_hit = article_num_match and article_num_match.group(0) in draft
            if law_hit or article_hit:
                cited = True
                break
        if not cited:
            problems.append("no_legislation_citation_found")

    return problems


def process_record(api_key: str, fname: str, idx: int, rec: dict, max_attempts: int = 3):
    msgs = {m["role"]: m for m in rec["messages"]}
    user_content = msgs["user"]["content"]
    assistant_content = msgs["assistant"]["content"]
    user_data = json.loads(user_content)
    assistant_data = json.loads(assistant_content)

    legislation = user_data.get("selected_legislation") or []

    if "responsible_unit" in assistant_data and "target_unit" not in assistant_data:
        assistant_data["target_unit"] = assistant_data.pop("responsible_unit")

    msgs["system"]["content"] = CANONICAL_SYSTEM_PROMPT

    if not legislation:
        msgs["assistant"]["content"] = json.dumps(assistant_data, ensure_ascii=False)
        rec["messages"] = [msgs["system"], msgs["user"], msgs["assistant"]]
        return rec, "skipped_no_legislation", []

    uid = record_uid(fname, idx, user_content)
    cpath = cache_path(uid)
    if os.path.exists(cpath):
        with open(cpath, encoding="utf-8") as f:
            cached = json.load(f)
        assistant_data["draft"] = cached["draft"]
        msgs["assistant"]["content"] = json.dumps(assistant_data, ensure_ascii=False)
        rec["messages"] = [msgs["system"], msgs["user"], msgs["assistant"]]
        return rec, "cached", cached.get("problems", [])

    if STOP_EVENT.is_set():
        msgs["assistant"]["content"] = json.dumps(assistant_data, ensure_ascii=False)
        rec["messages"] = [msgs["system"], msgs["user"], msgs["assistant"]]
        return rec, "failed", ["skipped_insufficient_balance"]

    user_prompt = build_user_prompt(user_data, assistant_data)
    last_problems = []
    last_draft = None
    for attempt in range(1, max_attempts + 1):
        try:
            prompt = user_prompt
            if attempt > 1 and last_problems:
                prompt += (
                    "\n\nONCEKI DENEMEDE SU SORUNLAR TESPIT EDILDI, DUZELT: "
                    + ", ".join(last_problems)
                )
            draft = call_deepseek(api_key, prompt)
            problems = validate_draft(draft, user_data, assistant_data)
            last_draft, last_problems = draft, problems
            if not problems:
                break
        except InsufficientBalanceError as e:
            STOP_EVENT.set()
            last_problems = [f"insufficient_balance:{e}"]
            break
        except Exception as e:
            last_problems = [f"api_error:{e}"]
            time.sleep(min(2 ** attempt, 10))

    if last_draft is None:
        msgs["assistant"]["content"] = json.dumps(assistant_data, ensure_ascii=False)
        rec["messages"] = [msgs["system"], msgs["user"], msgs["assistant"]]
        return rec, "failed", last_problems

    with open(cpath, "w", encoding="utf-8") as f:
        json.dump({"draft": last_draft, "problems": last_problems}, f, ensure_ascii=False)

    assistant_data["draft"] = last_draft
    msgs["assistant"]["content"] = json.dumps(assistant_data, ensure_ascii=False)
    rec["messages"] = [msgs["system"], msgs["user"], msgs["assistant"]]
    status = "regenerated" if not last_problems else "regenerated_with_warnings"
    return rec, status, last_problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Test icin ilk N kaydi isle")
    ap.add_argument("--files", type=str, default="train,val,test")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("HATA: DEEPSEEK_API_KEY ortam degiskeni tanimli degil.", file=sys.stderr)
        sys.exit(1)

    targets = {
        "train": "qwen2_train.jsonl",
        "val": "qwen2_val.jsonl",
        "test": "qwen2_test.jsonl",
    }
    selected = [targets[k] for k in args.files.split(",") if k.strip() in targets]

    overall_stats = {}

    for fname in selected:
        path = os.path.join(HERE, fname)
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        if args.limit:
            lines = lines[: args.limit]

        records = [json.loads(l) for l in lines]
        stats = {"regenerated": 0, "regenerated_with_warnings": 0, "cached": 0, "skipped_no_legislation": 0, "failed": 0}
        warn_log = []
        results = [None] * len(records)

        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(process_record, api_key, fname, idx, rec): idx
                for idx, rec in enumerate(records)
            }
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                idx = futures[fut]
                try:
                    rec, status, problems = fut.result()
                except Exception as e:
                    rec, status, problems = records[idx], "failed", [f"exception:{e}"]
                results[idx] = rec
                stats[status] = stats.get(status, 0) + 1
                if problems:
                    warn_log.append({"idx": idx, "status": status, "problems": problems})
                done += 1
                if done % 50 == 0 or done == len(records):
                    elapsed = time.time() - t0
                    print(f"[{fname}] {done}/{len(records)} islendi ({elapsed:.0f}s) stats={stats}", flush=True)

        out_path = os.path.join(HERE, fname.replace(".jsonl", "_updated.jsonl"))
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        warn_path = os.path.join(HERE, fname.replace(".jsonl", "_warnings.json"))
        with open(warn_path, "w", encoding="utf-8") as f:
            json.dump(warn_log, f, ensure_ascii=False, indent=2)

        overall_stats[fname] = stats
        print(f"== {fname} tamamlandi: {stats} -> {out_path}", flush=True)
        print(f"   uyari/hata kayitlari: {len(warn_log)} -> {warn_path}", flush=True)

    print("TUM ISLER TAMAMLANDI:", json.dumps(overall_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
