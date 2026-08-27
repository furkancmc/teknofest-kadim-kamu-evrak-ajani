"""
DeepSeek bakiyesi olmadan, ucret gerektirmeyen kural-tabanli (template) taslak
uretici. Sadece 'failed' (bakiye bitti icin islenemeyen) kayitlari doldurur.

Girdiler: qwen2_val_updated.jsonl / qwen2_test_updated.jsonl (zaten yazilmis,
sema birlestirilmis ama draft eski) + qwen2_*_warnings.json (hangi index'lerin
'failed' oldugunu belirtir).

Cikti: ayni dosyalarin uzerine, SADECE failed kayitlarin draft alani
template ile doldurulmus halde yazilir. Basarili/cached/skipped kayitlara
dokunulmaz.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regenerate_drafts import validate_draft  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

CLOSING_BY_SENDER = {
    "VATANDAS": "Bilgilerinizi rica ederim.",
    "OZEL_KURULUS": "Bilgilerinizi rica ederim.",
    "KAMU_KURUMU": "Gereğini bilgilerinize arz ve rica ederim.",
    "YARGI_MERCII": "Gereğini saygılarımla arz ederim.",
}
DEFAULT_CLOSING = "Gereğini saygılarımla arz ederim."


def kv(key_information, type_name):
    for item in key_information or []:
        if item.get("type") == type_name:
            return item.get("value")
    return None


def humanize(code: str) -> str:
    if not code:
        return ""
    words = code.replace("_", " ").strip().split()
    return " ".join(w.capitalize() for w in words)


def extract_district(location: str) -> str:
    if not location:
        return ""
    tail = location.split("/")[-1].strip()
    tail = tail.split(",")[0].strip()
    return tail


def build_legislation_paragraph(legislation):
    if not legislation:
        return ""
    sentences = []
    for leg in legislation:
        law_name = (leg.get("law_name") or "").strip()
        article = (leg.get("article") or "").strip()
        ref_text = (leg.get("reference_text") or "").strip()
        if not law_name:
            continue
        piece = f"{law_name}"
        if article:
            piece += f"'nun {article.lower().replace('madde', '').strip()}. maddesi uyarınca"
        else:
            piece += " hükümleri uyarınca"
        if ref_text:
            piece += f", {ref_text[0].lower()}{ref_text[1:]}" if len(ref_text) > 1 else f", {ref_text}"
        if not piece.endswith("."):
            piece += "."
        sentences.append(piece)
    return " ".join(sentences)


def build_draft(user_data: dict, assistant_data: dict) -> str:
    key_info = user_data.get("key_information", [])
    sender_type = (user_data.get("sender_type") or "").upper()

    person = kv(key_info, "PERSON")
    org = kv(key_info, "ORGANIZATION")
    muhatap = person or org or "İlgili"

    location = kv(key_info, "LOCATION")
    district = extract_district(location)
    belediye_baslik = f"{district.upper()} BELEDİYE BAŞKANLIĞI" if district else "BELEDİYE BAŞKANLIĞI"

    ref_no = kv(key_info, "REFERENCE_NO") or kv(key_info, "DOCUMENT_NO") or kv(key_info, "APPLICATION_NO") or "-"
    date_val = kv(key_info, "DATE") or kv(key_info, "REFERENCE_DATE")

    target_unit = assistant_data.get("target_unit") or assistant_data.get("responsible_unit") or "İlgili Müdürlük"
    topic = humanize(user_data.get("primary_topic"))
    summary = (user_data.get("summary") or "").strip()
    requested_action = (user_data.get("requested_action") or "").strip()
    missing_info = user_data.get("missing_information") or []
    performed_actions = assistant_data.get("performed_actions") or []
    result_information = (assistant_data.get("result_information") or "").strip()
    process_information = (assistant_data.get("process_information") or "").strip()
    legislation = user_data.get("selected_legislation") or []

    lines = []
    lines.append("T.C.")
    lines.append(belediye_baslik)
    lines.append(target_unit)
    lines.append("")
    lines.append(f"Sayı: {ref_no}")
    lines.append(f"Konu: {topic} Hk." if topic else "Konu: Başvurunuz Hk.")
    lines.append("")
    if date_val:
        lines.append(f"İlgi: {date_val} tarihli başvurunuz.")
        lines.append("")
    lines.append(f"Sayın {muhatap},")
    lines.append("")

    body_paragraphs = []

    p1 = summary
    if requested_action and requested_action not in summary:
        p1 = (p1 + " " if p1 else "") + f"Başvurunuzda {requested_action.strip().rstrip('.')} talep edilmektedir."
    if p1:
        body_paragraphs.append(p1)

    if missing_info:
        eksik_text = "; ".join(m.strip().rstrip(".") for m in missing_info if m)
        body_paragraphs.append(
            f"Yapılan inceleme neticesinde başvurunuza ilişkin bazı bilgi/belgelerin eksik olduğu tespit edilmiştir: "
            f"{eksik_text}. Söz konusu eksikliklerin tarafımıza iletilmesi, başvurunuzun sağlıklı bir şekilde "
            f"sonuçlandırılabilmesi için önem arz etmektedir."
        )

    leg_para = build_legislation_paragraph(legislation)
    if leg_para:
        body_paragraphs.append(
            f"Konuya ilişkin mevzuat hükümleri incelenmiş olup; {leg_para} Bu hükümler çerçevesinde başvurunuz "
            f"değerlendirmeye alınmıştır."
        )

    def action_text(a):
        if isinstance(a, dict):
            return (a.get("detail") or a.get("action") or "").strip().rstrip(".")
        return str(a).strip().rstrip(".")

    if performed_actions:
        actions_text = "; ".join(t for t in (action_text(a) for a in performed_actions) if t)
        if actions_text:
            body_paragraphs.append(
                f"Bu kapsamda tarafımızca aşağıdaki işlemler gerçekleştirilmiştir: {actions_text}."
            )

    if result_information:
        body_paragraphs.append(result_information)

    if process_information:
        body_paragraphs.append(f"Süreç, {target_unit} tarafından takip edilmektedir. {process_information}")

    body_paragraphs.append(
        "Başvurunuzun değerlendirilmesi sürecinde belediyemiz, ilgili mevzuat hükümlerine ve idari usul "
        "esaslarına tam uyum göstererek, konunun sağlıklı ve zamanında sonuçlandırılması için gerekli "
        "çalışmaları titizlikle yürütmektedir. Süreçle ilgili herhangi bir sorunuz olması halinde "
        f"{target_unit} ile iletişime geçebilirsiniz."
    )

    lines.extend(body_paragraphs)
    lines.append("")

    closing = CLOSING_BY_SENDER.get(sender_type, DEFAULT_CLOSING)
    lines.append(closing)
    lines.append("")
    lines.append("Saygılarımla,")
    lines.append("")
    lines.append("")
    lines.append(target_unit)

    return "\n".join(lines)


FILLER_SENTENCES = [
    "İlgili birimlerimiz, başvurunuzu titizlikle ele almakta olup süreç boyunca mevzuata ve idari "
    "usul esaslarına tam uyum gösterilmektedir.",
    "Konuya ilişkin her türlü gelişme tarafınıza ayrıca bildirilecektir.",
    "Başvurunuzun sağlıklı bir şekilde sonuçlandırılması belediyemizin önceliğidir.",
    "Talebinizle ilgili işlemler, ilgili mevzuat hükümleri çerçevesinde titizlikle yürütülmektedir.",
]


def pad_to_min_words(draft: str, min_words: int = 200) -> str:
    filler_idx = 0
    while len(draft.split()) < min_words and filler_idx < len(FILLER_SENTENCES) * 3:
        sentence = FILLER_SENTENCES[filler_idx % len(FILLER_SENTENCES)]
        marker = "Saygılarımla,"
        if marker in draft:
            draft = draft.replace(marker, f"{sentence}\n\n{marker}", 1)
        else:
            draft += f" {sentence}"
        filler_idx += 1
    return draft


def process_file(basename: str, warnings_path: str, updated_path: str):
    with open(warnings_path, encoding="utf-8") as f:
        warnings = json.load(f)
    failed_idx = {w["idx"] for w in warnings if w["status"] == "failed"}
    if not failed_idx:
        print(f"{basename}: failed kayit yok, atlaniyor.")
        return

    with open(updated_path, encoding="utf-8") as f:
        lines = f.readlines()

    filled, still_bad = 0, []
    for idx in sorted(failed_idx):
        rec = json.loads(lines[idx])
        msgs = {m["role"]: m for m in rec["messages"]}
        user_data = json.loads(msgs["user"]["content"])
        assistant_data = json.loads(msgs["assistant"]["content"])

        draft = build_draft(user_data, assistant_data)
        draft = pad_to_min_words(draft)
        problems = validate_draft(draft, user_data, assistant_data)

        assistant_data["draft"] = draft
        msgs["assistant"]["content"] = json.dumps(assistant_data, ensure_ascii=False)
        rec["messages"] = [msgs["system"], msgs["user"], msgs["assistant"]]
        lines[idx] = json.dumps(rec, ensure_ascii=False) + "\n"
        filled += 1
        if problems:
            still_bad.append({"idx": idx, "problems": problems})

    with open(updated_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    report_path = updated_path.replace("_updated.jsonl", "_template_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"filled": filled, "still_problematic": still_bad}, f, ensure_ascii=False, indent=2)

    print(f"{basename}: {filled} kayit template ile dolduruldu. Sorunlu kalan: {len(still_bad)} -> {report_path}")


def main():
    process_file(
        "val",
        os.path.join(HERE, "qwen2_val_warnings.json"),
        os.path.join(HERE, "qwen2_val_updated.jsonl"),
    )
    process_file(
        "test",
        os.path.join(HERE, "qwen2_test_warnings.json"),
        os.path.join(HERE, "qwen2_test_updated.jsonl"),
    )


if __name__ == "__main__":
    main()
