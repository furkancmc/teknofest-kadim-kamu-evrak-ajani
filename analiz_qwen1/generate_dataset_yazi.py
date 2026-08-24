"""
Yazi Isleri Muduurlugu - Sentetik Egitim Verisi Uretici (DeepSeek API)
=========================================================================

DeepSeek API'sini (OpenAI-uyumlu chat completions endpoint'i) kullanarak
belediyelerin Yazi Isleri Muduurlugu kapsaminda 70 adet TEMIZ (eksik bilgi
icermeyen) sentetik Turkce evrak + etiket kaydi uretir.

Onceki scriptlerle (Fen Isleri / Hukuk Isleri / Mali Hizmetler / Park ve
Bahceler / Sosyal Yardim / Temizlik Isleri / Veteriner Isleri) ayni mimari
(konu bazli, thread havuzu ile paralel uretim, semaya gore dogrulama, ara
kayit).

Kullanim:
    pip install -r requirements.txt
    copy .env.example .env   # anahtari .env icine yazin; .env commit edilmez
    python generate_dataset_yazi.py

Cikti:
    dataset_yazi.jsonl
    dataset_yazi.json
"""

import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from openai import OpenAI
from _env import deepseek_api_key

# ----------------------------------------------------------------------------
# Ayarlar
# ----------------------------------------------------------------------------

API_KEY = deepseek_api_key()
BASE_URL = "https://api.deepseek.com"
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

OUTPUT_JSONL = "dataset_yazi.jsonl"
OUTPUT_JSON = "dataset_yazi.json"

BATCH_SIZE = 3
MAX_TOTAL_ATTEMPTS_PER_TOPIC = 20
REQUEST_TIMEOUT = 120

TOPIC_WORKERS = 5
INTRA_TOPIC_PARALLEL_BATCHES = 2

print_lock = threading.Lock()
save_lock = threading.Lock()


def log(msg: str) -> None:
    with print_lock:
        print(msg, flush=True)


# Konu basina hedef sayi (bir sonraki uretim/top-up calistirmasi icin).
# NOT: RESMI_YAZISMA_ISLEMLERI zayif sinif (toplam 17 kayit) -> hedef
# yukseltildi.
TOPIC_TARGETS: dict[str, int] = {
    "DILEKCE_BILGI_EDINME": 14,
    "RESMI_YAZISMA_ISLEMLERI": 30,
    "GELEN_GIDEN_EVRAK": 14,
    "TEBLIGAT_ISLEMLERI": 14,
    "ELEKTRONIK_BELGE_IMZA_TEBLIGAT": 14,
}

ALLOWED_DOCUMENT_TYPES = {
    "TALEP_DILEKCE_BASVURUSU",
    "SIKAYET_IHBAR_BASVURUSU",
    "BILGI_TALEBI",
    "ITIRAZ_IDARI_BASVURU",
    "KURUMLAR_ARASI_RESMI_YAZI",
}

ALLOWED_SENDER_TYPES = {
    "VATANDAS",
    "KAMU_KURUMU",
    "OZEL_KURULUS",
    "YARGI_MERCII",
}

# Not: bu batch icin AMOUNT / HOUSEHOLD_INFORMATION / INCOME_INFORMATION yok.
ALLOWED_KEY_INFO_TYPES = {
    "PERSON",
    "ORGANIZATION",
    "DATE",
    "REFERENCE_DATE",
    "DOCUMENT_NO",
    "REFERENCE_NO",
    "APPLICATION_NO",
    "CASE_NO",
    "LOCATION",
    "CONTACT",
    "DEADLINE",
    "EVENT_DETAIL",
    "OTHER",
}

FORBIDDEN_LABEL_KEYS = {
    "priority",
    "priority_reason",
    "primary_unit",
    "secondary_unit",
    "expected_sources",
    "rag_queries",
    "routing_evidence",
    "questions_to_answer",
    "keywords",
    "related_legislation",
    "legal_sources",
    "recommended_laws",
}

TOPIC_HINTS = {
    "DILEKCE_BILGI_EDINME": (
        "Vatandas dilekcesinin kayda alinmasi, dilekce durumunun "
        "sorulmasi, bilgi edinme basvurusu, onceki basvurunun durumunun "
        "ogrenilmesi, basvuru numarasi hakkinda bilgi talebi, dilekcenin "
        "ilgili birime iletilip iletilmedigi, basvuru sonucunun ne zaman "
        "bildirilecegi, bilgi edinme basvurusuna iliskin kurum "
        "yazismalari. (DIKKAT: dilekcenin asil icerigi teknik olarak "
        "baska bir muduurluge aitse, sirf dilekce oldugu icin Yazi "
        "Isleri konusu YAPMA; bu topic yalnizca dilekce/bilgi edinme "
        "SUeRECI ana konu oldugunda kullanilir.)"
    ),
    "RESMI_YAZISMA_ISLEMLERI": (
        "Resmi yazi formati hakkinda bilgi talebi, ust yazi "
        "duzenlenmesi, kurumlar arasi resmi yazisma sureci, yazinin "
        "uygun usulle hazirlanmasi/iletilmesi, ilgi/ek/dagitim/sayi "
        "bilgilerinin yazismadaki kullanimi, resmi yazinin hangi "
        "yontemle gonderilecegi, belediye biriminden yazisma usulu "
        "hakkinda gorus/bilgi talebi."
    ),
    "GELEN_GIDEN_EVRAK": (
        "Gelen evrakin kayit durumu, evrak kayit numarasi sorgulama, "
        "evrakin hangi tarihte alindiginin sorulmasi, evrakin hangi "
        "birime sevk edildiginin sorulmasi, giden evrakin gonderim "
        "durumu, kurumdan cikan yazinin kayit bilgisi, yanlis birime "
        "sevk edilen evrakin duzeltilmesi, evrak havale/kayit sureci "
        "hakkinda bilgi talebi. (DIKKAT: ana konu evrak kayit/havale/"
        "sevk surecidir; belgenin esas icerigi baska uzmanlik birimine "
        "aitse onu Yazi Isleri konusu YAPMA.)"
    ),
    "TEBLIGAT_ISLEMLERI": (
        "Fiziki tebligatin gonderilip gonderilmedigi, tebligatin teslim "
        "durumu, tebligat tarihinin sorulmasi, tebligatin iade edilmesi, "
        "adres nedeniyle teslim edilemeyen tebligat, tebligat kayit "
        "bilgisi, kurum ici tebligat sureci hakkinda bilgi, tebligat "
        "evrakinin ilgili birime iletilmesi."
    ),
    "ELEKTRONIK_BELGE_IMZA_TEBLIGAT": (
        "Elektronik imzali belge dogrulama, e-imza kullanimina iliskin "
        "islem, elektronik belge gonderimi, elektronik tebligat durumu, "
        "elektronik belge kayit/sistem sorunu, elektronik yazismanin "
        "dogrulanmasi, e-tebligat adresi veya gonderim durumuna iliskin "
        "bilgi, kurumlar arasi elektronik belge islemleri."
    ),
}

REQUIRED_INFO_BY_TOPIC = {
    "DILEKCE_BILGI_EDINME": (
        "basvuru sahibinin/talep eden kurumun kim oldugu, ilgili basvuru/"
        "dilekce veya bilgi talebinin konusu, gerekiyorsa basvuruyu "
        "tanimlayan tarih/numara, istenen bilgi veya islem. (T.C. kimlik "
        "numarasi her vatandas basvurusunda zorunlu degildir.)"
    ),
    "RESMI_YAZISMA_ISLEMLERI": (
        "hangi yazi/yazisma islemi hakkinda talep oldugu, ilgili kurum/"
        "birim veya yazi, karsilasilan durum veya talep, istenen islem/bilgi"
    ),
    "GELEN_GIDEN_EVRAK": (
        "hangi evrakin kastedildigi, kayit/sevk/gonderim surecinin hangi "
        "kisminin soruldugu, evraki tanimlayan yeterli bilgi, istenen "
        "islem veya bilgi"
    ),
    "TEBLIGAT_ISLEMLERI": (
        "hangi tebligatin soz konusu oldugu, ilgili kisi/kurum, tebligat "
        "surecindeki durum, istenen bilgi veya islem"
    ),
    "ELEKTRONIK_BELGE_IMZA_TEBLIGAT": (
        "hangi elektronik belge/e-imza/e-tebligat isleminin soz konusu "
        "oldugu, ilgili kisi/kurum veya belge, sorun/talep, istenen islem"
    ),
}

DIVERSITY_SCENARIOS = [
    "vatandas dilekcesi",
    "bilgi edinme basvurusu",
    "kamu kurumundan gelen yazi",
    "belediye ic biriminden yazisma usulu talebi",
    "evrak kayit numarasi sorgusu",
    "sevk/havale durumu talebi",
    "giden evrak gonderim bilgisi",
    "fiziki tebligat durumu",
    "elektronik tebligat",
    "e-imza dogrulama",
    "elektronik belge iletim sorunu",
    "resmi yazi usulu hakkinda bilgi",
]

SYSTEM_PROMPT = """Sen, belediyelerde kullanilan resmi evraklara benzer SENTETIK egitim verileri ureten bir veri uretim uzmanisin.
Amacin, Yazi Isleri Muduurlugu kapsamina giren gercekci Turkce evraklar uretmek ve her evrak icin dogru etiketleri olusturmaktir.

KURALLAR:
1. Uretilen butun belgeler TEMIZ ve YETERLI BILGI ICEREN belgeler olmalidir. "missing_information" HER ZAMAN bos dizi [] olmalidir.
   Ancak bunu yapay bicimde saglama: belge metni, konusu acisindan gerekli temel bilgileri GERCEKTEN icermelidir.
   Kontrol et: hangi evrak/basvuru kastediliyor? islem sureci yeterince tanimli mi? bilgi edinme talebinde neyin
   soruldugu belli mi? tebligat isleminde hangi tebligatin kastedildigi belli mi? elektronik belge isleminde
   sorun/talep acik mi? gonderenin ne istedigi belli mi? Kritik bir bilgi eksikse metni tamamla, sonra [] yaz.
2. Tum kisi, kurum, adres, tarih, belge numarasi, basvuru numarasi ve olaylar KURGUSAL olmalidir. Gercek kisi/kurum/ozel olay kullanma.
3. document_type YALNIZCA su degerlerden biri olmali: TALEP_DILEKCE_BASVURUSU, SIKAYET_IHBAR_BASVURUSU, BILGI_TALEBI, ITIRAZ_IDARI_BASVURU, KURUMLAR_ARASI_RESMI_YAZI
   AYIRT ET (confusion matrix'te en cok karisan document_type ciftleri icin karar sirasi):
   a) Gonderen bir idari karar/islem/cezaya karsi resmi olarak itiraz ediyorsa -> ITIRAZ_IDARI_BASVURU.
   b) Gonderen bir KURAL IHLALI, UCUNCU TARAF KUSURU veya TEHLIKE bildiriyor ve bunun
      incelenmesini/durdurulmasini/cezalandirilmasini istiyorsa -> SIKAYET_IHBAR_BASVURUSU. Ayirt
      edici soru: "birisi/bir sey KURALA AYKIRI mi davraniyor, ya da tehlike/zarar mi olusturuyor?"
      Sikayetin ODAGI bir davranis/durumun YANLISLIGI olmali, sadece bir eksiklik degil.
   c) Gonderen (kural ihlali bildirmeden) somut bir ISLEM/HIZMET yapilmasini istiyorsa (yeni
      basvuru, talep, kurulum, onarim vb.) -> TALEP_DILEKCE_BASVURUSU. Bu, mevcut bir EKSIKLIK
      veya IHTIYAC bildirimi olabilir (or. "yol bozuk, onarilsin") -- eksiklik/ihtiyac bildirimi
      TEK BASINA sikayet YAPMAZ; kimse KURALA AYKIRI davranmiyorsa ve sadece hizmet isteniyorsa
      bu kategori kullanilir.
   d) Gonderen (bir islem/hizmet istemeden, ihlal bildirmeden) yalnizca bir konu/kayit/surec
      hakkinda BILGI/ACIKLAMA istiyorsa -> BILGI_TALEBI. Bu, gonderen KAMU_KURUMU olsa bile
      gecerlidir (bir kurum baska bir vatandas/kurum hakkinda bilgi soruyorsa yine BILGI_TALEBI
      olabilir).
   e) KURUMLAR_ARASI_RESMI_YAZI, YALNIZCA iki kurum arasindaki rutin resmi yazismalar icindir: bilgi
      talebi, talep veya sikayet CERCEVESINE GIRMEYEN durum bildirimi, evrak/karar iletimi, gorus
      sorma disinda bir kurumun baska kuruma yazdigi yazi. sender_type KAMU_KURUMU/YARGI_MERCII
      olmasi TEK BASINA yeterli degildir -- yazinin ICERIGI (a)-(d) kaliplarindan birine net
      sekilde giriyorsa o kural kullanilmali, sirf gonderen kurum diye KURUMLAR_ARASI_RESMI_YAZI
      YAZILMAMALI.
   (Yazi Isleri'nde ozellikle TALEP_DILEKCE_BASVURUSU, BILGI_TALEBI, KURUMLAR_ARASI_RESMI_YAZI daha sik olabilir, ama zorunlu degil.)
4. sender_type YALNIZCA su degerlerden biri olmali: VATANDAS, KAMU_KURUMU, OZEL_KURULUS, YARGI_MERCII. VATANDAS ve
   KAMU_KURUMU agirlikli kullan.
5. primary_topic sana her istekte belirtilecek; TAM OLARAK verilen degeri kullan, degistirme veya yeni kategori uretme.
6. key_information yalnizca belgede ACIKCA bulunan ve islemin anlasilmasi acisindan onemli bilgilerden olusmali.
   type degerleri YALNIZCA sunlar olabilir: PERSON, ORGANIZATION, DATE, REFERENCE_DATE, DOCUMENT_NO, REFERENCE_NO, APPLICATION_NO, CASE_NO, LOCATION, CONTACT, DEADLINE, EVENT_DETAIL, OTHER
   (Bu batch'te AMOUNT kullanma.)
   - Belgede olmayan bilgiyi ASLA ekleme. Bulunmayan bir bilgiyi "belirtilmemis" diye yazma.
   - Her tarih/isim/numarayi otomatik cikarma, yalnizca islem acisindan gercekten onemli olanlari sec.
   - Ayni bilgiyi tekrar etme. Imza sahibinin adi asil islem kisisi degilse gereksiz PERSON ekleme.
   - Basvuru numarasi varsa APPLICATION_NO, ana yazinin sayisi DOCUMENT_NO, ilgi yazisi numarasi REFERENCE_NO,
     ilgi yazisi tarihi REFERENCE_DATE olabilir. Tebligat/elektronik belgeye ait ozel numara varsa uygun sekilde
     DOCUMENT_NO veya REFERENCE_NO kullan.
7. requested_action kisa ama somut olmali: gonderenin Yazi Isleri Muduurlugunden/belediyeden TAM OLARAK ne
   istedigini yaz. "geregi yapilmasi" gibi genel ifadeler YASAK. Belgede istenmeyen yeni bir islem uretme. Yapilmamis
   islemi yapilmis gibi gosterme.
8. summary: 1-2 cumle, kisa, tarafsiz, belgeye sadik, yeni bilgi icermeyen bir ozet.
9. MEVZUAT UYDURMA. Kanun, yonetmelik veya madde numarasi uydurma.
10. Su alanlari KESINLIKLE labels icine ekleme: priority, priority_reason, primary_unit, secondary_unit,
    expected_sources, rag_queries, routing_evidence, questions_to_answer, keywords, related_legislation,
    legal_sources, recommended_laws. labels SADECE su 7 alani icermeli: document_type, sender_type, primary_topic,
    requested_action, key_information, missing_information, summary.
11. BIRIM SINIRI: Yazi Isleri Muduurlugu'nu diger uzman birimlerle KESINLIKLE karistirma. Bir dilekcenin ASIL
    TALEBI konuyu belirler, dilekce OLMASI onu Yazi Isleri konusu yapmaz. Ornek: "Yolumdaki cukurun onarilmasini
    istiyorum" -> bu Fen Isleri konusudur, DILEKCE_BILGI_EDINME DEGILDIR. Ama "12 Agustos'ta verdigim yol onarim
    dilekcemin hangi birime sevk edildigini ogrenmek istiyorum" -> bu Yazi Isleri (DILEKCE_BILGI_EDINME veya
    GELEN_GIDEN_EVRAK) kapsamina girer, cunku asil talep SUeRECIN kendisidir.
12. TEBLIGAT SINIRI: Hukuk Isleri ile Yazi Isleri tebligat senaryolarini karistirma. Yazi Isleri odagi: tebligatin
    kayit/gonderim/teslim sureci, tebligat evrakinin idari takibi, elektronik tebligat sistemi. Hukuk Isleri odagi
    (bu dataset'te KULLANMA): tebligatin hukuki gecerliligi, dava/icra surecindeki tebligat uyusmazligi, hukuki
    sonuc doguran tebligat degerlendirmesi. Yalnizca Yazi Isleri'nin idari/yazisma boyutunun baskin oldugu
    ornekleri uret.
13. Belgeleri birbirinin sablon kopyasi yapma: isim, kurum, tarih, uslup, uzunluk ve resmilik seviyesini her
    seferinde gercekten farklilastir. Her GELEN_GIDEN_EVRAK ornegini ayni "hangi birime sevk edildi?" kalibiyla
    uretme. Ayni senaryoyu yalnizca tarih veya numara degistirerek tekrar uretme.

CIKTI FORMATI:
Sadece gecerli bir JSON dondur (baska hicbir metin, aciklama veya markdown code fence olmadan).
JSON su sekilde bir NESNE olmali: {"records": [ ... ]}
"records" bir dizi olmali ve her eleman tam olarak su sekilde olmali:
{
  "document_text": "Uretilen evrakin tam metni (gercekci bicimlendirme, hitap, tarih, imza vb. icerebilir)",
  "labels": {
    "document_type": "...",
    "sender_type": "...",
    "primary_topic": "...",
    "requested_action": "...",
    "key_information": [ {"type": "...", "value": "..."} , ... ],
    "missing_information": [],
    "summary": "..."
  }
}
"""


def build_user_prompt(topic: str, count: int, used_summaries: list[str]) -> str:
    scenario_pool = DIVERSITY_SCENARIOS.copy()
    random.shuffle(scenario_pool)
    suggested = scenario_pool[:count]

    avoid_block = ""
    if used_summaries:
        recent = used_summaries[-15:]
        avoid_block = (
            "\nDaha once bu konuda uretilen bazi ozetler (bunlarla ayni senaryoyu, kisi/kurum adini, "
            "evrak/basvuru kurgusunu veya cumle kalibini TEKRARLAMA, farkli ve yeni senaryolar uret):\n- "
            + "\n- ".join(recent)
        )

    return f"""Simdi primary_topic = "{topic}" icin TAM OLARAK {count} adet TEMIZ belge kaydi uret.

Bu konunun kapsami ve alt senaryolari: {TOPIC_HINTS[topic]}

Bu konu icin belgede mutlaka anlasilir olmasi gereken temel bilgiler: {REQUIRED_INFO_BY_TOPIC[topic]}

Uretecegin {count} belge icin onerilen senaryo cesitliligi (birebir uyma zorunlulugu yok, ilham amacli,
istersen farkli sender_type/document_type kombinasyonlari da kullan, gercekci kombinasyonlar uret):
- {chr(10).join(f"{i+1}. {s}" for i, s in enumerate(suggested))}
{avoid_block}

Unutma: her belge farkli kisi/kurum adlari, farkli basvuru/evrak/tebligat numaralari, farkli tarih icermeli, ve
mumkun oldugunca farkli ALT SENARYO kullanmali (ayni topic icinde tek bir alt senaryoyu tekrar tekrar kullanma).
missing_information her zaman [] olmali ama belge gercekten yeterli bilgi icermeli. Belgenin ASIL TALEBININ
gercekten Yazi Isleri'nin idari/yazisma surecine ait oldugundan emin ol (sirf dilekce/tebligat gectigi icin
otomatik olarak Yazi Isleri konusu yapma). Mevzuat adi uydurma.

Sadece istenen JSON nesnesini dondur: {{"records": [...]}} (tam olarak {count} eleman)."""


def call_deepseek(client: OpenAI, topic: str, count: int, used_summaries: list[str]) -> list[dict[str, Any]]:
    user_prompt = build_user_prompt(topic, count, used_summaries)

    log(f"  [{topic}] ... istek gonderildi ({count} kayit), yanit bekleniyor (en fazla {REQUEST_TIMEOUT}sn)...")
    start = time.monotonic()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=1.1,
        max_tokens=8000,
        timeout=REQUEST_TIMEOUT,
    )
    elapsed = time.monotonic() - start
    log(f"  [{topic}] ... yanit alindi ({elapsed:.1f}sn)")

    content = response.choices[0].message.content
    data = json.loads(content)
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError("Yanit 'records' dizisi icermiyor.")
    return records


def validate_record(record: dict[str, Any], topic: str) -> tuple[bool, str]:
    if not isinstance(record, dict):
        return False, "kayit bir nesne degil"

    text = record.get("document_text")
    labels = record.get("labels")
    if not isinstance(text, str) or len(text.strip()) < 40:
        return False, "document_text eksik veya cok kisa"
    if not isinstance(labels, dict):
        return False, "labels eksik"

    forbidden_present = FORBIDDEN_LABEL_KEYS & set(labels.keys())
    if forbidden_present:
        return False, f"yasakli alan(lar) bulundu: {forbidden_present}"

    doc_type = labels.get("document_type")
    sender_type = labels.get("sender_type")
    primary_topic = labels.get("primary_topic")
    requested_action = labels.get("requested_action")
    key_info = labels.get("key_information")
    missing_info = labels.get("missing_information")
    summary = labels.get("summary")

    if doc_type not in ALLOWED_DOCUMENT_TYPES:
        return False, f"gecersiz document_type: {doc_type}"
    if sender_type not in ALLOWED_SENDER_TYPES:
        return False, f"gecersiz sender_type: {sender_type}"
    if primary_topic != topic:
        return False, f"primary_topic uyusmuyor: {primary_topic} != {topic}"
    if not isinstance(requested_action, str) or len(requested_action.strip()) < 10:
        return False, "requested_action eksik veya cok kisa"
    if re.fullmatch(r"(?i)\s*geregi(nin)? yap[iı]lmas[iı]\.?\s*", requested_action or ""):
        return False, "requested_action cok genel ('geregi yapilmasi')"
    if not isinstance(key_info, list):
        return False, "key_information dizi degil"
    for item in key_info:
        if not isinstance(item, dict):
            return False, "key_information elemani nesne degil"
        if item.get("type") not in ALLOWED_KEY_INFO_TYPES:
            return False, f"gecersiz key_information type: {item.get('type')}"
        if not isinstance(item.get("value"), str) or not item.get("value").strip():
            return False, "key_information value eksik"
    if missing_info != []:
        return False, f"missing_information bos degil: {missing_info}"
    if not isinstance(summary, str) or len(summary.strip()) < 10:
        return False, "summary eksik veya cok kisa"

    return True, "ok"


def generate_topic(client: OpenAI, topic: str, target: int) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    used_summaries: list[str] = []
    total_attempts = 0

    with ThreadPoolExecutor(max_workers=INTRA_TOPIC_PARALLEL_BATCHES) as pool:
        while len(collected) < target and total_attempts < MAX_TOTAL_ATTEMPTS_PER_TOPIC:
            remaining = target - len(collected)
            n_parallel = min(INTRA_TOPIC_PARALLEL_BATCHES, -(-remaining // BATCH_SIZE))
            futures = []
            for _ in range(max(1, n_parallel)):
                futures.append(pool.submit(call_deepseek, client, topic, BATCH_SIZE, list(used_summaries)))
                total_attempts += 1

            for fut in as_completed(futures):
                try:
                    raw_records = fut.result()
                except Exception as exc:  # noqa: BLE001
                    log(f"  [!] [{topic}] API/parse hatasi ({exc}), atlaniyor...")
                    continue

                for rec in raw_records:
                    if len(collected) >= target:
                        break
                    ok, reason = validate_record(rec, topic)
                    if not ok:
                        log(f"  [x] [{topic}] kayit reddedildi ({reason})")
                        continue
                    collected.append(rec)
                    used_summaries.append(rec["labels"]["summary"])
                    log(f"  [+] [{topic}] {len(collected)}/{target} kayit uretildi")

    if len(collected) < target:
        log(f"  [!] UYARI: {topic} icin sadece {len(collected)}/{target} gecerli kayit uretilebildi.")

    return collected[:target]


def save_records(records: list[dict[str, Any]]) -> None:
    with save_lock:
        with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


def main() -> None:
    if not API_KEY:
        print("HATA: DEEPSEEK_API_KEY tanimli degil.", file=sys.stderr)
        sys.exit(1)

    total_target = sum(TOPIC_TARGETS.values())
    print(f"Toplam hedef kayit: {total_target}")
    print(f"Model: {MODEL}")
    print(f"Paralellik: {TOPIC_WORKERS} konu ayni anda, konu basina {INTRA_TOPIC_PARALLEL_BATCHES} paralel batch")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=REQUEST_TIMEOUT, max_retries=0)

    results_by_topic: dict[str, list[dict[str, Any]]] = {t: [] for t in TOPIC_TARGETS}

    def flush_all() -> None:
        combined: list[dict[str, Any]] = []
        for t in TOPIC_TARGETS:
            combined.extend(results_by_topic[t])
        save_records(combined)

    try:
        with ThreadPoolExecutor(max_workers=TOPIC_WORKERS) as executor:
            future_to_topic = {
                executor.submit(generate_topic, client, topic, target): topic
                for topic, target in TOPIC_TARGETS.items()
            }
            for fut in as_completed(future_to_topic):
                topic = future_to_topic[fut]
                try:
                    results_by_topic[topic] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    log(f"[!] {topic} icin beklenmeyen hata: {exc}")
                flush_all()
                done_total = sum(len(v) for v in results_by_topic.values())
                log(f"=== {topic} tamamlandi -> toplam {done_total}/{total_target} ===")
    except KeyboardInterrupt:
        print("\n[!] Kullanici tarafindan durduruldu. Su ana kadar uretilenler kaydediliyor...")
    finally:
        flush_all()
        final_total = sum(len(v) for v in results_by_topic.values())
        print(f"\nToplam uretilen kayit: {final_total}/{total_target}")
        print(f"Kaydedildi: {OUTPUT_JSONL}, {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
