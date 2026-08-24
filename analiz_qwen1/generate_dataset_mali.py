"""
Mali Hizmetler Muduurlugu - Sentetik Egitim Verisi Uretici (DeepSeek API)
===========================================================================

DeepSeek API'sini (OpenAI-uyumlu chat completions endpoint'i) kullanarak
belediyelerin Mali Hizmetler Muduurlugu kapsaminda 70 adet TEMIZ (eksik
bilgi icermeyen) sentetik Turkce evrak + etiket kaydi uretir.

Fen Isleri / Hukuk Isleri scriptleriyle ayni mimari (konu bazli, thread
havuzu ile paralel uretim, semaya gore dogrulama, ara kayit).

Kullanim:
    pip install -r requirements.txt
    copy .env.example .env   # anahtari .env icine yazin; .env commit edilmez
    python generate_dataset_mali.py

Cikti:
    dataset_mali.jsonl
    dataset_mali.json
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

OUTPUT_JSONL = "dataset_mali.jsonl"
OUTPUT_JSON = "dataset_mali.json"

BATCH_SIZE = 3
MAX_TOTAL_ATTEMPTS_PER_TOPIC = 20
REQUEST_TIMEOUT = 120

TOPIC_WORKERS = 6
INTRA_TOPIC_PARALLEL_BATCHES = 2

print_lock = threading.Lock()
save_lock = threading.Lock()


def log(msg: str) -> None:
    with print_lock:
        print(msg, flush=True)


# Konu basina hedef sayi (bir sonraki uretim/top-up calistirmasi icin).
# NOT (taksonomi birlestirme): KAMU_ALACAGI_TAHSILAT, KAMU_ZARARI_MALI_ISLEMLER
# ile tanim/ornek acisindan buyuk oranda ortustugu (confusion matrix'te en
# cok karisan mali cift) ve hint metinlerinde zaten birbirini tekrar ettigi
# icin (bkz. "kamu zararinin tahsili") tek konuda birlestirildi. Zayif
# siniflar (EMLAK_VERGISI, BELEDIYE_VERGI_HARC) icin hedefler yukseltildi.
TOPIC_TARGETS: dict[str, int] = {
    "EMLAK_VERGISI": 30,
    "BELEDIYE_VERGI_HARC": 30,
    "VERGI_ITIRAZI": 12,
    "BUTCE_MUHASEBE_HARCAMA": 15,
    "KAMU_ZARARI_MALI_ISLEMLER": 20,
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
    "AMOUNT",
    "DEADLINE",
    "EVENT_DETAIL",
    "OTHER",
}

FORBIDDEN_LABEL_KEYS = {
    "expected_sources",
    "related_legislation",
    "legal_sources",
    "recommended_laws",
    "rag_queries",
    "priority",
    "priority_reason",
    "primary_unit",
    "secondary_unit",
    "routing_evidence",
    "keywords",
    "questions_to_answer",
}

TOPIC_HINTS = {
    "EMLAK_VERGISI": (
        "Emlak vergisi borcu, emlak vergisi tahakkuku, tasinmaza iliskin emlak "
        "vergisi bilgisi, mukellefiyet bilgisi, mukerrer veya hatali emlak vergisi "
        "kaydi, emlak vergisi odeme ve borc sorgulama talepleri. AYIRT ET: gonderen "
        "tahakkuk eden TUTARA veya kaydin DOGRULUGUNA itiraz ediyorsa (yanlis "
        "hesaplandigini, kendisine ait olmadigini iddia ediyorsa) VERGI_ITIRAZI "
        "kullan; bu topic sadece bilgi/sorgulama/odeme talepleri icindir, itiraz "
        "icermez."
    ),
    "BELEDIYE_VERGI_HARC": (
        "Belediye vergi ve harclari, ilan ve reklam vergisi, eglence vergisi, "
        "isgal harci, cesitli belediye harc ve ucretleri, tahakkuk veya odeme "
        "bilgisi talepleri. AYIRT ET: konu emlak vergisiyse EMLAK_VERGISI kullan; "
        "tahakkuka itirazsa VERGI_ITIRAZI kullan; konu belediyenin kendi ic butce/"
        "odenek/hakedis islemiyse (vergi mukellefinden degil, ic birimden veya "
        "yukleniciden gelen) BUTCE_MUHASEBE_HARCAMA kullan."
    ),
    "VERGI_ITIRAZI": (
        "Vergi veya harc tahakkukuna itiraz, yanlis veya mukerrer tahakkuk "
        "iddiasi, borcun kendisine ait olmadigi iddiasi, vergi kaydinin "
        "duzeltilmesi talebi, hesaplanan tutara yonelik idari basvuru. Bu, HANGI "
        "vergi/harc turu olursa olsun (emlak vergisi, ilan-reklam vergisi vb.) "
        "gecerlidir -- itiraz varsa vergi turune bakilmaksizin VERGI_ITIRAZI "
        "kullanilir, EMLAK_VERGISI/BELEDIYE_VERGI_HARC kullanilmaz."
    ),
    "BUTCE_MUHASEBE_HARCAMA": (
        "Belediye butce islemleri, muhasebe kayitlari, odeme emri ve harcama "
        "belgeleri, hakedis veya fatura odeme surecleri, birimlerden gelen "
        "odenek talepleri, mali kayit ve odeme bilgisi talepleri. AYIRT ET: "
        "gonderen bir VERGI MUKELLEFI/vatandas ise ve kendi vergi/harc borcu "
        "hakkindaysa BELEDIYE_VERGI_HARC/EMLAK_VERGISI kullan; bu topic belediye "
        "biriminin/yuklenicinin KENDI ic mali islemleri (odenek, hakedis, fatura "
        "odemesi) icindir."
    ),
    "KAMU_ZARARI_MALI_ISLEMLER": (
        "Kamu zarari tespiti, fazla veya yersiz odeme, kamu zararinin tahsili, "
        "mali sorumluluk sureci, zarar tutarina iliskin bilgi/belge talepleri, "
        "mali inceleme sonucunda yapilacak islemler. Ayrica: belediye "
        "alacaklarinin genel tahsilati, vadesi gecmis borc, odeme islemleri, "
        "borc yapilandirma veya odeme durumu hakkinda bilgi, tahsilat "
        "kayitlari, kamu alacagina iliskin resmi kurum yazismalari (eski "
        "KAMU_ALACAGI_TAHSILAT konusu bu basliga dahil edilmistir). "
        "AYIRT ET: vergi/harc TAHAKKUKUNA itiraz VERGI_ITIRAZI'dir; bir "
        "vergi/harc turunun kendisi (ilan-reklam vergisi, isgal harci vb.) "
        "hakkindaki genel bilgi/odeme talebi BELEDIYE_VERGI_HARC'tir; burada "
        "konu spesifik olarak tahsilat sureci, kamu zarari veya fazla/"
        "yersiz odemedir."
    ),
}

REQUIRED_INFO_BY_TOPIC = {
    "EMLAK_VERGISI": (
        "ilgili mukellef veya tasinmaz, tasinmazin yeterli tanimi/adresi, "
        "vergiyle ilgili sorun veya bilgi talebi, talep edilen islem. "
        "(T.C. kimlik numarasi zorunlu degil; kisi adi ve tasinmaz bilgisi "
        "islemi yeterince tanimliyorsa yeterlidir.)"
    ),
    "BELEDIYE_VERGI_HARC": (
        "hangi vergi/harc veya ucretin soz konusu oldugu, ilgili kisi/isletme/"
        "islem, uyusmazlik/bilgi talebi/odeme konusunun ne oldugu, istenen islem"
    ),
    "VERGI_ITIRAZI": (
        "itiraz edilen vergi/harc/tahakkuk, ilgili kisi veya kurulus, itirazin "
        "temel gerekcesi, talep edilen sonuc (tahakkuk numarasi senaryoda varsa eklenmeli)"
    ),
    "BUTCE_MUHASEBE_HARCAMA": (
        "ilgili harcama/odeme/butce islemi, islemin konusu, ilgili kurum/birim "
        "veya yuklenici, talep edilen bilgi veya islem"
    ),
    "KAMU_ZARARI_MALI_ISLEMLER": (
        "kamu zararina, tahsilata veya mali isleme konu olay, ilgili "
        "kisi/birim/kurulus, varsa tutar, talep edilen islem veya bilgi"
    ),
}

DIVERSITY_SCENARIOS = [
    "vatandas dilekcesi",
    "ozel sirket basvurusu",
    "kamu kurumundan gelen resmi yazi",
    "belediye ic biriminden mali bilgi talebi",
    "odeme/tahsilat bilgi talebi",
    "vergi itirazi",
    "emlak vergisi basvurusu",
    "hakedis veya odeme yazisi",
    "butce/odenek yazisi",
    "kamu zarari bilgi/belge talebi",
    "borc/tahsilat islemi",
    "yanlis veya mukerrer tahakkuk iddiasi",
]

SYSTEM_PROMPT = """Sen, belediyelerde kullanilan resmi evraklara benzer SENTETIK egitim verileri ureten bir veri uretim uzmanisin.
Amacin, Mali Hizmetler Muduurlugu kapsamina giren gercekci Turkce evraklar uretmek ve her evrak icin dogru etiketleri olusturmaktir.

KURALLAR:
1. Uretilen butun belgeler TEMIZ ve YETERLI BILGI ICEREN belgeler olmalidir. "missing_information" HER ZAMAN bos dizi [] olmalidir.
   Ancak bunu yapay bicimde saglama: belge metni, konusu acisindan gerekli temel bilgileri GERCEKTEN icermelidir.
   Kontrol et: hangi mali islemin soz konusu oldugu belli mi? ilgili kisi/kurum belli mi? gerekiyorsa ilgili tasinmaz/
   borc/tahakkuk tanimli mi? itiraz varsa itiraz gerekcesi belli mi? gonderenin ne istedigi belli mi? odeme veya
   muhasebe isleminde islem yeterince tanimli mi? Kritik bir bilgi eksikse metni tamamla, sonra [] yaz.
   Bilgi metnin herhangi bir bolumunde (ekler dahil) aciklanmissa eksik sayilmaz.
2. Tum kisi, kurum, adres, vergi/borc bilgileri, tutarlar, tarihler, belge numaralari ve olaylar KURGUSAL olmalidir. Gercek kisi/kurum/ozel olay kullanma.
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
   (Mali Hizmetler'de ozellikle TALEP_DILEKCE_BASVURUSU, BILGI_TALEBI, ITIRAZ_IDARI_BASVURU, KURUMLAR_ARASI_RESMI_YAZI daha sik olabilir, ama zorunlu degil.)
4. sender_type YALNIZCA su degerlerden biri olmali: VATANDAS, KAMU_KURUMU, OZEL_KURULUS, YARGI_MERCII
   VATANDAS, KAMU_KURUMU ve OZEL_KURULUS agirlikli kullan; YARGI_MERCII yalnizca gercekten uygun bir senaryo varsa kullan.
5. primary_topic sana her istekte belirtilecek; TAM OLARAK verilen degeri kullan, degistirme veya yeni kategori uretme.
6. key_information yalnizca belgede ACIKCA bulunan ve islemin anlasilmasi acisindan onemli bilgilerden olusmali.
   type degerleri YALNIZCA sunlar olabilir: PERSON, ORGANIZATION, DATE, REFERENCE_DATE, DOCUMENT_NO, REFERENCE_NO, APPLICATION_NO, CASE_NO, LOCATION, CONTACT, AMOUNT, DEADLINE, EVENT_DETAIL, OTHER
   - Belgede olmayan bilgiyi ASLA yazma. Eksik bir degeri "belirtilmemis" seklinde ekleme.
   - Her tarih/isim/numarayi otomatik key_information yapma, yalnizca gercekten onemli olanlari sec.
   - Ana yazinin sayisi DOCUMENT_NO, ilgi yazisinin numarasi REFERENCE_NO, ilgi yazisinin tarihi REFERENCE_DATE olabilir.
   - Parasal tutarlar AMOUNT olarak yazilabilir. Basvuru numarasi varsa APPLICATION_NO kullan.
   - Ayni bilgiyi iki farkli type ile tekrar etme.
   - Imza sahibinin adi ana olay acisindan onemli degilse PERSON olarak ekleme.
   - HER belgede zorunlu olarak T.C. kimlik numarasi veya para tutari kullanma; yalnizca senaryoya gercekten uyuyorsa ekle.
7. requested_action kisa ama somut olmali: gonderenin Mali Hizmetler Muduurlugunden/belediyeden TAM OLARAK ne istedigini yaz.
   "geregi yapilmasi" gibi genel ifadeler YASAK. Belgede istenmeyen yeni bir islem uretme. Yapilmamis islemi yapilmis gibi gosterme.
8. summary: 1-2 cumle, kisa, tarafsiz, belgeye sadik, yeni bilgi icermeyen, talep ile gerceklesmis islemi birbirine karistirmayan bir ozet.
9. MEVZUAT UYDURMA. Kanun adi, yonetmelik adi veya madde numarasi uydurma. Eger belge senaryosunda mevzuat adi kullanilmasi
   zorunlu degilse HICBIR mevzuat adi kullanma.
10. Su alanlari KESINLIKLE labels icine ekleme: expected_sources, related_legislation, legal_sources, recommended_laws,
    rag_queries, priority, priority_reason, primary_unit, secondary_unit, routing_evidence, keywords, questions_to_answer.
    labels SADECE su 7 alani icermeli: document_type, sender_type, primary_topic, requested_action, key_information,
    missing_information, summary.
11. Belgeleri birbirinin sablon kopyasi yapma: isim, tarih, konum, tutar, mali islem senaryosu, uslup, uzunluk ve
    resmilik seviyesini her seferinde gercekten farklilastir. Ayni primary_topic icinde dahi senaryolar birbirinden
    anlamli bicimde farkli olmalidir. Her olayi vergi itirazina donusturme; konunun kapsamina sadik kal.

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
            "tasinmaz/tutar kurgusunu veya cumle kalibini TEKRARLAMA, farkli ve yeni senaryolar uret):\n- "
            + "\n- ".join(recent)
        )

    return f"""Simdi primary_topic = "{topic}" icin TAM OLARAK {count} adet TEMIZ belge kaydi uret.

Bu konunun kapsami: {TOPIC_HINTS[topic]}

Bu konu icin belgede mutlaka anlasilir olmasi gereken temel bilgiler: {REQUIRED_INFO_BY_TOPIC[topic]}

Uretecegin {count} belge icin onerilen senaryo cesitliligi (birebir uyma zorunlulugu yok, ilham amacli,
istersen farkli sender_type/document_type kombinasyonlari da kullan, gercekci kombinasyonlar uret):
- {chr(10).join(f"{i+1}. {s}" for i, s in enumerate(suggested))}
{avoid_block}

Unutma: her belge farkli kisi/kurum adlari, farkli tasinmaz/tahakkuk/dosya kurgusu, farkli konum/tarih icermeli.
missing_information her zaman [] olmali ama belge gercekten yeterli bilgi icermeli. Her belgede zorunlu olarak
T.C. kimlik no veya para tutari kullanma; yalnizca senaryoya gercekten uyuyorsa ekle. Mevzuat adi uydurma.

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
