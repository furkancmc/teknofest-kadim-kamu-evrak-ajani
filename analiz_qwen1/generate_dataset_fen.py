"""
Fen Isleri Muduurlugu - Sentetik Egitim Verisi Uretici (DeepSeek API)
======================================================================

Bu script, DeepSeek API'sini (OpenAI-uyumlu chat completions endpoint'i)
kullanarak belediyelerin Fen Isleri Muduurlugu kapsaminda 70 adet TEMIZ
(eksik bilgi icermeyen) sentetik Turkce evrak + etiket kaydi uretir.

Kullanim:
    pip install -r requirements.txt
    copy .env.example .env   # anahtari .env icine yazin; .env commit edilmez
    python generate_dataset.py

Cikti:
    dataset.jsonl   -> her satirda bir kayit (document_text + labels)
    dataset.json    -> ayni veri, tek bir JSON dizisi olarak

Notlar:
    - Anahtar yalniz .env veya DEEPSEEK_API_KEY ortam degiskeninden okunur.
    - Model adi DEEPSEEK_MODEL ile degistirilebilir (varsayilan: deepseek-chat).
"""

import json
import os
import random
import re
import sys
import time
from typing import Any

from openai import OpenAI
from _env import deepseek_api_key

# ----------------------------------------------------------------------------
# Ayarlar
# ----------------------------------------------------------------------------

API_KEY = deepseek_api_key()
BASE_URL = "https://api.deepseek.com"
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

OUTPUT_JSONL = "dataset.jsonl"
OUTPUT_JSON = "dataset.json"

BATCH_SIZE = 3          # bir API cagrisinda uretilecek belge sayisi (cesitlilik icin kucuk tutuluyor)
MAX_RETRIES_PER_BATCH = 5
REQUEST_TIMEOUT = 120

# Konu basina hedef sayi (bir sonraki uretim/top-up calistirmasi icin).
# NOT: YAPIM_ISLERI zayif sinif (toplam veri setinde sadece 16 kayit) ->
# hedef yukseltildi.
TOPIC_TARGETS: dict[str, int] = {
    "YOL_BAKIM_YAPIM": 12,
    "KALDIRIM_YAYA_YOLU": 12,
    "OTOPARK_DUZENLEMESI": 11,
    "YAPIM_ISLERI": 30,
    "IHALE_SOZLESME_SURECLERI": 12,
    "YAPIM_MUAYENE_KABUL": 11,
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
    "LOCATION",
    "DEADLINE",
    "AMOUNT",
    "EVENT_DETAIL",
    "OTHER",
}

TOPIC_HINTS = {
    "YOL_BAKIM_YAPIM": (
        "Bozuk yol, asfalt hasari, yol bakim/onarma, yol yapim talebi, yol kaplamasi, "
        "yol calismasina iliskin kurum yazismalari. AYIRT ET: sorun ARAC YOLUNDA "
        "(asfalt, kaplama) ise bu topic; sorun KALDIRIM/YAYA YOLUNDA ise "
        "KALDIRIM_YAYA_YOLU kullan."
    ),
    "KALDIRIM_YAYA_YOLU": (
        "Bozuk kaldirim, yaya yolu, kaldirim yapim/onarim talebi, yaya guvenligini "
        "etkileyen fiziksel sorunlar. AYIRT ET: sorun ARAC YOLUNDA (asfalt, "
        "kaplama) ise YOL_BAKIM_YAPIM kullan; bu topic sadece kaldirim/yaya "
        "yoluyla ilgili olmali."
    ),
    "OTOPARK_DUZENLEMESI": (
        "Belediye otoparki, otopark yapimi/duzenlenmesi, otopark alanina iliskin "
        "teknik talepler, otopark duzenlemesine iliskin kurum yazismalari."
    ),
    "YAPIM_ISLERI": (
        "Belediye hizmet binasi veya tesis yapimi, yapim isindeki fiziksel kusurlar, "
        "yuklenici tarafindan yurutulen yapim isleri, teknik saha incelemesi, "
        "belediye yapim projesine iliskin talepler. AYIRT ET: konu ihalenin/"
        "sozlesmenin kendisi (ihale sureci, sozlesme hukumleri, yuklenici "
        "yukumlulugu) ise IHALE_SOZLESME_SURECLERI; konu isin gecici/kesin "
        "KABUL veya MUAYENE surecine iliskinse (kabul tutanagi, eksik/kusurlu "
        "is tespiti) YAPIM_MUAYENE_KABUL kullan. Bu topic, insaatin/isin "
        "fiziksel yurutulmesi ve genel durumu ile ilgili olmali."
    ),
    "IHALE_SOZLESME_SURECLERI": (
        "Yapim ihalesi, yapim sozlesmesinin uygulanmasi, yuklenici islemleri, "
        "sozlesme kapsamindaki is ve yukumlulukler, yapim isi ihale sureclerine "
        "iliskin kurum/sirket yazismalari. AYIRT ET: yazismanin konusu ihale/"
        "sozlesme sureci veya yukumlulugu degil de isin fiziksel durumu ya da "
        "kabul/muayene sureci ise sirasiyla YAPIM_ISLERI veya YAPIM_MUAYENE_"
        "KABUL kullan."
    ),
    "YAPIM_MUAYENE_KABUL": (
        "Gecici kabul, kesin kabul, muayene ve kabul islemleri, eksik veya kusurlu "
        "islerin tespiti, kabul surecine iliskin bilgi/belge yazismalari. AYIRT "
        "ET: konu genel olarak yapim isinin ilerleyisi/kusurlariysa (henuz "
        "kabul/muayene sureci baslamamissa) YAPIM_ISLERI kullan; bu topic "
        "SADECE resmi kabul/muayene surecine (gecici kabul, kesin kabul, "
        "muayene komisyonu vb.) atifta bulunan belgeler icindir."
    ),
}

REQUIRED_INFO_BY_TOPIC = {
    "YOL_BAKIM_YAPIM": "sorun/talebin niteligi, ilgili yol/sokak/cadde veya yeterli konum bilgisi, istenen islem",
    "KALDIRIM_YAYA_YOLU": "kaldirim/yaya yolu sorununun niteligi, konum, istenen islem",
    "OTOPARK_DUZENLEMESI": "ilgili otopark/alan, talep/sorun, istenen islem",
    "YAPIM_ISLERI": "ilgili yapim isi/proje/tesis, konu olan sorun veya islem, istenen islem",
    "IHALE_SOZLESME_SURECLERI": (
        "ilgili yapim isi/sozlesmenin hangisi oldugu, taraf/kurum/yuklenici bilgisi, "
        "yazismanin konusu, istenen islem"
    ),
    "YAPIM_MUAYENE_KABUL": "hangi yapim isine iliskin oldugu, kabul/muayene surecinin konusu, talep edilen islem",
}

DIVERSITY_SCENARIOS = [
    "kisa vatandas dilekcesi",
    "detayli vatandas basvurusu",
    "kamu kurumundan gelen resmi yazi",
    "yuklenici/sirket yazisi",
    "bilgi talebi",
    "teknik islem talebi",
    "sikayet/ihbar",
    "kurumlar arasi bilgi veya islem talebi",
]

SYSTEM_PROMPT = """Sen, belediyelerde kullanilan resmi evraklara benzer SENTETIK egitim verileri ureten bir veri uretim uzmanisin.
Amacin, Fen Isleri Muduurlugu kapsamina giren gercekci Turkce evraklar uretmek ve her evrak icin dogru etiketleri olusturmaktir.

KURALLAR:
1. Uretilen butun belgeler TEMIZ ve YETERLI BILGI ICEREN belgeler olmalidir. "missing_information" HER ZAMAN bos dizi [] olmalidir.
   Ancak bunu yapay bicimde saglama: belge metni, konusu acisindan gerekli temel bilgileri GERCEKTEN icermelidir.
   Kontrol et: konu anlasilabiliyor mu? olay/talep yeterince tanimli mi? islem yapilacak yer/is/proje belli mi?
   gonderenin ne istedigi belli mi? Kritik bir bilgi eksikse metni tamamla, sonra [] yaz.
2. Tum kisi, kurum, adres, tarih, belge numarasi, ihale numarasi ve olaylar KURGUSAL olmalidir. Gercek kisi/kurum kullanma.
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
4. sender_type YALNIZCA su degerlerden biri olmali: VATANDAS, KAMU_KURUMU, OZEL_KURULUS, YARGI_MERCII
5. primary_topic sana her istekte belirtilecek; TAM OLARAK verilen degeri kullan, degistirme veya yeni kategori uretme.
6. key_information yalnizca belgede ACIKCA bulunan ve islemin anlasilmasi acisindan onemli bilgilerden olusmali.
   type degerleri YALNIZCA sunlar olabilir: PERSON, ORGANIZATION, DATE, REFERENCE_DATE, DOCUMENT_NO, REFERENCE_NO, APPLICATION_NO, LOCATION, DEADLINE, AMOUNT, EVENT_DETAIL, OTHER
   - Belgede olmayan bilgiyi yazma. Her tarih/isim/numarayi otomatik key_information yapma, yalnizca gercekten onemli olanlari sec.
   - Ilgi yazisinin tarihi -> REFERENCE_DATE, ilgi yazisinin numarasi -> REFERENCE_NO, ana yazinin sayisi -> DOCUMENT_NO
   - Ayni bilgiyi tekrar etme.
7. requested_action: belgeyi gonderenin Fen Isleri Muduurlugunden NET olarak ne istedigini belirt (ozel ve somut olsun,
   "geregi yapilmasi" gibi genel ifadeler YASAK). Belgede istenmeyen yeni islem ekleme.
8. summary: 1-2 cumle, kisa, tarafsiz, belgeye sadik, yeni bilgi icermeyen bir ozet.
9. Mevzuat adi, kanun/yonetmelik numarasi UYDURMA. "expected_sources" veya "related_legislation" gibi alanlar ekleme.
10. Yapilmamis bir belediye islemini yapilmis gibi gosterme.
11. Belgeleri birbirinin sablon kopyasi yapma: isim, tarih, konum, uslup, uzunluk ve resmilik seviyesini her seferinde
    gercekten farklilastir. Ayni primary_topic icinde dahi senaryolar birbirinden anlamli bicimde farkli olmalidir.

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


def build_user_prompt(topic: str, count: int, used_scenarios: list[str], used_summaries: list[str]) -> str:
    scenario_pool = DIVERSITY_SCENARIOS.copy()
    random.shuffle(scenario_pool)
    suggested = scenario_pool[:count]

    avoid_block = ""
    if used_summaries:
        recent = used_summaries[-12:]
        avoid_block = (
            "\nDaha once bu konuda uretilen bazi ozetler (bunlarla ayni senaryoyu, kisi/kurum adini, "
            "konum adini veya cumle kalibini TEKRARLAMA, farkli ve yeni senaryolar uret):\n- "
            + "\n- ".join(recent)
        )

    return f"""Simdi primary_topic = "{topic}" icin TAM OLARAK {count} adet TEMIZ belge kaydi uret.

Bu konunun kapsami: {TOPIC_HINTS[topic]}

Bu konu icin belgede mutlaka anlasilir olmasi gereken temel bilgiler: {REQUIRED_INFO_BY_TOPIC[topic]}

Uretecegin {count} belge icin onerilen senaryo cesitliligi (birebir uyma zorunlulugu yok, ilham amacli,
istersen farkli sender_type/document_type kombinasyonlari da kullan, gercekci kombinasyonlar uret):
- {chr(10).join(f"{i+1}. {s}" for i, s in enumerate(suggested))}
{avoid_block}

Unutma: her belge farkli kisi/kurum adlari, farkli konum/sokak/proje adlari, farkli tarihler ve belge numaralari
icermeli. missing_information her zaman [] olmali ama belge gercekten yeterli bilgi icermeli.

Sadece istenen JSON nesnesini dondur: {{"records": [...]}} (tam olarak {count} eleman)."""


def call_deepseek(client: OpenAI, topic: str, count: int, used_summaries: list[str]) -> list[dict[str, Any]]:
    user_prompt = build_user_prompt(topic, count, DIVERSITY_SCENARIOS, used_summaries)

    print(f"  ... API'ye istek gonderildi ({count} kayit), yanit bekleniyor "
          f"(en fazla {REQUEST_TIMEOUT}sn)...", flush=True)
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
    print(f"  ... yanit alindi ({elapsed:.1f}sn)", flush=True)

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
    attempt = 0

    while len(collected) < target and attempt < MAX_RETRIES_PER_BATCH * 4:
        remaining = target - len(collected)
        batch_count = min(BATCH_SIZE, remaining)
        attempt += 1

        try:
            raw_records = call_deepseek(client, topic, batch_count, used_summaries)
        except Exception as exc:  # noqa: BLE001
            print(f"  [!] {topic}: API/parse hatasi ({exc}), tekrar deneniyor...")
            time.sleep(2)
            continue

        for rec in raw_records:
            if len(collected) >= target:
                break
            ok, reason = validate_record(rec, topic)
            if not ok:
                print(f"  [x] {topic}: kayit reddedildi ({reason})")
                continue
            collected.append(rec)
            used_summaries.append(rec["labels"]["summary"])
            print(f"  [+] {topic}: {len(collected)}/{target} kayit uretildi")

    if len(collected) < target:
        print(f"  [!] UYARI: {topic} icin sadece {len(collected)}/{target} gecerli kayit uretilebildi.")

    return collected


def save_records(records: list[dict[str, Any]]) -> None:
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

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=REQUEST_TIMEOUT, max_retries=0)

    all_records: list[dict[str, Any]] = []
    try:
        for topic, target in TOPIC_TARGETS.items():
            print(f"\n=== {topic} (hedef: {target}) ===")
            topic_records = generate_topic(client, topic, target)
            all_records.extend(topic_records)
            save_records(all_records)
            print(f"  -> ara kayit yapildi ({len(all_records)}/{total_target} toplam)")
    except KeyboardInterrupt:
        print("\n[!] Kullanici tarafindan durduruldu. Su ana kadar uretilenler kaydediliyor...")
    finally:
        save_records(all_records)
        print(f"\nToplam uretilen kayit: {len(all_records)}/{total_target}")
        print(f"Kaydedildi: {OUTPUT_JSONL}, {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
