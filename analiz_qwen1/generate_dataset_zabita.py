"""
Zabita Muduurlugu - Sentetik Egitim Verisi Uretici (DeepSeek API)
====================================================================

DeepSeek API'sini (OpenAI-uyumlu chat completions endpoint'i) kullanarak
belediyelerin Zabita Muduurlugu kapsaminda 70 adet TEMIZ (eksik bilgi
icermeyen) sentetik Turkce evrak + etiket kaydi uretir.

Onceki scriptlerle (Fen Isleri / Hukuk Isleri / Mali Hizmetler / Park ve
Bahceler / Sosyal Yardim / Temizlik Isleri / Veteriner Isleri / Yazi Isleri)
ayni mimari (konu bazli, thread havuzu ile paralel uretim, semaya gore
dogrulama, ara kayit).

Kullanim:
    pip install -r requirements.txt
    copy .env.example .env   # anahtari .env icine yazin; .env commit edilmez
    python generate_dataset_zabita.py

Cikti:
    dataset_zabita.jsonl
    dataset_zabita.json
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

OUTPUT_JSONL = "dataset_zabita.jsonl"
OUTPUT_JSON = "dataset_zabita.json"

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
# NOT (taksonomi birlestirme): BELEDIYE_EMIR_YASAK_DENETIM, tanimi geregi
# "ozel baska bir topic'e girmeyen" bir catch-all kategoriydi; en kucuk
# sinif olmasi (8 kayit) ve hem KABAHAT_IDARI_YAPTIRIM hem SEYYAR_SATICI_
# PAZAR_YERI ile karismasi nedeniyle KABAHAT_IDARI_YAPTIRIM'a birlestirildi.
TOPIC_TARGETS: dict[str, int] = {
    "ISYERI_RUHSAT_DENETIM": 14,
    "KALDIRIM_ALAN_ISGALI": 14,
    "SEYYAR_SATICI_PAZAR_YERI": 14,
    "KABAHAT_IDARI_YAPTIRIM": 22,
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
    "ISYERI_RUHSAT_DENETIM": (
        "Ruhsatsiz faaliyet gosterdigi iddia edilen isyeri, isyeri acma "
        "ve calisma ruhsatina iliskin denetim talebi, ruhsat "
        "bilgilerinin guncelliginin kontrolu, faaliyet konusu ile ruhsat "
        "arasindaki uyumsuzluk iddiasi, calisma saatleri veya isletme "
        "kosullarina iliskin denetim, kamu kurumundan isyeri denetimi "
        "talebi, isletmenin faaliyet kosullarina iliskin vatandas "
        "sikayeti, isyeri hakkinda yapilan denetimin durumuna iliskin "
        "bilgi talebi. (Cevresel gurultu olcumu veya sanayi kaynakli "
        "cevre kirliligi ana konuysa bu topic'e uretme; ruhsat/isletme "
        "denetimi boyutu baskin olmali.) AYIRT ET: eger olay ZATEN bir "
        "idari para cezasi/tutanak KESILMIS ve buna itiraz ediliyorsa "
        "KABAHAT_IDARI_YAPTIRIM kullan; bu topic henuz ceza kesilmemis, "
        "ruhsat durumunun kontrol/denetim asamasindaki durumlar icindir."
    ),
    "KALDIRIM_ALAN_ISGALI": (
        "Kaldirimin masa/sandalye ile kapatilmasi, isyeri onundeki "
        "urunlerin yaya gecisini engellemesi, reklam panosu veya stand "
        "ile kaldirim isgali, kamuya ait gecis alaninin malzeme ile "
        "kapatilmasi, bina onune esya birakilmasi, vatandasin yaya "
        "gecisinin engellendigine dair sikayeti, gecici isgal hakkinda "
        "bilgi/denetim talebi, isletmeye ait malzemelerin kamu alanina "
        "tasmasi."
    ),
    "SEYYAR_SATICI_PAZAR_YERI": (
        "Izinsiz seyyar satis, cadde veya meydanda seyyar satici "
        "sikayeti, pazar yerinde tezgah duzeni ihlali, pazar yeri "
        "tahsis/yerlesim duzenine iliskin basvuru, pazar alaninda gecis "
        "yollarinin kapatilmasi, pazar yeri kurallarina iliskin denetim "
        "talebi, kayit disi satis iddiasi, pazar esnafindan gelen "
        "islem/bilgi talebi."
    ),
    "KABAHAT_IDARI_YAPTIRIM": (
        "Idari para cezasina itiraz, zabita tutanagina itiraz, kabahat "
        "nedeniyle uygulanan yaptirimin incelenmesi, ceza kararinin "
        "gerekcesinin sorulmasi, idari yaptirim kaydinda hata oldugu "
        "iddiasi, tekrarlanan ihlal nedeniyle uygulanan islem, "
        "yaptirimin yeniden degerlendirilmesi talebi, kurumlar arasi "
        "idari yaptirim bilgi/belge yazismasi. Ayrica: belediye "
        "tarafindan belirlenen yasaklara aykirilik ihbari, kamuya acik "
        "alanda belediye duzenine aykiri davranis, izinsiz ilan/afis veya "
        "benzeri uygulamalar, kamusal alandaki uygunsuz kullanimin "
        "denetlenmesi talebi, kamu kurumundan zabita denetimi talebi "
        "(eski BELEDIYE_EMIR_YASAK_DENETIM konusu bu basliga dahil "
        "edilmistir: hem verilmis bir cezaya/tutanaga itiraz hem de henuz "
        "ceza kesilmemis bir ihlalin denetim/yaptirim icin bildirilmesi "
        "artik ayni topic altinda). AYIRT ET: olay OZELLIKLE seyyar satici "
        "veya pazar yeri ile ilgiliyse SEYYAR_SATICI_PAZAR_YERI, kaldirim "
        "isgaliyse KALDIRIM_ALAN_ISGALI, isyeri ruhsat/denetimiyse "
        "ISYERI_RUHSAT_DENETIM kullanilmali; bu topic sadece bu dort "
        "kategoriden birine girmeyen genel belediye emir/yasak ihlalleri "
        "ve idari yaptirim/ceza sureçleri icindir."
    ),
}

REQUIRED_INFO_BY_TOPIC = {
    "ISYERI_RUHSAT_DENETIM": (
        "ilgili isyeri/isletme, isyerinin yeterli konum/adres bilgisi, "
        "denetim gerektiren durum, istenen islem"
    ),
    "KALDIRIM_ALAN_ISGALI": (
        "isgalin gerceklestigi yer, isgalin ne sekilde gerceklestigi, "
        "ilgili isletme/kisi mumkunse tanimli, istenen denetim veya "
        "kaldirma islemi"
    ),
    "SEYYAR_SATICI_PAZAR_YERI": (
        "olayin gerceklestigi pazar/cadde/meydan, seyyar satis veya "
        "pazar yeri sorununun niteligi, varsa ilgili kisi/tezgah/alan, "
        "istenen islem"
    ),
    "KABAHAT_IDARI_YAPTIRIM": (
        "hangi idari yaptirim/tutanagin soz konusu oldugu, ilgili "
        "kisi/isletme, itiraz/talebin gerekcesi, istenen sonuc (veya, ihbar/"
        "denetim talebi senaryolarinda: hangi davranis/uygulamanin "
        "sikayet edildigi, olayin yeri, ilgili kisi/isletme/alan, istenen "
        "denetim veya islem)"
    ),
}

DIVERSITY_SCENARIOS = [
    "vatandas sikayeti",
    "vatandas ihbari",
    "isletme sahibinin itirazi",
    "esnaf basvurusu",
    "kamu kurumundan denetim talebi",
    "ruhsat kontrolu talebi",
    "kaldirim isgali",
    "seyyar satis",
    "pazar yeri duzeni",
    "zabita tutanagina itiraz",
    "idari para cezasi bilgi talebi",
    "belediye emir/yasaklarina iliskin denetim",
]

SYSTEM_PROMPT = """Sen, belediyelerde kullanilan resmi evraklara benzer SENTETIK egitim verileri ureten bir veri uretim uzmanisin.
Amacin, Zabita Muduurlugu kapsamina giren gercekci Turkce evraklar uretmek ve her evrak icin dogru etiketleri olusturmaktir.

KURALLAR:
1. Uretilen butun belgeler TEMIZ ve YETERLI BILGI ICEREN belgeler olmalidir. "missing_information" HER ZAMAN bos dizi [] olmalidir.
   Ancak bunu yapay bicimde saglama: belge metni, konusu acisindan gerekli temel bilgileri GERCEKTEN icermelidir.
   Kontrol et: olayin yeri belli mi? hangi isletme/kisi/alanin soz konusu oldugu yeterince tanimli mi? ihlal/sikayetin
   niteligi belli mi? idari yaptirima itiraz varsa hangi isleme itiraz edildigi ve gerekcesi belli mi? gonderenin ne
   istedigi belli mi? Kritik bir bilgi eksikse metni tamamla, sonra [] yaz.
2. Tum kisi, kurum, isletme, adres, tarih, belge numarasi ve olaylar KURGUSAL olmalidir. Gercek kisi/kurum/isletme kullanma.
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
   (Zabita'da ozellikle SIKAYET_IHBAR_BASVURUSU, TALEP_DILEKCE_BASVURUSU, ITIRAZ_IDARI_BASVURU, KURUMLAR_ARASI_RESMI_YAZI daha sik olabilir, ama zorunlu degil.)
4. sender_type YALNIZCA su degerlerden biri olmali: VATANDAS, KAMU_KURUMU, OZEL_KURULUS, YARGI_MERCII. VATANDAS,
   KAMU_KURUMU ve OZEL_KURULUS agirlikli kullan.
5. primary_topic sana her istekte belirtilecek; TAM OLARAK verilen degeri kullan, degistirme veya yeni kategori uretme.
6. key_information yalnizca belgede ACIKCA bulunan ve islemin anlasilmasi acisindan onemli bilgilerden olusmali.
   type degerleri YALNIZCA sunlar olabilir: PERSON, ORGANIZATION, DATE, REFERENCE_DATE, DOCUMENT_NO, REFERENCE_NO, APPLICATION_NO, CASE_NO, LOCATION, CONTACT, AMOUNT, DEADLINE, EVENT_DETAIL, OTHER
   - Belgede olmayan bilgiyi ASLA ekleme. Bulunmayan bilgiyi "belirtilmemis" diye yazma.
   - Her tarih/isim/numarayi otomatik cikarma, yalnizca islem acisindan gercekten onemli olanlari sec.
   - Ayni bilgiyi tekrar etme. Imza sahibinin adi olayin tarafi degilse gereksiz PERSON ekleme.
   - Ana yazinin sayisi DOCUMENT_NO, ilgi yazisi numarasi REFERENCE_NO, ilgi yazisi tarihi REFERENCE_DATE olabilir.
   - Ceza/tutanak numarasi varsa olay baglamina gore REFERENCE_NO veya OTHER kullanilabilir.
   - Parasal idari yaptirim tutari belgede onemliyse AMOUNT kullanilabilir.
7. requested_action kisa ama somut olmali: gonderenin Zabita Muduurlugunden/belediyeden TAM OLARAK ne istedigini
   yaz. "geregi yapilmasi" gibi genel ifadeler YASAK. Belgede talep edilmeyen yeni islem ekleme. Henuz yapilmamis bir
   denetimi yapilmis gibi gosterme.
8. summary: 1-2 cumle, kisa, tarafsiz, belgeye sadik, yeni bilgi icermeyen bir ozet. Sikayet veya ihbarlarda
   iddialari kesinlesmis gercek gibi yazma.
9. MEVZUAT UYDURMA. Kanun, yonetmelik veya madde numarasi uydurma.
10. Su alanlari KESINLIKLE labels icine ekleme: priority, priority_reason, primary_unit, secondary_unit,
    expected_sources, rag_queries, routing_evidence, questions_to_answer, keywords, related_legislation,
    legal_sources, recommended_laws. labels SADECE su 7 alani icermeli: document_type, sender_type, primary_topic,
    requested_action, key_information, missing_information, summary.
11. BIRIM SINIRLARI: Zabita ile diger birimleri KESINLIKLE karistirma.
    Zabita odagi: isyeri ruhsati ve belediye denetimi, kaldirim/kamu alani isgali, seyyar satis, pazar yeri duzeni,
    belediye emir ve yasaklari, kabahat ve idari yaptirim surecleri.
    Cevre Koruma'ya ait ana konulari UeRETME: cevresel gurultu olcumu, hava kirliligi, sanayi kaynakli cevresel
    etki, CED. Isyeri kaynakli bir sikayette ana talep cevresel gurultunun olculmesi/teknik cevre denetimiyse
    Cevre Koruma kapsamindadir, bu dataset'e alma; ana talep isletmenin ruhsat/faaliyet kosullarina veya belediye
    duzenine uygunlugunun denetlenmesiyse Zabita kapsamindadir.
    Veteriner Isleri'ne ait ana konulari UeRETME: yarali/hasta hayvan, sahipsiz hayvan sagligi, kuduz suphesi.
    Fen Isleri'ne ait ana konulari UeRETME: yol, asfalt, kaldirimin fiziksel bozuklugu. Onemli ayrim: "Kaldirim
    kirik ve cukur" -> FEN ISLERI; "Isletme kaldirima masa koymus, gecis engelleniyor" -> ZABITA.
    Hukuk sinir: idari yaptirimda vatandas/isletmenin cezanin yeniden incelenmesini istemesi KABAHAT_IDARI_YAPTIRIM
    olabilir, ama konu dava/mahkeme savunmasi veya kapsamli hukuki gorus surecine donusmusse Hukuk Isleri
    kapsamindadir, bu dataset'e alma; yalnizca Zabita'nin kendi idari yaptirim/denetim sureciyle dogrudan iliskili
    ornekleri uret.
12. Belgeleri birbirinin sablon kopyasi yapma: isletme adi, adres, olay turu, uslup, uzunluk ve resmilik seviyesini
    her seferinde gercekten farklilastir. Her ISYERI_RUHSAT_DENETIM kaydini ruhsatsiz isyeri senaryosu yapma, her
    KALDIRIM_ALAN_ISGALI kaydini masa-sandalye senaryosu yapma, her KABAHAT_IDARI_YAPTIRIM kaydini ayni ceza itirazi
    kalibiyla uretme. Ayni senaryoyu yalnizca isletme adi veya adres degistirerek tekrar uretme.

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
            "\nDaha once bu konuda uretilen bazi ozetler (bunlarla ayni senaryoyu, isletme/kisi adini, "
            "konum/olay kurgusunu veya cumle kalibini TEKRARLAMA, farkli ve yeni senaryolar uret):\n- "
            + "\n- ".join(recent)
        )

    return f"""Simdi primary_topic = "{topic}" icin TAM OLARAK {count} adet TEMIZ belge kaydi uret.

Bu konunun kapsami ve alt senaryolari: {TOPIC_HINTS[topic]}

Bu konu icin belgede mutlaka anlasilir olmasi gereken temel bilgiler: {REQUIRED_INFO_BY_TOPIC[topic]}

Uretecegin {count} belge icin onerilen senaryo cesitliligi (birebir uyma zorunlulugu yok, ilham amacli,
istersen farkli sender_type/document_type kombinasyonlari da kullan, gercekci kombinasyonlar uret):
- {chr(10).join(f"{i+1}. {s}" for i, s in enumerate(suggested))}
{avoid_block}

Unutma: her belge farkli isletme/kisi/kurum adlari, farkli konum/mahalle/cadde adlari, farkli tarih ve belge
numaralari icermeli, ve mumkun oldugunca farkli ALT SENARYO kullanmali (ayni topic icinde tek bir alt senaryoyu
tekrar tekrar kullanma). missing_information her zaman [] olmali ama belge gercekten yeterli bilgi icermeli.
Zabita'nin kendi denetim/ruhsat/isgal/pazar/idari yaptirim kapsaminda kal; Cevre Koruma (gurultu, hava kirliligi,
CED), Veteriner Isleri (hayvan sagligi), Fen Isleri (yol/kaldirim fiziksel bozuklugu) veya Hukuk Isleri (dava/
mahkeme sureci) agirlikli senaryo uretme. Mevzuat adi uydurma.

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
