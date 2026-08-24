"""
Imar ve Sehircilik Muduurlugu - Sentetik Egitim Verisi Uretici (DeepSeek API)
================================================================================

DeepSeek API'sini (OpenAI-uyumlu chat completions endpoint'i) kullanarak
belediyelerin Imar ve Sehircilik Muduurlugu kapsaminda 70 adet TEMIZ (eksik
bilgi icermeyen) sentetik Turkce evrak + etiket kaydi uretir.

Onceki scriptlerle (Fen Isleri / Hukuk Isleri / Mali Hizmetler / Park ve
Bahceler / Sosyal Yardim / Temizlik Isleri / Veteriner Isleri / Yazi Isleri /
Zabita / Cevre Koruma) ayni mimari (konu bazli, thread havuzu ile paralel
uretim, semaya gore dogrulama, ara kayit).

Kullanim:
    pip install -r requirements.txt
    copy .env.example .env   # anahtari .env icine yazin; .env commit edilmez
    python generate_dataset_imar.py

Cikti:
    dataset_imar.jsonl
    dataset_imar.json
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

OUTPUT_JSONL = "dataset_imar.jsonl"
OUTPUT_JSON = "dataset_imar.json"

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
TOPIC_TARGETS: dict[str, int] = {
    "YAPI_RUHSATI_IMAR_DURUMU": 12,
    "KACAK_RUHSATA_AYKIRI_YAPI": 12,
    "IMAR_PLANI_PLAN_DEGISIKLIGI": 12,
    "ADA_PARSEL_ARSA_DUZENLEME": 25,
    "YAPI_DENETIMI": 11,
    "RISKLI_YAPI_KENTSEL_DONUSUM": 11,
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
    "YAPI_RUHSATI_IMAR_DURUMU": (
        "Yapi ruhsati hakkinda bilgi talebi, yeni yapi icin ruhsat "
        "sureci, ruhsat basvurusunun durumu, tasinmazin imar durumu "
        "hakkinda bilgi isteme, yapilasma kosullari hakkinda bilgi "
        "talebi, bina yuksekligi/kat durumu gibi yapilasma bilgilerinin "
        "sorulmasi, ruhsat belgesine iliskin bilgi/belge talebi, kamu "
        "kurumundan ruhsat veya imar durumu bilgisi istenmesi. AYIRT ET: "
        "konu ADA/PARSEL SINIRLARI, parselasyon veya arsa duzenlemesiyse "
        "(parsel birlestirme/ayirma, duzenleme ortakligi vb.) ADA_PARSEL_"
        "ARSA_DUZENLEME kullan; bu topic bina/yapilasma haklari ve ruhsat "
        "sureciyle ilgili olmali."
    ),
    "KACAK_RUHSATA_AYKIRI_YAPI": (
        "Ruhsatsiz yapi iddiasi, ruhsata aykiri ilave kat, projeye "
        "aykiri balkon/kapama, izinsiz eklenti, yapi sinirlarini asan "
        "imalat iddiasi, komsu parselde kacak yapi sikayeti, yapidaki "
        "aykiriligin incelenmesi talebi, kamu kurumundan ruhsata "
        "aykirilik incelemesi talebi. (Iddialari kesin tespit gibi "
        "yazma; 'ruhsatsiz oldugu iddia edilmektedir', 'projeye aykiri "
        "olabilecegi bildirilmektedir' gibi tarafsiz dil kullan.)"
    ),
    "IMAR_PLANI_PLAN_DEGISIKLIGI": (
        "Imar plani hakkinda bilgi talebi, plan degisikligi talebi, "
        "plan degisikligine itiraz, tasinmazin kullanim kararinin "
        "sorulmasi, plan notu hakkinda bilgi, mevcut plan kararinin "
        "degistirilmesi talebi, askidaki plana yonelik itiraz, kamu "
        "kurumundan plan bilgisi veya gorus talebi."
    ),
    "ADA_PARSEL_ARSA_DUZENLEME": (
        "Ada/parsel bilgisine iliskin islem, parselasyon duzenlemesi "
        "hakkinda bilgi, arsa duzenleme sureci, parsel siniri veya imar "
        "parseli bilgisi, duzenleme ortaklik islemleri hakkinda bilgi "
        "talebi, parselin mevcut imar durumuyla iliskisi, tasinmazin "
        "ada/parsel bazinda incelenmesi, kamu kurumundan tasinmaz/"
        "parsel bilgisi talebi. AYIRT ET: konu bina yapma hakki, ruhsat "
        "sureci veya yapilasma kosullariysa (kat/yukseklik, ruhsat "
        "basvurusu) YAPI_RUHSATI_IMAR_DURUMU kullan; bu topic parsel/ada "
        "sinirlari ve arazi duzenlemesiyle ilgili olmali."
    ),
    "YAPI_DENETIMI": (
        "Devam eden yapinin projeye uygunlugunun incelenmesi, yapi "
        "denetim sureci hakkinda bilgi talebi, insaatta teknik "
        "aykirilik iddiasi, denetim sonucunun sorulmasi, yapi denetim "
        "kurulusu/islemine iliskin basvuru, yapinin onayli projeye "
        "uygunlugu, saha incelemesi talebi, kamu kurumundan yapi "
        "denetim bilgisi istenmesi."
    ),
    "RISKLI_YAPI_KENTSEL_DONUSUM": (
        "Riskli yapi sureci hakkinda bilgi talebi, bina guvenligine "
        "iliskin donusum basvurusu, riskli yapi olarak degerlendirilen "
        "binaya iliskin islem, kentsel donusum surecinin asamalarinin "
        "sorulmasi, yapi maliklerinin donusum talebi, riskli yapi "
        "kararina iliskin bilgi veya itiraz, donusum alanindaki "
        "tasinmaz hakkinda bilgi, kamu kurumundan riskli yapi/donusum "
        "bilgi talebi."
    ),
}

REQUIRED_INFO_BY_TOPIC = {
    "YAPI_RUHSATI_IMAR_DURUMU": (
        "ilgili tasinmaz/yapi, konum veya ada-parsel gibi yeterli "
        "tanimlayici bilgi, ruhsat/imar durumu talebinin konusu, "
        "istenen islem veya bilgi"
    ),
    "KACAK_RUHSATA_AYKIRI_YAPI": (
        "sikayet edilen yapinin yeterli konumu, aykirilik iddiasinin "
        "niteligi, ilgili yapi/parsel, istenen inceleme veya islem"
    ),
    "IMAR_PLANI_PLAN_DEGISIKLIGI": (
        "ilgili alan/tasinmaz/ada-parsel, mevcut plan veya degisiklik "
        "talebinin konusu, itiraz varsa itirazin temel gerekcesi, "
        "istenen bilgi/degisiklik/inceleme"
    ),
    "ADA_PARSEL_ARSA_DUZENLEME": (
        "ada-parsel veya tasinmazin yeterli tanimi, duzenleme/inceleme "
        "konusunun ne oldugu, ilgili kisi/kurum, istenen bilgi veya islem"
    ),
    "YAPI_DENETIMI": (
        "ilgili yapi, denetim gerektiren durum veya bilgi talebi, yapi "
        "konumu/parsel bilgisi, istenen inceleme/sonuc/bilgi"
    ),
    "RISKLI_YAPI_KENTSEL_DONUSUM": (
        "ilgili bina/tasinmaz, konum veya yeterli tanimlayici bilgi, "
        "riskli yapi/donusum surecinin hangi asamasinin konu oldugu, "
        "talep edilen bilgi/islem"
    ),
}

DIVERSITY_SCENARIOS = [
    "vatandas bilgi talebi",
    "yapi ruhsati basvurusu",
    "imar durumu sorgusu",
    "kacak yapi ihbari",
    "komsu yapi sikayeti",
    "plan degisikligi talebi",
    "plan degisikligine itiraz",
    "ada/parsel bilgi talebi",
    "ozel sirket/yuklenici yazisi",
    "kamu kurumundan tasinmaz bilgisi talebi",
    "yapi denetimi basvurusu",
    "riskli yapi/kentsel donusum bilgi talebi",
]

SYSTEM_PROMPT = """Sen, belediyelerde kullanilan resmi evraklara benzer SENTETIK egitim verileri ureten bir veri uretim uzmanisin.
Amacin, Imar ve Sehircilik Muduurlugu kapsamina giren gercekci Turkce evraklar uretmek ve her evrak icin dogru etiketleri olusturmaktir.

KURALLAR:
1. Uretilen butun belgeler TEMIZ ve YETERLI BILGI ICEREN belgeler olmalidir. "missing_information" HER ZAMAN bos dizi [] olmalidir.
   Ancak bunu yapay bicimde saglama: belge metni, konusu acisindan gerekli temel bilgileri GERCEKTEN icermelidir.
   Kontrol et: hangi yapi/tasinmaz/parselin soz konusu oldugu belli mi? konum yeterince acik mi? imar/ruhsat/plan/
   denetim talebinin konusu belli mi? kacak yapi ihbarinda aykiriligin niteligi belli mi? plan itirazinda hangi
   plan/alanin kastedildigi belli mi? riskli yapi/donusum talebinde ilgili bina belli mi? gonderenin ne istedigi
   acik mi? Kritik bir bilgi eksikse metni tamamla, sonra [] yaz.
2. Tum kisi, kurum, adres, ada-parsel bilgileri, tarih, belge numarasi, ruhsat bilgileri ve olaylar KURGUSAL olmalidir. Gercek kisi/kurum/tasinmaz kullanma.
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
   (Imar ve Sehircilik'te ozellikle TALEP_DILEKCE_BASVURUSU, BILGI_TALEBI, ITIRAZ_IDARI_BASVURU, KURUMLAR_ARASI_RESMI_YAZI, SIKAYET_IHBAR_BASVURUSU daha sik olabilir, ama zorunlu degil.)
4. sender_type YALNIZCA su degerlerden biri olmali: VATANDAS, KAMU_KURUMU, OZEL_KURULUS, YARGI_MERCII. VATANDAS,
   KAMU_KURUMU ve OZEL_KURULUS agirlikli kullan.
5. primary_topic sana her istekte belirtilecek; TAM OLARAK verilen degeri kullan, degistirme veya yeni kategori uretme.
6. key_information yalnizca belgede ACIKCA bulunan ve islemin anlasilmasi acisindan onemli bilgilerden olusmali.
   type degerleri YALNIZCA sunlar olabilir: PERSON, ORGANIZATION, DATE, REFERENCE_DATE, DOCUMENT_NO, REFERENCE_NO, APPLICATION_NO, CASE_NO, LOCATION, CONTACT, AMOUNT, DEADLINE, EVENT_DETAIL, OTHER
   - Belgede olmayan bilgiyi ASLA ekleme. Bulunmayan bilgiyi "belirtilmemis" seklinde yazma.
   - Her isim/tarih/numarayi otomatik cikarma, yalnizca islem acisindan gercekten onemli olanlari sec.
   - Ayni bilgiyi tekrar etme. Imza sahibinin adi ana olayin kisisi degilse gereksiz PERSON ekleme.
   - Ada/parsel bilgisini mumkunse LOCATION veya OTHER icinde anlamli bicimde tut.
   - Ana yazinin sayisi DOCUMENT_NO, basvuru numarasi APPLICATION_NO, ilgi yazisi numarasi REFERENCE_NO, ilgi
     yazisi tarihi REFERENCE_DATE olabilir.
7. requested_action kisa ama somut olmali: gonderenin Imar ve Sehircilik Muduurlugunden/belediyeden TAM OLARAK ne
   istedigini yaz. "geregi yapilmasi" gibi genel ifadeler YASAK. Belgede talep edilmeyen yeni islem ekleme. Henuz
   yapilmamis incelemeyi yapilmis gibi yazma.
8. summary: 1-2 cumle, kisa, tarafsiz, belgeye sadik, yeni bilgi icermeyen bir ozet. Kacak/aykiri yapi
   sikayetlerini kesinlesmis tespit gibi yazma.
9. IDDIA VE TESPIT KURALI: sikayetlerde kesin hukum verme. "Yapi kacaktir." YAZMA; bunun yerine "Basvuru sahibi,
   yapinin ruhsatsiz olarak insa edildigini iddia etmektedir." gibi YAZ. "Insaat projeye aykiridir." YAZMA; bunun
   yerine "Yapinin onayli projeye aykiri imalat icerdigi bildirilmektedir." gibi YAZ.
10. MEVZUAT UYDURMA. Kanun, yonetmelik, madde veya mevzuat kaynagi uydurma.
11. Su alanlari KESINLIKLE labels icine ekleme: priority, priority_reason, primary_unit, secondary_unit,
    expected_sources, rag_queries, routing_evidence, questions_to_answer, keywords, related_legislation,
    legal_sources, recommended_laws. labels SADECE su 7 alani icermeli: document_type, sender_type, primary_topic,
    requested_action, key_information, missing_information, summary.
12. BIRIM SINIRLARI: Imar ve Sehircilik ile Fen Isleri ve Hukuk Isleri konularini KESINLIKLE karistirma.
    Imar ve Sehircilik odagi: ruhsat, imar durumu, imar plani, kacak/ruhsata aykiri yapi, ada/parsel, yapi
    denetimi, riskli yapi/kentsel donusum. Fen Isleri odagi (bu dataset'e ALMA): yol/asfalt/kaldirim, belediye
    fiziksel yapim isleri, bakim/onarim, teknik altyapi uygulamalari. Ornek: "Binanin ruhsata aykiri kati oldugu
    iddia ediliyor." -> IMAR; "Belediye hizmet binasinin catisi akiyor, onarim istiyoruz." -> FEN. Hukuk Isleri
    odagi (bu dataset'e ALMA): dava, mahkeme, hukuki gorus, uyusmazligin yargisal boyutu. Ornek: "Imar plani
    degisikligine idari basvuru ile itiraz ediyorum." -> IMAR; "Imar plani islemi hakkinda acilan davaya savunma
    hazirlanmasi isteniyor." -> HUKUK.
13. YAPI_RUHSATI_IMAR_DURUMU / YAPI_DENETIMI ayrimi: YAPI_RUHSATI_IMAR_DURUMU ruhsatin varligi/basvurusu, imar
    durumu, yapilasma kosullariyla ilgilidir. YAPI_DENETIMI devam eden/tamamlanan yapinin onayli projeye/teknik
    surece uygunlugunun denetlenmesi, saha kontrolu, yapi denetim islemleriyle ilgilidir. Ornek: "Bu parsele kac
    kat yapi yapilabilir?" -> YAPI_RUHSATI_IMAR_DURUMU; "Insaatin onayli projeden farkli yapildigi iddia ediliyor,
    yerinde kontrol istiyorum." -> YAPI_DENETIMI (veya aykirilik acikca ruhsata iliskinse KACAK_RUHSATA_AYKIRI_YAPI).
14. Belgeleri birbirinin sablon kopyasi yapma: isim, ada/parsel, konum, uslup, uzunluk ve resmilik seviyesini her
    seferinde gercekten farklilastir. Her YAPI_RUHSATI_IMAR_DURUMU kaydini ayni imar durumu sorgusuna donusturme,
    her KACAK_RUHSATA_AYKIRI_YAPI kaydini yalnizca kacak kat senaryosu yapma, her RISKLI_YAPI_KENTSEL_DONUSUM
    kaydini ayni riskli bina basvurusu seklinde uretme. Ayni senaryoyu yalnizca ada/parsel veya isim degistirerek
    tekrar uretme.

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
            "ada-parsel/konum kurgusunu veya cumle kalibini TEKRARLAMA, farkli ve yeni senaryolar uret):\n- "
            + "\n- ".join(recent)
        )

    return f"""Simdi primary_topic = "{topic}" icin TAM OLARAK {count} adet TEMIZ belge kaydi uret.

Bu konunun kapsami ve alt senaryolari: {TOPIC_HINTS[topic]}

Bu konu icin belgede mutlaka anlasilir olmasi gereken temel bilgiler: {REQUIRED_INFO_BY_TOPIC[topic]}

Uretecegin {count} belge icin onerilen senaryo cesitliligi (birebir uyma zorunlulugu yok, ilham amacli,
istersen farkli sender_type/document_type kombinasyonlari da kullan, gercekci kombinasyonlar uret):
- {chr(10).join(f"{i+1}. {s}" for i, s in enumerate(suggested))}
{avoid_block}

Unutma: her belge farkli kisi/kurum adlari, farkli ada-parsel/konum bilgileri, farkli tarih ve belge numaralari
icermeli, ve mumkun oldugunca farkli ALT SENARYO kullanmali (ayni topic icinde tek bir alt senaryoyu tekrar tekrar
kullanma). missing_information her zaman [] olmali ama belge gercekten yeterli bilgi icermeli. Kacak/aykiri yapi
iddialarini kesinlesmis tespit gibi yazma, tarafsiz/temkinli dil kullan ("iddia edilmektedir", "bildirilmektedir").
Imar ve Sehircilik kapsaminda kal (Fen Isleri fiziksel bakim/onarim veya Hukuk Isleri dava/mahkeme sureci agirlikli
senaryo uretme). Mevzuat adi uydurma.

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
