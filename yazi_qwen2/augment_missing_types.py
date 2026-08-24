"""
augment_missing_types.py
------------------------
ONAY_YAZISI ve TESPIT_TUTANAGI icin DeepSeek API ile yeni ChatML kayitlari uretir
ve mevcut _updated.jsonl dosyalarina (%80 train / %10 val / %10 test) orantili dager.

Hedef: Her tipten 200 kayit uret => train +160, val +20, test +20
"""

import json, os, time, random, copy, sys, io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from openai import OpenAI

API_KEY      = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL     = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL        = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
TARGET_COUNT = 200   # Her tip icin uretilecek toplam kayit

SPLIT_RATIO  = {"train": 0.80, "val": 0.10, "test": 0.10}

# Her API cagrisi arasinda beklenecek sure (saniye) - rate limit korumasi
REQUEST_DELAY_SEC = 3

# Checkpoint dosyasi - yarida kalirsa devam eder
CHECKPOINT_FILE = "augment_checkpoint.json"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Log dosyasina yaz
LOG_FILE = os.path.join(BASE_DIR, "augment_run.log")
_log_fh  = open(LOG_FILE, "a", encoding="utf-8")

def log(msg: str):
    """Hem ekrana hem log dosyasina yaz."""
    print(msg)
    _log_fh.write(msg + "\n")
    _log_fh.flush()

# ─── Üretilecek tipler ve bağlamları ─────────────────────────────────────────
TYPE_CONFIGS = {
    "ONAY_YAZISI": {
        "process_status": "TAMAMLANDI",
        "description": "İşlem tamamlanmış ve vatandaşın talebi onaylanmıştır. Onay/tamamlanma bildirimi yapılmaktadır.",
        "scenarios": [
            {"document_type": "SIKAYET_IHBAR_BASVURUSU", "primary_topic": "ISYERI_RUHSAT_DENETIM",
             "sender_type": "VATANDAS", "extra": "Ruhsatsız işyeri ihbarı sonuçlanmış, yaptırım uygulanmıştır."},
            {"document_type": "TALEP_DILEKCE_BASVURUSU", "primary_topic": "YAPI_RUHSAT_IZIN",
             "sender_type": "VATANDAS", "extra": "Yapı ruhsatı talebi olumlu sonuçlanmıştır."},
            {"document_type": "SIKAYET_IHBAR_BASVURUSU", "primary_topic": "SEYYAR_SATICI_PAZAR_YERI",
             "sender_type": "VATANDAS", "extra": "Seyyar satıcı şikayeti sonuçlanmış, gerekli işlem yapılmıştır."},
            {"document_type": "BILGI_TALEBI", "primary_topic": "ELEKTRONIK_BELGE_IMZA_TEBLIGAT",
             "sender_type": "VATANDAS", "extra": "Bilgi edinme talebi tamamlanmış, belgeler iletilmiştir."},
            {"document_type": "TALEP_DILEKCE_BASVURUSU", "primary_topic": "HAYVAN_SAHIPLENDIRME_KISIRLASTIRMA",
             "sender_type": "VATANDAS", "extra": "Sahipsiz hayvan talebi yerine getirilmiştir."},
            {"document_type": "SIKAYET_IHBAR_BASVURUSU", "primary_topic": "GURULTU_SIKAYETI",
             "sender_type": "VATANDAS", "extra": "Gürültü şikayeti sonuçlanmış, işyerine uyarı yapılmıştır."},
            {"document_type": "TALEP_DILEKCE_BASVURUSU", "primary_topic": "CALISMAYAN_SOKAK_LAMBASI",
             "sender_type": "VATANDAS", "extra": "Sokak lambası arızası giderilmiştir."},
            {"document_type": "TALEP_DILEKCE_BASVURUSU", "primary_topic": "CADDE_YOL_KALDIRIM_BAKIM_ONARIM",
             "sender_type": "VATANDAS", "extra": "Kaldırım onarımı tamamlanmıştır."},
            {"document_type": "ITIRAZ_IDARI_BASVURUSU", "primary_topic": "IDARI_PARA_CEZASI_ITIRAZ",
             "sender_type": "VATANDAS", "extra": "İtiraz incelenmiş ve haklı bulunarak ceza kaldırılmıştır."},
            {"document_type": "TALEP_DILEKCE_BASVURUSU", "primary_topic": "ISYERI_RUHSAT_DENETIM",
             "sender_type": "OZEL_KURULUS", "extra": "İşyeri ruhsatı başvurusu olumlu sonuçlanmıştır."},
        ]
    },
    "TESPIT_TUTANAGI": {
        "process_status": "INCELEMEDE",
        "description": "İnceleme devam etmektedir ve tespit tutanağı düzenlenmektedir. Genellikle kamu kurumları arası veya kamu zararı, yapı denetimi, denetim tespiti gibi konularda kullanılır.",
        "scenarios": [
            {"document_type": "BILGI_TALEBI", "primary_topic": "KAMU_ZARARI_MALI_ISLEMLER",
             "sender_type": "KAMU_KURUMU", "extra": "İç denetim birimi kamu zararı şüphesiyle belge talep etmiştir."},
            {"document_type": "KURUMLAR_ARASI_RESMI_YAZI", "primary_topic": "YAPI_RUHSAT_IZIN",
             "sender_type": "KAMU_KURUMU", "extra": "İmar aykırı yapı tespiti için yerinde inceleme yapılmaktadır."},
            {"document_type": "BILGI_TALEBI", "primary_topic": "TEBLIGAT_ISLEMLERI",
             "sender_type": "KAMU_KURUMU", "extra": "Tebligat usulsüzlüğü iddiasıyla inceleme başlatılmıştır."},
            {"document_type": "SIKAYET_IHBAR_BASVURUSU", "primary_topic": "ISYERI_RUHSAT_DENETIM",
             "sender_type": "KAMU_KURUMU", "extra": "Denetim birimince işyeri denetimi yapılmakta, tutanak düzenlenmektedir."},
            {"document_type": "KURUMLAR_ARASI_RESMI_YAZI", "primary_topic": "RISKLI_YAPI_KENTSEL_DONUSUM",
             "sender_type": "KAMU_KURUMU", "extra": "Riskli yapı tespiti devam etmekte, teknik rapor hazırlanmaktadır."},
            {"document_type": "BILGI_TALEBI", "primary_topic": "ELEKTRONIK_BELGE_IMZA_TEBLIGAT",
             "sender_type": "KAMU_KURUMU", "extra": "Elektronik imza doğrulaması incelenmekte, tespit tutanağı düzenlenmektedir."},
            {"document_type": "KURUMLAR_ARASI_RESMI_YAZI", "primary_topic": "OTOPARK_DUZENLEMESI",
             "sender_type": "KAMU_KURUMU", "extra": "Otopark ihlali tespiti için yerinde muayene yapılmaktadır."},
            {"document_type": "BILGI_TALEBI", "primary_topic": "CADDE_YOL_KALDIRIM_BAKIM_ONARIM",
             "sender_type": "KAMU_KURUMU", "extra": "Yol yapım hatası iddiasıyla teknik inceleme sürmektedir."},
            {"document_type": "SIKAYET_IHBAR_BASVURUSU", "primary_topic": "GURULTU_SIKAYETI",
             "sender_type": "KAMU_KURUMU", "extra": "Gürültü ölçüm tespiti için yerinde inceleme yapılmaktadır."},
            {"document_type": "KURUMLAR_ARASI_RESMI_YAZI", "primary_topic": "KAMU_ZARARI_MALI_ISLEMLER",
             "sender_type": "KAMU_KURUMU", "extra": "Hakediş ödemelerinde usulsüzlük şüphesiyle inceleme başlatılmıştır."},
        ]
    }
}

# ─── Mevzuat havuzu (bağlam göre seçilecek) ──────────────────────────────────
LEGISLATION_POOL = {
    "ISYERI_RUHSAT_DENETIM": [
        {"law_name": "İşyeri Açma ve Çalışma Ruhsatlarına İlişkin Yönetmelik", "article": "Madde 6",
         "reference_text": "Yetkili idarelerden usulüne uygun işyeri açma ve çalışma ruhsatı alınmadan işyeri açılamaz ve çalıştırılamaz."},
        {"law_name": "5326 sayılı Kabahatler Kanunu", "article": "Madde 32",
         "reference_text": "Yetkili makamlar tarafından verilen emirlere uymayanlara idari para cezası verilir."},
    ],
    "KAMU_ZARARI_MALI_ISLEMLER": [
        {"law_name": "5018 sayılı Kamu Malî Yönetimi ve Kontrol Kanunu", "article": "Madde 71",
         "reference_text": "Kamu görevlilerinin kasıt, kusur veya ihmallerinden kaynaklanan kamu zararları ilgililerden tahsil edilir."},
        {"law_name": "6183 sayılı Amme Alacaklarının Tahsil Usulü Hakkında Kanun", "article": "Madde 55",
         "reference_text": "Amme alacağını vadesinde ödemeyenlere 15 gün içinde borçlarını ödemeleri için ödeme emri tebliğ olunur."},
    ],
    "YAPI_RUHSAT_IZIN": [
        {"law_name": "3194 sayılı İmar Kanunu", "article": "Madde 21",
         "reference_text": "Yapı ruhsatı alınması zorunlu yapıların ruhsatsız inşa edilmesi yasaktır."},
        {"law_name": "4708 sayılı Yapı Denetimi Hakkında Kanun", "article": "Madde 2",
         "reference_text": "Yapı denetim kuruluşlarının denetim sorumluluğu kapsamında yapıların teknik mevzuata uygunluğu denetlenir."},
    ],
    "RISKLI_YAPI_KENTSEL_DONUSUM": [
        {"law_name": "6306 sayılı Afet Riski Altındaki Alanların Dönüştürülmesi Hakkında Kanun", "article": "Madde 3",
         "reference_text": "Riskli yapıların tespiti, ilgili kurum ve kuruluşlarca yapılır veya yaptırılır."},
        {"law_name": "3194 sayılı İmar Kanunu", "article": "Madde 32",
         "reference_text": "Ruhsata aykırı veya ruhsatsız yapılar ilgili idarece yıktırılır."},
    ],
    "TEBLIGAT_ISLEMLERI": [
        {"law_name": "7201 sayılı Tebligat Kanunu", "article": "Madde 21",
         "reference_text": "Adresinde bulunmama hâlinde tebliğ, muhatabın kapı komşusuna yapılır ve durum kapıya yapıştırılır."},
        {"law_name": "5070 sayılı Elektronik İmza Kanunu", "article": "Madde 5",
         "reference_text": "Güvenli elektronik imza, elle atılan ıslak imza ile aynı hukuki sonucu doğurur."},
    ],
    "ELEKTRONIK_BELGE_IMZA_TEBLIGAT": [
        {"law_name": "4982 sayılı Bilgi Edinme Hakkı Kanunu", "article": "Madde 5",
         "reference_text": "Kurumlar, kanundaki istisnalar dışındaki her türlü bilgi veya belgeyi başvuranların erişimine 15 iş günü içinde sunmakla yükümlüdür."},
        {"law_name": "5070 sayılı Elektronik İmza Kanunu", "article": "Madde 5",
         "reference_text": "Güvenli elektronik imza, elle atılan ıslak imza ile aynı hukuki sonucu doğurur."},
    ],
    "DEFAULT": [
        {"law_name": "5393 sayılı Belediye Kanunu", "article": "Madde 14",
         "reference_text": "Belediye, mahallî müşterek nitelikte olmak şartıyla çevre ve çevre sağlığı, temizlik ve katı atık hizmetlerini yapar veya yaptırır."},
        {"law_name": "3071 sayılı Dilekçe Hakkının Kullanılmasına Dair Kanun", "article": "Madde 7",
         "reference_text": "Türk vatandaşlarının ve yabancıların kendileriyle veya kamu ile ilgili dilek ve şikayetleri hakkında yetkili makamlarca 30 gün içinde gerekçeli cevap verilir."},
    ]
}

PERSON_NAMES = [
    "Ahmet Yılmaz", "Fatma Demir", "Mehmet Kaya", "Ayşe Çelik", "Ali Veli",
    "Zeynep Arslan", "Mustafa Öztürk", "Hatice Şahin", "İbrahim Aydın", "Emine Kurt",
    "Hasan Doğan", "Meryem Güler", "Yusuf Koç", "Rukiye Aktaş", "Ömer Çetin",
    "Sümeyye Yıldız", "Recep Kılıç", "Havva Polat", "Bayram Taş", "Güler Aksoy",
    "Murat Efe", "Elif Tekin", "Serkan Bozkurt", "Derya Avcı", "Tolga Karaca",
]

MUNICIPALITIES = [
    "Ankara Büyükşehir", "İstanbul Büyükşehir", "İzmir Büyükşehir", "Bursa Büyükşehir",
    "Antalya Büyükşehir", "Konya Büyükşehir", "Adana Büyükşehir", "Gaziantep Büyükşehir",
    "Kayseri Büyükşehir", "Mersin Büyükşehir", "Erzurum Büyükşehir", "Eskişehir Büyükşehir",
    "Çankaya", "Keçiören", "Yenimahalle", "Mamak", "Etimesgut", "Sincan",
    "Kadıköy", "Üsküdar", "Bakırköy", "Beyoğlu", "Fatih", "Başakşehir",
    "Bornova", "Karşıyaka", "Konak", "Buca", "Gaziemir",
    "Nilüfer", "Osmangazi", "Yıldırım",
    "Muratpaşa", "Kepez", "Konyaaltı",
    "Selçuklu", "Karatay", "Meram",
    "Seyhan", "Yüreğir", "Çukurova",
    "Şahinbey", "Şehitkamil",
    "Melikgazi", "Kocasinan",
    "Yenişehir", "Toroslar",
]

UNITS = [
    "Zabıta Müdürlüğü", "İmar ve Şehircilik Müdürlüğü", "Fen İşleri Müdürlüğü",
    "Mali Hizmetler Müdürlüğü", "Yazı İşleri Müdürlüğü", "Hukuk İşleri Müdürlüğü",
    "Ruhsat ve Denetim Müdürlüğü", "Çevre Koruma ve Kontrol Müdürlüğü",
    "Veteriner İşleri Müdürlüğü", "Sosyal Yardım İşleri Müdürlüğü",
    "Park ve Bahçeler Müdürlüğü", "Temizlik İşleri Müdürlüğü",
    "Muhasebe Müdürlüğü", "Bilgi İşlem Müdürlüğü", "İnsan Kaynakları Müdürlüğü",
]

# ─── System prompt (mevcut veriyle aynı) ─────────────────────────────────────
SYSTEM_PROMPT = """Sen bir belediye evrak yönetim asistanısın. Sana Qwen 1 modelinin ürettiği yapılandırılmış JSON verisi ve bir süreç durumu (process_status) verilecektir.

Görevin: Bu girdi verisi ve süreç durumuna göre, belediyenin hangi aksiyonları alması gerektiğini, hangi birimin sorumlu olduğunu, yanıt yazısının türünü ve tam teşekküllü resmî yazı taslağını içeren bir JSON çıktısı üretmektir.

CIKTI FORMATI - SADECE GECERLI JSON DONDUR, BASKA HICBIR SEY EKLEME:
{
  "performed_actions": ["aksiyon1", "aksiyon2"],
  "responsible_unit": "Sorumlu Mudurlu Adi",
  "response_type": "BILGI_YAZISI",
  "process_status": "INCELEMEDE",
  "result_information": "Vatandasa iletilecek sonuc bilgisi (1-2 cumle)",
  "process_information": "Ic sistem takip notu (YALNIZCA TEK CUMLE, hitapsiz)",
  "draft": "T.C.\\n[BELEDIYE ADI] BELEDIYE BASKANLIGI\\n...tam yazi metni..."
}

ALAN KISITLARI - SADECE BU DEGERLER KULLANILABILIR:

response_type icin izin verilen 7 deger (baska deger KABUL EDILMEZ):
  BILGI_YAZISI
  BILDIRIM_YAZISI
  EKSIK_BILGI_BELGE_TAMAMLAMA_YAZISI
  RET_YAZISI
  YONLENDIRME_YAZISI
  TESPIT_TUTANAGI
  ONAY_YAZISI

process_status icin izin verilen 5 deger (baska deger KABUL EDILMEZ):
  INCELEMEDE
  TAMAMLANDI
  EKSIK_BILGI_BEKLENIYOR
  REDDEDILDI
  YONLENDIRILDI

EKSIK BILGI ZORUNLULUGU:
  Eger process_status=EKSIK_BILGI_BEKLENIYOR ise response_type MUTLAKA EKSIK_BILGI_BELGE_TAMAMLAMA_YAZISI olmalidir.
  Bunun disinda EKSIK_BILGI_BELGE_TAMAMLAMA_YAZISI kullanilmaz.

KURAL 1 - SIFIR DOGRUDAN KANUN/MADDE ATFI:
  Hicbir alanda dogrudan kanun numarasi, madde numarasi veya kanun adi GECMEYECEKTIR.
  YASAK ornekler: "3194 sayili", "Madde 18", "md. 32", "Kanun No: 5393", "Imar Kanunu"
  DOGRU ornekler: "ilgili mevzuat hukumleri", "yasal duzenlemeler", "belediye emir ve yasaklari cercevesinde"
  NOT ONEMLI: Tarihlerdeki yillar (2025, 2026) ve basvuru tarihleri (03/01/2025) kanun numarasi DEGILDIR.
  Taslak yazida tarih belirtmek zorundaysaniz "../../....", "ilgili tarihli" veya bos birakin.

KURAL 2 - ARZ/RICA HIYERARSISI (draft alaninin kapanis ifadesi ZORUNLUDUR):
  Gonderici VATANDAS veya OZEL_KURULUS ise:
    DOGRU kapanis: "Bilgilerinizi rica ederim." veya "Geregini saygılarimla rica ederim."
    YANLIS: "arz ederim." kullanmak (YASAK)
  Ust makama (Valilik, Kaymakamlık, Mahkeme, Savcılık, Bakanlık) ise:
    DOGRU kapanis: "Geregini saygılarimla arz ederim."
    YANLIS: "rica ederim." kullanmak (YASAK)
  Esdurey kamu kurumuna (KAMU_KURUMU) ise:
    DOGRU kapanis: "Geregini bilgilerinize arz ve rica ederim."

KURAL 3 - process_information REGISTER AYRIMI:
  process_information = ic sistem notu = TAM OLARAK TEK CUMLE, hitapsiz, nokta ile biten.
    YANLIS: "Basvuru incelendi. Birime yonlendirildi." (iki cumle - KABUL EDILMEZ)
    YANLIS: "Sayin Mudur, basvuru alindi." (hitap var - KABUL EDILMEZ)
    DOGRU:  "Basvuru Park ve Bahceler Mudurluğüne yonlendirilmistir."
  draft = tam resmi yazi = baslik + muhatap + govde + kapanis, minimum 250 kelime.

GENEL: Tum 7 alan dolu olmalidir. Bos string veya bos liste birakma."""


def build_user_input(response_type_target: str, scenario: dict) -> dict:
    """Belirtilen response_type için uygun user input JSON oluştur."""
    config = TYPE_CONFIGS[response_type_target]
    person = random.choice(PERSON_NAMES)
    muni = random.choice(MUNICIPALITIES)
    ref_no = f"E-2025-{random.randint(1000, 9999)}"
    date = f"{random.randint(1, 28):02d}.{random.randint(1, 12):02d}.2025"

    topic = scenario["primary_topic"]
    legislation = LEGISLATION_POOL.get(topic, LEGISLATION_POOL["DEFAULT"])
    # %85 mevzuatli, %15 mevzuatsiz
    use_legislation = random.random() < 0.85

    return {
        "document_type": scenario["document_type"],
        "sender_type": scenario["sender_type"],
        "primary_topic": topic,
        "requested_action": scenario["extra"],
        "key_information": [
            {"type": "PERSON", "value": person},
            {"type": "ORGANIZATION", "value": f"{muni} Belediyesi"},
            {"type": "DATE", "value": date},
            {"type": "REFERENCE_NO", "value": ref_no},
        ],
        "missing_information": [],
        "summary": f"{person}, {muni} Belediyesi'ne {scenario['extra'].lower()} konusunda başvuruda bulunmuştur.",
        "process_status": config["process_status"],
        "selected_legislation": legislation if use_legislation else [],
    }


def generate_record(client, response_type_target: str, scenario: dict, max_retries: int = 3) -> dict | None:
    """DeepSeek API ile tek kayıt üret."""
    user_input = build_user_input(response_type_target, scenario)
    config = TYPE_CONFIGS[response_type_target]

    # Ek talimat - response_type'i zorla
    extra_instruction = f"""
ONEMLI: Bu gorev icin response_type MUTLAKA "{response_type_target}" olmali.
process_status MUTLAKA "{config['process_status']}" olmali.
{config['description']}
Baska response_type KABUL EDILMEZ.
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + extra_instruction},
        {"role": "user", "content": json.dumps(user_input, ensure_ascii=False)},
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.8,
                max_tokens=2000,
            )
            content = response.choices[0].message.content.strip()

            # JSON parse et
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            parsed = json.loads(content)

            # Doğrulama
            if parsed.get("response_type") != response_type_target:
                log(f"  [UYARI] Yanlis response_type: {parsed.get('response_type')} (beklenen: {response_type_target}), retry {attempt+1}")
                time.sleep(1)
                continue
            if parsed.get("process_status") != config["process_status"]:
                log(f"  [UYARI] Yanlis process_status: {parsed.get('process_status')}, retry {attempt+1}")
                time.sleep(1)
                continue

            # Gerekli alanlar kontrolu
            required = ["performed_actions", "responsible_unit", "response_type",
                       "process_status", "result_information", "process_information", "draft"]
            if not all(parsed.get(k) for k in required):
                log(f"  [UYARI] Eksik alan, retry {attempt+1}")
                time.sleep(1)
                continue

            # ChatML formatina donustur
            chatml_record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_input, ensure_ascii=False)},
                    {"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)},
                ]
            }
            return chatml_record

        except json.JSONDecodeError as e:
            log(f"  [HATA] JSON parse hatasi (attempt {attempt+1}): {e}")
            time.sleep(2)
        except Exception as e:
            log(f"  [HATA] API hatasi (attempt {attempt+1}): {e}")
            time.sleep(3)

    return None


def save_checkpoint(data: dict):
    """Uretilen kayitlari checkpoint dosyasina kaydet."""
    with open(os.path.join(BASE_DIR, CHECKPOINT_FILE), "w", encoding="utf-8") as f:
        # Her kaydi JSON string olarak serialize et
        serializable = {rt: [json.dumps(r, ensure_ascii=False) for r in recs]
                        for rt, recs in data.items()}
        json.dump(serializable, f, ensure_ascii=False)
    log(f"  [CHECKPOINT] {sum(len(v) for v in data.values())} kayit kaydedildi.")


def load_checkpoint() -> dict | None:
    """Varsa checkpoint'ten yukle."""
    path = os.path.join(BASE_DIR, CHECKPOINT_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        result = {rt: [json.loads(s) for s in recs] for rt, recs in raw.items()}
        total = sum(len(v) for v in result.values())
        log(f"  [CHECKPOINT] Mevcut checkpoint bulundu: {total} kayit yuklendi.")
        return result
    except Exception as e:
        log(f"  [CHECKPOINT HATA] {e}")
        return None


def distribute_records(records: list, split_ratio: dict) -> dict:
    """Kayıtları train/val/test'e dağıt."""
    random.shuffle(records)
    total = len(records)
    train_n = int(total * split_ratio["train"])
    val_n   = int(total * split_ratio["val"])
    # Kalan test'e
    return {
        "train": records[:train_n],
        "val":   records[train_n:train_n+val_n],
        "test":  records[train_n+val_n:],
    }


def append_to_jsonl(filepath: str, records: list):
    """Kayitlari mevcut JSONL dosyasina ekle."""
    with open(filepath, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  -> {len(records)} kayit eklendi: {os.path.basename(filepath)}")


def main():
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    log("=" * 60)
    log("AUGMENTATION BASLIYOR")
    log(f"Hedef: Her tipten {TARGET_COUNT} kayit | Istek arasi bekleme: {REQUEST_DELAY_SEC}s")
    log("=" * 60)

    # Checkpoint varsa yukle, yoksa bosla
    checkpoint = load_checkpoint()
    all_generated = checkpoint if checkpoint else {"ONAY_YAZISI": [], "TESPIT_TUTANAGI": []}

    for rt_type, config in TYPE_CONFIGS.items():
        already_have = len(all_generated.get(rt_type, []))
        remaining = TARGET_COUNT - already_have

        if remaining <= 0:
            log(f"\n[ATLANDI] {rt_type}: zaten {already_have} kayit mevcut, hedef karsilandi.")
            continue

        log(f"\n{'='*60}")
        log(f"Uretiliyor: {rt_type} (mevcut: {already_have}, kalan: {remaining}/{TARGET_COUNT})")
        log(f"{'='*60}")

        if rt_type not in all_generated:
            all_generated[rt_type] = []

        generated = all_generated[rt_type]
        scenarios = config["scenarios"]
        attempt_count = 0
        max_attempts = remaining * 3  # basarisiz denemelere karsi buffer

        while len(generated) < TARGET_COUNT and attempt_count < max_attempts:
            # Senaryolari dongusel sec
            scenario = scenarios[attempt_count % len(scenarios)]
            attempt_count += 1

            current = len(generated) + 1
            log(f"  [{current}/{TARGET_COUNT}] {scenario['primary_topic']} ({scenario['document_type']})...",)
            record = generate_record(client, rt_type, scenario)
            if record:
                generated.append(record)
                log("    -> OK")
            else:
                log("    -> BASARISIZ, bir sonraki senaryoya geciliyor")

            # Her 10 kayit'te bir checkpoint kaydet
            if len(generated) % 10 == 0:
                save_checkpoint(all_generated)

            # API rate limit korumasi - istekler arasi ZORUNLU bekleme
            log(f"    [Bekleniyor: {REQUEST_DELAY_SEC}s...]")
            time.sleep(REQUEST_DELAY_SEC)

        log(f"\n{rt_type}: {len(generated)}/{TARGET_COUNT} kayit uretildi")
        all_generated[rt_type] = generated
        # Tip tamamlaninca checkpoint kaydet
        save_checkpoint(all_generated)

    # --- Dagitim ve dosyalara ekleme ---
    log(f"\n{'='*60}")
    log("DAGITIM VE DOSYALARA EKLEME")
    log(f"{'='*60}")

    for rt_type, records in all_generated.items():
        if not records:
            log(f"[ATLANDI] {rt_type}: hic kayit uretilmedi")
            continue

        splits = distribute_records(records, SPLIT_RATIO)
        log(f"\n{rt_type} dagilimi: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

        for split_name, split_records in splits.items():
            if not split_records:
                continue
            filepath = os.path.join(BASE_DIR, f"qwen2_{split_name}_updated.jsonl")
            append_to_jsonl(filepath, split_records)

    # --- Final istatistik ---
    log(f"\n{'='*60}")
    log("FINAL ISTATISTIKLER")
    log(f"{'='*60}")
    import collections
    for split in ["train", "val", "test"]:
        fname = os.path.join(BASE_DIR, f"qwen2_{split}_updated.jsonl")
        rt_counts = collections.Counter()
        total = 0
        with open(fname, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    r = json.loads(line)
                    asst = next((m for m in r.get("messages", []) if m["role"] == "assistant"), None)
                    if asst:
                        c = json.loads(asst["content"])
                        rt_counts[c.get("response_type", "MISSING")] += 1
                        total += 1
                except: pass
        log(f"\n{split.upper()} ({total} kayit):")
        for k, v in sorted(rt_counts.items(), key=lambda x: -x[1]):
            log(f"  {k}: {v} ({100*v/total:.1f}%)")

    log("\nTUM ISLEMLER TAMAMLANDI.")
    _log_fh.close()


if __name__ == "__main__":
    main()
