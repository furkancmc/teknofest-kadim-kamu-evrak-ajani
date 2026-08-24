# Belediye Evrak Ajanı

**Kamu Evrak ve Yazışma Süreçleri için Uçtan Uca Yapay Zekâ Dil Ajanı Sistemi**

> TEKNOFEST 2026 — Yapay Zeka Dil Ajanları Yarışması, 1. Senaryo (Kamu Evrak).  
> **Görev 1** evrak sınıflandırma ve içerik analizi · **Görev 2** resmî yazı taslaklama ve birim yönlendirme.

Belediye dilekçesi/görseli gelir → OCR okur → analiz JSON üretir → mevzuat maddesi getirilir → **memur onaylar** → resmî yazı taslağı basılır. Hedef müdürlüğü model uydurmaz; `primary_topic` → tablo.

---

## GitHub’da gördüğünüz sistem (ne yaptık)

| Katman | Ne |
|---|---|
| **4 ajan + HITL** | Okuyucu → Analiz → Mevzuat → *memur durur* → Yazıcı |
| **vLLM (canlı)** | İki motor: (1) Qwen2.5-VL OCR, `max_loras=1` · (2) tek Qwen2.5-7B-Instruct, **multi-LoRA** `max_loras=2` (`analiz_lora` + `yazi_lora`). Merge yok; ikinci 7B taban yok. |
| **Üç LoRA** | OCR (VL) · Qwen1 analiz · Qwen2 yazı — hepsi taban Instruct/VL üzerine adaptör |
| **RAG** | BGE-M3 + BM25 + RRF + BGE-reranker, **CPU** (GPU LLM’lere kalır). Sorgu = konu anahtarı + `requested_action` |
| **Yönlendirme** | Deterministik `belediye_konu.json` (58 konu → 11 müdürlük) |
| **Veri** | Görev 1–2: sentetik kurgu JSONL. OCR: kamuya açık Yargıtay/Danıştay metni → sentetik PNG (A/B/C/D). RAG: açık mevzuat |
| **Demo** | Colab A100, Gradio `share=True`: `app/notebooks/Qwen1_RAG_Demo_Colab.ipynb` |

Holdout özeti (aynı test, LoRA vs ham model): OCR CER **0.4283 → 0.1076** · Qwen1 `document_type` **%70.1 → %85.5** · Qwen2 `response_type` **%50.0 → %82.18** · RAG H@1 **69.0** (üretim). Ayrıntı [§9](#9-değerlendirme-sonuçları).

---

## İçindekiler

1. [Mimari ve Uçtan Uca Akış](#1-mimari-ve-uçtan-uca-akış) · [vLLM + multi-LoRA](#canlı-çıkarım-vllm--multi-lora-tek-a100)
2. [Depo Yapısı](#2-depo-yapısı)
3. [Modüller ve Modellerin Nasıl Eğitildiği](#3-modüller-ve-modellerin-nasıl-eğitildiği)
4. [Veri Şeması](#4-veri-şeması)
5. [Resmî Yazışma Protokolleri](#5-resmî-yazışma-protokolleri)
6. [11 Müdürlük ve RAG Mevzuat Havuzu](#6-11-müdürlük-ve-rag-mevzuat-havuzu)
7. [Kurulum ve Baştan Sona Çalıştırma](#7-kurulum-ve-baştan-sona-çalıştırma)
8. [Modeller ve Veri: Repoda Ne Var, Ne Yok](#8-modeller-ve-veri-repoda-ne-var-ne-yok)
9. [Değerlendirme Sonuçları](#9-değerlendirme-sonuçları)
10. [Ticarileşme ve Ölçeklenebilirlik](#10-ticarileşme-ve-ölçeklenebilirlik)
11. [Lisans, Etik ve Kaynaklar](#11-lisans-etik-ve-kaynaklar)

---

## 1. Mimari ve Uçtan Uca Akış

Sistem, her biri belirli bir rolü üstlenen **beş özelleşmiş modülden** oluşur. Akışın tam ortasında **insan-döngüde (Human-in-the-Loop, HITL)** memur kararı vardır: model karar verir ama son sözü memur söyler.

```
   Evrak görseli / metni
           │
           ▼
 ┌───────────────────────┐
 │ 1) OKUYUCU  (OCR)      │  Qwen2.5-VL-7B + LoRA  →  ham metin
 │    QWEN_VL_ocr/         │
 └───────────┬───────────┘
             ▼
 ┌───────────────────────┐
 │ 2) ANALİZ  (Qwen1)     │  Qwen2.5-7B-Instruct + LoRA
 │    analiz_qwen1/       │  →  JSON: document_type, sender_type, primary_topic,
 └───────────┬───────────┘      requested_action, key_information, missing_information, summary
             ▼
 ┌───────────────────────┐
 │ 3) MEVZUAT  (RAG)      │  BGE-M3 (dense) + BM25 + RRF + BGE-reranker-v2-m3
 │    mevzuat_rag/        │  →  ilgili kanun/yönetmelik maddeleri (selected_legislation)
 └───────────┬───────────┘
             ▼
        ⏸  MEMUR (HITL)        madde işaretler · süreç durumu seçer · not girer
             │
             ▼
 ┌───────────────────────┐
 │ 4) YAZICI  (Qwen2)     │  Qwen2.5-7B-Instruct + LoRA
 │    yazi_qwen2/         │  →  JSON: performed_actions, target_unit, response_type,
 └───────────┬───────────┘      process_status, result_information, process_information, draft
             ▼
   Birim yönlendirme = primary_topic → müdürlük  (deterministik tablo, model değil)
   app/maps/belediye_konu.json
```

Tümünü tek arayüzde birleştiren orkestratör ve Gradio demosu `app/` altındadır.

### Canlı çıkarım: vLLM + multi-LoRA (tek A100)

Eğitim Unsloth/TRL ile yapılır; **jüri demosu vLLM** üzerindedir. Adaptörler tabana **merge edilmez**.

```
GPU (A100 80 GB)
├── vLLM motor 1 — Qwen2.5-VL-7B + OCR LoRA     max_loras=1     görsel → metin
└── vLLM motor 2 — Qwen2.5-7B-Instruct            max_loras=2
        ├── LoRA adı analiz_lora  (Qwen1)         metin → 7 alanlı JSON
        └── LoRA adı yazi_lora    (Qwen2)         JSON + HITL + maddeler → taslak

CPU
└── RAG  BGE-M3 embed + BM25 + reranker           (~4 GB; GPU'ya konulmaz, yoksa vLLM OOM)
```

İkinci bir 7B taban yüklenmez. Colab’da ilk **Run all** vLLM paket çakışması için bilinçli `RuntimeError` verir → Restart → ikinci Run all. Ayrıntılı hücre notları demo notebook’un en üstündedir.

---

## 2. Depo Yapısı

```
belediye-evrak-ajani/
├── README.md                      # (bu dosya) GitHub ana sayfa + tüm sistem
├── LICENSE                        # MIT
├── .gitignore                     # model ağırlıkları / büyük veri hariç tutulur
├── requirements.txt               # üst düzey bağımlılıklar (içinde vllm)
├── docs/
│   └── demo_onizleme_mock.html           # statik arayüz önizlemesi
│
├── app/                           # ORKESTRATÖR + GRADIO DEMO (uçtan uca akış + HITL)
│   ├── notebooks/
│   │   └── Qwen1_RAG_Demo_Colab.ipynb    # Ana demo (Colab A100, share=True)
│   ├── src/                              # Notebook hücrelerinin okunabilir kaynak ikizleri
│   │   ├── gradio_app.py                 # Gradio arayüzü
│   │   ├── draft_prompt.py               # Yazıcı sistem prompt'u
│   │   ├── topic_map.py                  # konu→birim / konu→RAG eşleme yükleyici
│   │   └── vllm_worker.py                # vLLM işçisi (OCR/analiz; GPU doluysa düşülür)
│   └── maps/
│       └── belediye_konu.json            # konu→müdürlük ve konu→RAG anahtar tablosu
│
├── QWEN_VL_ocr/                   # 1) OKUYUCU — Qwen2.5-VL-7B OCR (LoRA)
│   ├── download_dataset.py               # HF kaynak metin setini indirir
│   ├── generate_dataset_1000.py          # 1000 sentetik belge görseli üretir (800/100/100)
│   ├── qwen_ocr_baseline_test.ipynb      # ham model CER
│   ├── qwen_ocr_finetune_lora.ipynb      # LoRA eğitimi + aynı test setinde CER
│   └── data/                             # eğitim setinden 4 örnek görsel (A/B/C/D)
│
├── analiz_qwen1/                  # 2) ANALİZ — Qwen1 evrak analiz LoRA (Görev 1)
│   ├── generate_dataset_*.py             # 11 müdürlük için sentetik evrak üretimi
│   ├── build_finetune_splits.py          # train/val/test split
│   ├── _env.py, .env.example             # DeepSeek API anahtarı yönetimi (.env commit edilmez)
│   ├── Qwen1_FineTune_Colab.ipynb        # LoRA eğitim notebook'u
│   └── veri/                             # üretilmiş eğitim seti (tam)
│       ├── train.jsonl (1957) · val.jsonl (246) · test.jsonl (241)
│       └── split_stats.txt               # konu bazında dağılım
│
├── yazi_qwen2/                    # 4) YAZICI — Qwen2 karar destek + resmî yazı LoRA (Görev 2)
│   ├── 01_finetune_qwen25_7b.ipynb       # LoRA eğitimi (Unsloth, BF16)
│   ├── 02_evaluate_qwen25_7b.ipynb       # LoRA holdout (response_type %82.18)
│   ├── 03_evaluate_base_model.ipynb      # ham Instruct baseline (response_type %50.0)
│   ├── augment_missing_types.py          # nadir yanıt türlerini dengeleme (eğitim seti)
│   ├── canonical_system_prompt.txt       # Qwen2 kanonik sistem prompt'u
│   └── veri/                             # TEMSİLİ ÖRNEK (tam set Drive'da)
│       ├── qwen2_train_ornek.jsonl (250) · qwen2_val_ornek.jsonl (60) · qwen2_test_ornek.jsonl (60)
│
└── mevzuat_rag/                   # 3) MEVZUAT — RAG arama motoru (dense + BM25 + rerank)
    ├── requirements.txt
    └── rag/
        ├── search.py                     # Searcher: hibrit arama + RRF + reranker
        ├── config.py · scope.py · rerankers.py
        ├── embed/models.py               # BGE-M3 embedding
        └── index/bm25.py                 # BM25 sözcük araması
```

**Notebook'lar (ne işe yarar):**

| Dosya | Ne yapar |
|---|---|
| `QWEN_VL_ocr/qwen_ocr_baseline_test.ipynb` | Ham VL model, 100 holdout belgede CER |
| `QWEN_VL_ocr/qwen_ocr_finetune_lora.ipynb` | OCR LoRA eğitimi + aynı testte CER |
| `analiz_qwen1/Qwen1_FineTune_Colab.ipynb` | Analiz LoRA + 241 kayıtlık holdout |
| `yazi_qwen2/01_finetune_qwen25_7b.ipynb` | Yazıcı LoRA |
| `yazi_qwen2/02_evaluate_qwen25_7b.ipynb` | Yazıcı LoRA holdout — `response_type` **%82.18** (522 kayıt) |
| `yazi_qwen2/03_evaluate_base_model.ipynb` | Yazıcı **ham** Instruct — `response_type` **%50.0** (aynı 522) |
| `app/notebooks/Qwen1_RAG_Demo_Colab.ipynb` | Jüri demosu: OCR → analiz → RAG → memur → yazı |

---

## 3. Modüller ve Modellerin Nasıl Eğitildiği

Üç model eğitilmiştir; hepsi **tam eğitim yerine LoRA** (Low-Rank Adaptation) ile, tek GPU'da uygulanabilir olacak şekilde uyarlanmıştır.

### 3.1. OCR — Qwen2.5-VL-7B (LoRA) · `QWEN_VL_ocr/`

- **Amaç:** Türkçe belge görsellerinden metin çıkarımı.
- **Yaklaşım:** Görsel encoder dondurulur; yalnızca dil modeli katmanlarına LoRA adaptörleri eklenir. Böylece 7B model sınırlı veriyle (1000 örnek) ve tek GPU ile eğitilebilir.
- **Veri:** Açık Hugging Face derlemesi [`erdem-erdem/Turkish-Law-Documents-700k-clustered`](https://huggingface.co/datasets/erdem-erdem/Turkish-Law-Documents-700k-clustered) indirilir (`download_dataset.py`). Metinler Yargıtay ve Danıştay’ın **kamuya açık** karar arama sitelerinden derlenmiş düz metinlerdir; asıl mahkeme dosyası / tarama PDF’si değildir. `generate_dataset_1000.py` her metni HTML antetli sayfa olarak yeniden basar (PNG). Belediye dilekçesi, CİMER kaydı veya EBYS evrakı **yoktur**.
- **Bozulma grupları** (tarayıcı/kamera gürültüsüne benzer sentetik bozma; `test/` eğitimde kullanılmaz):

  | Grup | Ne | Örnek (repoda) |
  |---|---|---|
  | **A** | Temiz baskı, bozma yok (200 DPI) | `QWEN_VL_ocr/data/A_0003.png` |
  | **B** | Çok az bozulma (hafif bulanıklık, küçük kontrast/parlaklık kayması) | `QWEN_VL_ocr/data/B_0077.png` |
  | **C** | Belirgin bozukluk (daha güçlü blur, kontrast düşüşü, Gauss gürültüsü) | `QWEN_VL_ocr/data/C_0065.png` |
  | **D** | Bariz kötü tarama (düşük DPI, ağır blur/gürültü, tuz-biber, dönme) | `QWEN_VL_ocr/data/D_0064.png` |

  Tam set 800 train / 100 valid / 100 test (grup başına 250). Repoya yalnız bu dört örnek konmuştur (`QWEN_VL_ocr/data/`); 1000’lik zip Drive’dadır.
- **Eğitim:** Unsloth/TRL, `r=16`, 3 epoch. Baseline (ham VL) ile LoRA **aynı** holdout üzerinde CER ve Türkçe karakter hata oranı (ı/i, ş/s, ğ/g, ö/o, ü/u, ç/c karışmaları) ile karşılaştırılır.
- **Canlı (vLLM):** OCR **ayrı** bir vLLM motorunda çalışır (`Qwen2.5-VL-7B` + OCR LoRA, `max_loras=1`, bf16). Adaptör merge edilmez. GPU doluysa Colab `app/src/vllm_worker.py` alt sürecine düşer.

### 3.2. Analiz — Qwen1 (Qwen2.5-7B-Instruct + LoRA) · `analiz_qwen1/` · **Görev 1**

- **Amaç:** Ham vatandaş/kurum başvuru metnini okuyup yapılandırılmış JSON'a çevirmek.
- **Veri (gerçek sayılar):** **2.444** sentetik Türkçe evrak · **58 konu** · **11 müdürlük**.

  | Split | Kayıt |
  |---|---:|
  | train.jsonl | 1.957 |
  | val.jsonl | 246 |
  | test.jsonl | 241 |
  | **Toplam** | **2.444** |

  Konu bazında ayrıntılı dağılım `analiz_qwen1/veri/split_stats.txt` dosyasındadır.
- **Üretim:** Her müdürlük için ayrı bir `generate_dataset_*.py` (imar, fen, hukuk, mali, park, sosyal, temizlik, veteriner, yazı, zabıta, çevre) birim şemasına uygun sentetik evrak üretir; `build_finetune_splits.py` train/val/test ayırır. Üretimde DeepSeek API kullanılır (anahtar `.env` üzerinden, repoya girmez).
- **Çıktı alanları:** `document_type`, `sender_type`, `primary_topic`, `requested_action`, `key_information`, `missing_information`, `summary`.
- **Canlı (vLLM):** Analiz, yazıcıyla **aynı** Qwen2.5-7B-Instruct motorunda `analiz_lora` adıyla takılır (`max_loras=2`). İkinci 7B taban yok.

### 3.3. Yazıcı — Qwen2 (Qwen2.5-7B-Instruct + LoRA) · `yazi_qwen2/` · **Görev 2**

- **Amaç:** Qwen1'in JSON'u + süreç durumu + RAG mevzuatını alıp; idari aksiyonları, hedef birimi, yanıt türünü, süreç notunu ve **resmî yazı taslağını** üretmek.
- **Veri:**

  | Split | Kayıt |
  |---|---:|
  | qwen2_train_updated.jsonl | 4.240 |
  | qwen2_val_updated.jsonl | 863 |
  | qwen2_test_updated.jsonl | 522 |
  | **Toplam** | **5.625** |

  Format: **ChatML JSONL** (system / user / assistant).

- **1→N durum eşlemesi (veri çoğaltma yöntemi):** Qwen1'in ürettiği **tek** analiz JSON'u, gerçek idari hayattaki gibi **birden çok sürece** dönüşebilir. Aynı başvuru; "incelemede", "eksik bilgi bekleniyor", "yönlendirildi", "reddedildi" veya "tamamlandı" durumlarının her birinde **farklı** bir resmî yazı gerektirir. Bu nedenle her tohum analiz kaydı, farklı `process_status` değerleriyle eşlenerek N ayrı eğitim örneğine açılır (1→N). Böylece model, aynı evrakın süreç durumuna göre nasıl farklı yazıştığını öğrenir. Bu yüzden Qwen2 seti (5.625), Qwen1 tohum setinden (2.444) daha büyüktür.

- **Üretim hattı (yalnız eğitim seti):** DeepSeek ile taslak üretildi; JSON / enum / arz-rica / mevzuat atfı doğrulayıcıdan geçti. Nadir türler `augment_missing_types.py` ile dengelendi. Canlı çıkarım bu script’lere bağlı değildir.
- **Ölçüm:** ham Instruct = `03_evaluate_base_model.ipynb` (`response_type` %50.0) · LoRA = `02_evaluate_qwen25_7b.ipynb` (`response_type` %82.18). İkisi de Unsloth/HF; **jüri demosu vLLM**.
- **Canlı (vLLM):** aynı Instruct motorunda `yazi_lora`. Memur HITL kararı + `selected_legislation` girdidir.

- **RAG entegrasyonu (tasarım oranı ~%85 / %15):** Kayıtların yaklaşık **%85'inde** `selected_legislation` dolu (2–4 mevzuat maddesi) — model mevzuata **doğrudan atıf** yapar. Yaklaşık **%15'inde** liste boştur (**fallback**): RAG'dan sonuç dönmediğinde model **halüsinasyon yapmadan** genel idari usule uygun yazı üretmeyi öğrenir.

- **Eğitim setindeki gerçek dağılımlar (train, ölçülen):**

  | `response_type` | Adet | | `process_status` | Adet |
  |---|---:|---|---|---:|
  | EKSIK_BILGI_BELGE_TAMAMLAMA_YAZISI | 1.127 | | EKSIK_BILGI_BEKLENIYOR | 1.127 |
  | BILDIRIM_YAZISI | 727 | | TAMAMLANDI | 944 |
  | BILGI_YAZISI | 709 | | INCELEMEDE | 849 |
  | YONLENDIRME_YAZISI | 631 | | YONLENDIRILDI | 688 |
  | RET_YAZISI | 478 | | REDDEDILDI | 632 |
  | ONAY_YAZISI | 315 | | | |
  | TESPIT_TUTANAGI | 253 | | | |

  Kanonik sistem prompt'unun (kurallar, izinli değerler, arz/rica hiyerarşisi) tamamı `yazi_qwen2/canonical_system_prompt.txt` dosyasındadır.

### 3.4. Önerilen ince ayar hiperparametreleri (Qwen1 / Qwen2)

- Taban: `Qwen/Qwen2.5-7B-Instruct` (3B/14B da desteklenir)
- Framework: **eğitim** Unsloth / HuggingFace TRL (SFTTrainer) · **canlı servis** vLLM (multi-LoRA, merge yok)
- LoRA: `r=16` (veya 32), `alpha=32` (veya 64), `target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`
- Max seq len: 2048–4096 · Epoch: 3 (warmup 0.05) · LR: `2e-4` (QLoRA) / `5e-5` (LoRA 16-bit) · scheduler: cosine · precision: bf16

---

## 4. Veri Şeması

### 4.1. Qwen1 (Analiz) çıktısı → Qwen2 girdisi

`user` içeriği tek bir JSON string'idir:

```json
{
  "document_type": "ITIRAZ_IDARI_BASVURU",
  "sender_type": "VATANDAS",
  "primary_topic": "TEBLIGAT_ISLEMLERI",
  "requested_action": "2024/128 sayılı encümen kararının tebligatının yeniden yapılması",
  "key_information": [
    {"type": "PERSON", "value": "Ali Veli"},
    {"type": "DOCUMENT_NO", "value": "2024/128"},
    {"type": "REFERENCE_NO", "value": "E-2025-1142"}
  ],
  "missing_information": [],
  "summary": "Ali Veli, encümen kararı tebligatının usulsüz olduğunu belirterek yeniden tebliğ talep etmektedir.",
  "process_status": "INCELEMEDE",
  "selected_legislation": [
    {"law_name": "3071 sayılı Dilekçe Hakkının Kullanılmasına Dair Kanun", "article": "Madde 7", "reference_text": "..."}
  ]
}
```

### 4.2. Qwen2 (Yazıcı) çıktısı

```json
{
  "performed_actions": [
    {"action": "BASVURU_KAYDI", "detail": "Başvuru E-2025-1142 numarasıyla kaydedilmiştir."},
    {"action": "EVRAK_INCELEMESI", "detail": "İlgili karar ve tebligat mazbatası incelenmeye başlanmıştır."}
  ],
  "target_unit": "Yazı İşleri Müdürlüğü",
  "response_type": "BILGI_YAZISI",
  "process_status": "INCELEMEDE",
  "result_information": "Başvurunuz incelemeye alınmıştır; süreç değerlendirmesi devam etmektedir.",
  "process_information": "Başvuru kayıtlı olup tebligat usulü incelemesi sürmektedir.",
  "draft": "T.C.\nBELEDİYE BAŞKANLIĞI\nYazı İşleri Müdürlüğü\n\nSayı: E-2025-1142\nKonu: Tebligat İşlemleri Hk.\n\nSayın Ali Veli,\n...\nBilgilerinize rica ederim."
}
```

**İzinli değerler** (`canonical_system_prompt.txt` — bunlar dışında değer üretilmez):
- `response_type` (7): `BILGI_YAZISI`, `BILDIRIM_YAZISI`, `EKSIK_BILGI_BELGE_TAMAMLAMA_YAZISI`, `RET_YAZISI`, `YONLENDIRME_YAZISI`, `TESPIT_TUTANAGI`, `ONAY_YAZISI`
- `process_status` (5): `INCELEMEDE`, `TAMAMLANDI`, `EKSIK_BILGI_BEKLENIYOR`, `REDDEDILDI`, `YONLENDIRILDI`
- **Zorunlu kural:** `process_status = EKSIK_BILGI_BEKLENIYOR` ise `response_type` **mutlaka** `EKSIK_BILGI_BELGE_TAMAMLAMA_YAZISI` olur.

---

## 5. Resmî Yazışma Protokolleri

Üretilen `draft` metinleri T.C. Resmî Yazışma Usul ve Esaslarına göre valide edilir:

- **Arz / rica hiyerarşisi (kapanış ifadesi zorunlu):**
  - Gönderen **VATANDAS / ÖZEL_KURULUŞ** → "Bilgilerinizi rica ederim." / "Gereğini saygılarımla rica ederim."
  - **Üst makam** (Valilik, Kaymakamlık, Mahkeme, Savcılık, Bakanlık) → "Gereğini saygılarımla arz ederim."
  - **Eşdüzey kamu kurumu** → "Gereğini bilgilerinize arz ve rica ederim."
- **Başlık & muhatap:** `T.C.` + Başkanlık başlığı, `Sayı`, `Konu`, `Muhatap`, (varsa) `İlgi` bölümleri eksiksiz.
- **Mevzuat entegrasyonu:** Atıflar başlık satırına değil, gövde paragrafına, olaya uyarlanmış doğal cümle içinde yerleştirilir (referans metnin birebir kopyası değil). **Yalnızca** `selected_legislation` içindeki kanun/maddeler kullanılabilir; liste dışı madde uydurmak yasaktır.
- **Register ayrımı:** `process_information` = iç sistem notu, **tek cümle, hitapsız**; `draft` = tam resmî yazı, min. ~250 kelime.

---

## 6. 11 Müdürlük ve RAG Mevzuat Havuzu

Birim yönlendirmesi `app/maps/belediye_konu.json` içindeki deterministik `topic_to_unit` tablosuyla yapılır (model birim seçmez). 58 konu 11 müdürlüğe eşlenir:

| Müdürlük | Örnek mevzuat |
|---|---|
| İmar ve Şehircilik | 3194 İmar K., 6306 Kentsel Dönüşüm, 4708 Yapı Denetimi, 2942 Kamulaştırma |
| Fen İşleri | 4734 Kamu İhale, 4735 Sözleşmeler, 2918 Trafik, Otopark Yön. |
| Zabıta | 5326 Kabahatler, 1608, İşyeri Ruhsat Yön., Pazar Yerleri Yön. |
| Çevre Koruma ve Kontrol | 2872 Çevre K., Gürültü Kontrol Yön., Sanayi Hava Kirliliği Yön. |
| Temizlik İşleri | 2872 Çevre K., Sıfır Atık, Atık Yönetimi, Ambalaj Atığı Yön. |
| Veteriner İşleri | 5199 Hayvanları Koruma, 5996 Veteriner Hizmetleri, Kuduz/Çip Yön. |
| Sosyal Yardım İşleri | 5393 Belediye K. m.14, 5378 Engelliler, 2828 Sosyal Hizmetler, 2022 |
| Park ve Bahçeler | 2872 Çevre K., 6831 Orman, 5393 Belediye K. |
| Mali Hizmetler | 1319 Emlak V., 2464 Belediye Gelirleri, 213 VUK, 6183 AATUHK, 5018 |
| Hukuk İşleri | 6100 HMK, 2004 İİK m.89, 2942 Kamulaştırma, 7201 Tebligat |
| Yazı İşleri | 4982 Bilgi Edinme, 3071 Dilekçe Hakkı, 5070 E-İmza, Resmî Yazışma Yön. |

**RAG arama motoru** (`mevzuat_rag/`): dense retrieval **BGE-M3** + **BM25** sözcük araması, **RRF** ile birleştirme, **BGE-reranker-v2-m3** ile yeniden sıralama. Chunk = madde başına ~480 token. Aday havuzu k=20, retrieval ağırlığı 0.25.

Üretim sorgusu iki parçadır: `topic_to_rag[primary_topic]` (mevzuat dilindeki kısa anahtar) + `requested_action` (talep cümlesi; adres gibi duruyorsa düşülür). OCR metni, özet ve kişi/adres `key_information` sorguya girmez — mevzuat maddelerinde yoktur. İndeks (`chunks.jsonl`, vektörler) repoda yoktur; demo `rag.zip` ile Drive'dan yükler.

---

## 7. Kurulum ve Baştan Sona Çalıştırma

### 7.1. Bağımlılıklar

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

DeepSeek API'siyle **yeni sentetik veri** üretecekseniz (zorunlu değil, eğitim setleri hazır):

```bash
copy analiz_qwen1\.env.example analiz_qwen1\.env   # kendi DEEPSEEK_API_KEY değerinizi yazın (.env commit edilmez)
```

### 7.2. Veri (yeniden üretmek isteğe bağlı)

```bash
# Analiz (Qwen1) seti
cd analiz_qwen1
python generate_dataset_imar.py        # ... ve diğer 10 müdürlük script'i
python build_finetune_splits.py        # → veri/train|val|test.jsonl

# OCR seti
cd ../QWEN_VL_ocr
python download_dataset.py
python generate_dataset_1000.py --dataset_path ./turkish_law_dataset --output_dir ./dataset_1000
```

### 7.3. Eğitim (Colab A100 önerilir)

1. `QWEN_VL_ocr/qwen_ocr_baseline_test.ipynb` → `QWEN_VL_ocr/qwen_ocr_finetune_lora.ipynb`  (OCR LoRA + CER)
2. `analiz_qwen1/Qwen1_FineTune_Colab.ipynb`  → `lora_adapter_qwen1`
3. `yazi_qwen2/01_finetune_qwen25_7b.ipynb` → `lora_adapter_qwen2`, ardından `02_evaluate_qwen25_7b.ipynb` (baseline için `03_evaluate_base_model.ipynb`)

### 7.4. Uçtan uca demo (jüri)

Canlı koşu **Colab notebook**'tur: `app/notebooks/Qwen1_RAG_Demo_Colab.ipynb`. `app/src/gradio_app.py` aynı arayüzün okunabilir kaynak ikizidir.

Yükleme: **Instruct vLLM** (`analiz_lora` + `yazi_lora`, `max_loras=2`) → **VL vLLM** (OCR LoRA) → RAG **CPU**. Merge / ikinci 7B Instruct yok. `vllm` `requirements.txt` içindedir; Colab’da notebook kendi sürümünü kurar.

1. Runtime → A100 80 GB.
2. Drive'da şunlar dursun: `lora_adapter_qwen1/`, `lora_adapter_qwen2/`, `qwen_vl_7b_model/`, `qwen_vl_7b_lora_finetuned/`, `finetune_data_qwen1/` (`train.jsonl`, `belediye_konu.json`, `rag.zip` veya `rag/`).
3. **Run all** → ilk turda paket çakışması için kırmızı `RuntimeError` (bilinçli) → **Restart session** → tekrar **Run all**.
4. Gradio `share=True` linki açılır.

**Arayüz — Belediye Evrak Masası** (sol belge, orta memur + maddeler, sağ taslak):

![Belediye Evrak Masası arayüzü](docs/demo_ekran.png)

Statik önizleme: `docs/demo_onizleme_mock.html`.

### 7.5. GitHub'a koymak

Bu klasörde henüz remote yok; ilk `git commit` sonrası GitHub'da boş repo açıp `git remote add origin …` / `git push -u origin main` gerekir.

Push'a **girmez** (bilinçli, `.gitignore` + şartname Bölüm 7): `.env` / API anahtarı, `*.safetensors`, LoRA klasörleri, RAG vektörleri, OCR zip, Qwen2 tam jsonl. Anahtar için yalnızca `.env.example` vardır.

Push'a **girer:** kaynak, notebook'lar (`03_evaluate_base_model.ipynb` dahil), Qwen1 `veri/*.jsonl`, Qwen2 örnek jsonl, OCR A–D örnekleri (`QWEN_VL_ocr/data/`), `docs/`, `LICENSE` (MIT), bu README.

---

## 8. Modeller ve Veri: Repoda Ne Var, Ne Yok

Şartname **Bölüm 7** gereği: üçüncü taraf açık-ağırlık veya uygun-lisanslı olmayan modeller **depoya yüklenmez**; yalnızca erişim bağlantısı, sürüm, lisans ve kullanım talimatı dokümantasyonda belirtilir. Buna uygun olarak:

**Repoda VAR:** tüm kaynak kod, eğitim + değerlendirme notebook'ları (OCR baseline/LoRA, Qwen1 FT, Qwen2 `01`/`02`/`03`), sentetik veri üretim script'leri, Qwen1 eğitim setinin **tamamı** (`analiz_qwen1/veri/`), Qwen2 setinin **temsili örneği** (`yazi_qwen2/veri/*_ornek.jsonl`), OCR eğitim setinden **4 örnek görsel** (`QWEN_VL_ocr/data/`, A/B/C/D), sistem prompt'ları, eşleme tabloları, Gradio kaynak + Colab demo.

**Repoda YOK (kasıtlı):**

| Bileşen | Nerede / Nasıl | Lisans |
|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` (taban) | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Apache-2.0 |
| `Qwen/Qwen2.5-VL-7B-Instruct` (OCR taban) | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Apache-2.0 (Qwen lisansı) |
| `BAAI/bge-m3` (embedding) | [HuggingFace](https://huggingface.co/BAAI/bge-m3) | MIT |
| `BAAI/bge-reranker-v2-m3` | [HuggingFace](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Apache-2.0 |
| Eğitilmiş LoRA adaptörleri (`lora_adapter_qwen1/2`, OCR LoRA) | Notebook çıktısı; ağırlıklar Google Drive'da tutulur | MIT (bu proje) |
| Qwen2 tam eğitim seti (33 MB) | Script'lerle yeniden üretilir; örnek repoda | MIT (bu proje) |
| RAG üretilen indeks (`chunks.jsonl`, `bge-m3.npy`), OCR 1000’lik zip | Script'lerle yeniden üretilir; 4 örnek `QWEN_VL_ocr/data/` | — |

> Qwen2'nin ~33 MB'lık tam eğitim seti repoyu şişirmemek için doğrudan konmadı; yerine `yazi_qwen2/veri/` altında **370 satırlık temsili örnek** vardır. Tam set aynı pipeline ile yeniden üretilebilir veya talep üzerine paylaşılır. Gerçek kişisel veri (TCKN, ad, adres, telefon) hiçbir sette **paylaşılmaz**; tüm veri sentetiktir.

---

## 9. Değerlendirme Sonuçları

Tüm modeller **eğitimde kullanılmayan, aynı sabit test setleri** üzerinde ölçülmüştür (ince ayarlı model ile baseline **aynı** holdout set üzerinde karşılaştırılır).

### 9.1. OCR — Qwen2.5-VL-7B (test: 100 belge, 21.571 Türkçe karakter)

| Model | Ortalama CER ↓ | Türkçe karakter doğruluğu ↑ |
|---|---:|---:|
| Baseline (saf Instruct, LoRA yok) | 0.4283 | %78.55 |
| **İnce ayarlı (LoRA)** | **0.1076** | **%85.90** |

> LoRA, karakter hata oranını **~4 kat** düşürdü (0.4283 → 0.1076); Türkçe karakter doğruluğu +7.35 puan. Ölçüm Unsloth/HF notebook'larında; canlı OCR **vLLM + OCR LoRA**.

### 9.2. Analiz — Qwen1 (Qwen2.5-7B-Instruct, test: 241 kayıt)

| Metrik | Baseline (LoRA yok) | **İnce ayarlı (LoRA)** |
|---|---:|---:|
| `document_type` doğruluk | %70.1 | **%85.5** |
| `sender_type` doğruluk | %78.0 | **%95.4** |
| `primary_topic` doğruluk | — | **%72.2** |
| `missing_information` doğruluk | %74.3 | **%97.9** |

> **`primary_topic` neden yalnız ince ayarlı model için?** Ham (baseline) model projeye özgü **58 konu taksonomisini** bilmez; etiketleri serbest metin olarak üretir, dolayısıyla birebir eşleşme kıyası anlamsızdır (tabloda "—"). İnce ayar sonrası model 58 konuyu öğrenir ve `primary_topic` doğruluğu **%72.2**'ye ulaşır.
>
> **`missing_information` doğruluk** = modelin, evrakta **eksik bilgi olup olmadığını** (liste boş mu, dolu mu) doğru tahmin etme oranı. Örn. bir dilekçede ada/parsel numarası eksikse bunu "eksik" olarak işaretlemesi beklenir. Canlı analiz **vLLM + `analiz_lora`**.

### 9.3. Yazıcı — Qwen2 (Qwen2.5-7B-Instruct, aynı 522 kayıtlık test seti)

Ham model ile son ince ayar, **aynı 522 kayıtlık holdout** üzerinde. Önceki (farklı veri setiyle yapılan) 1. ince ayar bu kıyasa dahil edilmez.

| Model | Notebook | `response_type` doğruluk ↑ |
|---|---|---:|
| Baseline (saf Instruct, LoRA yok) | `03_evaluate_base_model.ipynb` | %50.0 |
| **İnce ayarlı (LoRA)** | `02_evaluate_qwen25_7b.ipynb` | **%82.18** |

> LoRA, yanıt türü doğruluğunu **+32.18 puan** yükseltti (50.0 → 82.18). Bu iki notebook Unsloth/HF ile ölçer; canlı yazı üretimi **vLLM + `yazi_lora`**.

### 9.4. RAG Mevzuat Motoru (aynı 126 kavramsal sorguluk gold set)

Tüm koşular **aynı dondurulmuş 126 sorgu** üzerinde, tek değişken değiştirilerek yapılmıştır. İlerleme sağlayan koşular:

| Koşu | Retriever | Kavramsal ceza | H@1 ↑ | H@3 ↑ | H@5 | MRR | miss ↓ |
|---|---|---|---:|---:|---:|---:|---:|
| Taban | bge-m3 | hayır | 59.5 | 84.1 | 92.9 | 0.729 | 6 |
| **Üretim (seçilen)** | **bge-m3** | **evet** | **69.0** | **86.5** | **92.9** | **0.789** | **4** |
| Alternatif | e5-large | evet | 70.6 | 84.9 | 91.3 | 0.792 | 6 |

> **En büyük kazanç:** kavramsal ceza (`article_rank`), bge-m3 + k=20'de H@1'i **59.5 → 69.0** çekti (+9.5 puan), maliyetsiz. `multilingual-e5-large` yalnız H@1'de +1.6 önde ama H@3/H@5/miss'te geride; yazı ajanı top-3 kullandığından **bge-m3 üretime alındı**.
>
> **Chunk 480 → 1024 token denemesi (tek satır):** üretim ayarında H@1 69.0 / H@3 84.9 / H@5 91.3 / miss 6 → 480'i **geçmedi** (H@3 ve miss kötüleşti), reddedildi. Denenip elenen diğerleri: LoRA reranker (H@1 60.3, kazanç yok), k=40 (süre ~2×). Tüm sistemin çalışan hâli `app/notebooks/Qwen1_RAG_Demo_Colab.ipynb` içindedir.

---

## 10. Ticarileşme ve Ölçeklenebilirlik

**Gerçek dünya uygulanabilirliği.** Sistem, Türkiye'deki ~1.390 belediyenin günlük evrak/yazışma yükünü hedefler. Girdi (dilekçe/CİMER/kurum yazısı), süreç durumları ve çıktı (resmî yazı taslağı) hâlihazırdaki EBYS (Elektronik Belge Yönetim Sistemi) akışlarıyla birebir örtüşür; entegrasyon bir **API/eklenti** olarak konumlanabilir.

- **İnsan-döngüde (HITL) tasarım** kurumsal güveni ve mevzuat sorumluluğunu korur: model öneri üretir, **memur onaylar**. Bu, kamu kurumlarında benimsenmenin ön koşuludur.
- **Maliyet/ölçek:** Canlı servis **vLLM**. Analiz + yazı **multi-LoRA** (`max_loras=2`) ile tek Instruct 7B motorda; OCR ayrı VL motorunda. İkinci 7B taban ve LoRA merge yok — VRAM yaklaşık yarıya iner, tek A100 yeter. Adaptörler küçük; yeni birim/mevzuat için tam yeniden eğitim gerekmez.
- **Veri gizliliği:** Tüm yığın açık ağırlıklı modeller (Qwen2.5, BGE) ve açık kaynak kütüphanelerle çalışır; **kurum içi (on-premise)** kurulabilir, veri dışarı çıkmaz — kamu için kritik.
- **Genişleme yolu:** Belediyeden bakanlık/il müdürlüğü yazışmalarına, farklı mevzuat havuzlarına RAG korpusu değiştirilerek taşınabilir; mimari domain-bağımsızdır.

## 11. Lisans, Etik ve Kaynaklar

- **Lisans:** MIT (bkz. `LICENSE`). Şartname Bölüm 7'nin açık kaynak lisans zorunluluğunu karşılar.
- **Etik / veri (şartname §5.5):** Yarışmada gerçek belediye/CİMER/EBYS evrakı kullanılmaz. Bu projedeki iş bölümü:
  - **Görev 1–2 (analiz + yazı):** DeepSeek ile üretilmiş **kurgu** dilekçe ve resmî yazışma taslakları (`analiz_qwen1/veri/`, `yazi_qwen2/`).
  - **OCR:** Hugging Face üzerindeki açık derlemeden alınan **kamuya açık yargı kararı metinleri**; HTML/PNG olarak yeniden basılır. Taranmış mahkeme dosyası, vatandaş dilekçesi veya kurum içi evrak değildir.
  - **RAG:** kamuya açık mevzuat PDF’leri.
  Gerçek TCKN / adres / telefon paylaşılmaz (sentetik kurguda da uydurulmuş değerler kullanılır).
- **Üçüncü taraf lisansları:** Kullanılan tüm modeller ve kütüphaneler kendi lisans koşullarına uygun biçimde kullanılır (Bölüm 8 tablosu).
- **Kaynaklar:**
  - Qwen2.5 / Qwen2.5-VL — Alibaba Cloud (Apache-2.0)
  - vLLM — canlı çıkarım (multi-LoRA)
  - BGE-M3, BGE-reranker-v2-m3 — BAAI
  - OCR kaynak metin derlemesi — [`erdem-erdem/Turkish-Law-Documents-700k-clustered`](https://huggingface.co/datasets/erdem-erdem/Turkish-Law-Documents-700k-clustered) (Hugging Face). Birincil kurumlar: [Yargıtay Karar Arama](https://karararama.yargitay.gov.tr/), [Danıştay Karar Arama](https://kararara.danistay.gov.tr/). HF kartına göre belgeler kamu kaydıdır.
  - Sentetik evrak üretimi — DeepSeek API
