"""Merkezi sabitler.

Buradaki her sabit chunk sınırlarını veya karşılaştırmanın adilliğini etkiler.
Değiştirirsen chunks.jsonl'i ve tüm embedding'leri yeniden üretmen gerekir.
"""

from __future__ import annotations

from pathlib import Path

# --- Yollar ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_DIR = PROJECT_ROOT.parent  # ham PDF'ler: ajan_veri/ kökü + müdürlük klasörleri
OUT_DIR = PROJECT_ROOT / "out"

CHUNKS_FILE = OUT_DIR / "chunks.jsonl"
MANIFEST_FILE = OUT_DIR / "manifest.jsonl"
INGEST_REPORT_FILE = OUT_DIR / "ingest_report.json"

# --- Chunking -------------------------------------------------------------
#
# Karar: 1 chunk = 1 madde. Kısa madde komşusuyla BİRLEŞTİRİLMEZ.
# Gerekçe: iki madde aynı chunk'a girerse retrieval sonucunda "hangi maddeye
# atıf yapıyorum" belirsizleşir; yarışmada madde atfı doğruluğu puanlanıyor.
# Kısa maddenin zayıf embedding'i birleştirmeyle değil, CONTEXT_PREFIX ile
# çözülür.

# Toplam chunk bütçesi (prefix DAHİL). 512 değil 480:
# multilingual-e5-large'ın pozisyon gömmesi 512 token ile sınırlı, üstü
# sessizce kırpılır. 480 tavanı, 4 modelin hiçbirinde kırpma olmamasını
# garanti eder — aksi hâlde e5 haksız yere düşük skor alır ve kıyas bozulur.
# 1024 deneyi: rag/out/chunk1024/ — üretim B'de H@3/miss kötüleşti, geri alındı.
MAX_CHUNK_TOKENS = 480

# Gövde hedefi (prefix hariç). Prefix ~60-80 token tutuyor.
TARGET_BODY_TOKENS = 380

# Uzun madde bölünürken örtüşme. Sadece AYNI madde içinde uygulanır;
# madde sınırında örtüşme yoktur (iki madde birbirine sızmamalı).
OVERLAP_TOKENS = 64

# Bu eşiğin altındaki madde "kısa" sayılır — sadece raporlama için,
# davranışı değiştirmez (birleştirme yok).
SHORT_ARTICLE_TOKENS = 40

# Ortak tokenizer. 4 modelin kendi tokenizer'ıyla ölçersek her model farklı
# chunk sınırı görür ve kıyas adil olmaz. Tek yardımcı cetvel kullanılır.
TOKENIZER_ID = "xlm-roberta-base"

# --- Karşılaştırılacak embedding modelleri --------------------------------
#
# Hepsi açık ağırlık. Her birinin kendi prompt konvansiyonu var (bkz.
# embed/models.py) — konvansiyona uyulmazsa model kendi eğitildiği
# dağılımın dışında çalışır ve haksız yere kaybeder.

EMBEDDING_MODELS = {
    "bge-m3": {
        "hf_id": "BAAI/bge-m3",
        "dim": 1024,
        "max_seq": 8192,
        "trust_remote_code": False,
        "query_prefix": "",  # bge-m3 prefix istemez
        "passage_prefix": "",
    },
    "e5-large": {
        "hf_id": "intfloat/multilingual-e5-large",
        "dim": 1024,
        "max_seq": 512,  # BAĞLAYICI KISIT — MAX_CHUNK_TOKENS bunu gözetir
        "trust_remote_code": False,
        "query_prefix": "query: ",  # e5 bu önekler olmadan belirgin kötüleşir
        "passage_prefix": "passage: ",
    },
    "gte-base": {
        "hf_id": "Alibaba-NLP/gte-multilingual-base",
        "dim": 768,
        "max_seq": 8192,
        "trust_remote_code": True,
        "query_prefix": "",
        "passage_prefix": "",
        # gte'nin özel attention kodu (`new-impl`) değişken uzunluklu
        # dizileri "unpad" edip tek tampona diziyor; RTX 4060 + cu124'te bu
        # yol `CUBLAS_STATUS_NOT_SUPPORTED` ile çöküyor (kısa ve eşit boylu
        # metinlerde çökmüyor, o yüzden ilk denemede görünmedi). Standart
        # padded attention'a düşürülüyor — sonuç aynı, yalnız biraz yavaş.
        "config_kwargs": {
            "unpad_inputs": False,
            "use_memory_efficient_attention": False,
        },
    },
    "jina-v3": {
        "hf_id": "jinaai/jina-embeddings-v3",
        "dim": 1024,  # Matryoshka, 1024 tam boyut
        "max_seq": 8192,
        "trust_remote_code": True,
        # jina-v3 prefix yerine görev LoRA adaptörü kullanır:
        "query_task": "retrieval.query",
        "passage_task": "retrieval.passage",
        "query_prefix": "",
        "passage_prefix": "",
    },
}

# Atıf zenginleştirmesi (bir maddenin atıf yaptığı maddelerin BAŞLIKLARINI
# chunk'a eklemek). Recall'ı artırması umuluyordu ama başka bir maddenin
# kelimelerini chunk'a enjekte ettiği için yanlış eşleşme de üretebilir.
# Ablasyonla ölçülüyor — varsayılan, ölçüm sonucuna göre belirlenecek.
XREF_ENABLED = False

EMBEDDING_NORMALIZE = True
# RTX 4060 Laptop 8GB. 32 muhtemelen sığar ama koşu ortasında OOM olması
# yarım saati çöpe atar; 16 ile hız farkı küçük, risk yok.
EMBEDDING_BATCH_SIZE = 16

# --- Reranker -------------------------------------------------------------
#
# Aday havuzu 20: jina-v3 dense'in H@20'si %98 ve K=50'ye kadar artmıyor,
# yani 20'nin ötesi reranker'a yeni doğru cevap taşımıyor, yalnız çeldirici
# ekliyor ve yavaşlatıyor.
RERANK_CANDIDATES = 20
RERANK_BATCH_SIZE = 16

RERANKER_MODELS = {
    "bge-rr-v2-m3": {
        "hf_id": "BAAI/bge-reranker-v2-m3",
        "trust_remote_code": False,
    },
    "jina-rr-v2": {
        "hf_id": "jinaai/jina-reranker-v2-base-multilingual",
        "trust_remote_code": True,
    },
}

# --- Değerlendirme --------------------------------------------------------

EVAL_TESTSET_FILE = PROJECT_ROOT / "eval" / "testset.jsonl"
EVAL_K_VALUES = (1, 3, 5, 10)
