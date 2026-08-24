"""4 embedding modeli için ortak arayüz.

Her modelin kendi prompt konvansiyonu var ve bunlara uyulması adil kıyasın
ön şartı: e5 `query:`/`passage:` öneki olmadan belirgin kötüleşir, jina-v3
görev adaptörü seçilmezse genel amaçlı moda düşer. Konvansiyonu atlamak
"model kötü çıktı" sonucunu üretir, oysa yanlış çağrılmıştır.
"""

from __future__ import annotations

import numpy as np

from ..config import EMBEDDING_MODELS, EMBEDDING_NORMALIZE, EMBEDDING_BATCH_SIZE


class Embedder:
    """Tek bir embedding modelini sarmalar.

    Kullanım:
        emb = Embedder("bge-m3")
        vecs = emb.encode_passages([...])   # chunk metinleri
        qvec = emb.encode_queries([...])    # arama sorguları
    """

    def __init__(self, name: str, device: str = "cuda", batch_size: int | None = None):
        if name not in EMBEDDING_MODELS:
            raise KeyError(f"Bilinmeyen model: {name}. Seçenekler: {list(EMBEDDING_MODELS)}")
        self.name = name
        self.spec = EMBEDDING_MODELS[name]
        self.device = device
        self.batch_size = batch_size or EMBEDDING_BATCH_SIZE
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            self.spec["hf_id"],
            device=self.device,
            trust_remote_code=self.spec.get("trust_remote_code", False),
            # `unpad_inputs` gibi ayarlar modelin __init__'ine değil
            # config'ine ait; model_kwargs ile geçirmek TypeError veriyor.
            config_kwargs=self.spec.get("config_kwargs") or None,
        )
        # Kırpma sınırını chunk bütçemize sabitle; modelin varsayılanı
        # (8192) gereksiz bellek ayırır.
        from ..config import MAX_CHUNK_TOKENS

        self._model.max_seq_length = min(self.spec["max_seq"], MAX_CHUNK_TOKENS + 32)
        return self._model

    def _encode(self, texts: list[str], prefix: str, task: str | None) -> np.ndarray:
        model = self._load()
        payload = [prefix + t for t in texts] if prefix else texts
        kwargs = {
            "batch_size": self.batch_size,
            "normalize_embeddings": EMBEDDING_NORMALIZE,
            "convert_to_numpy": True,
            # Kapalı: 500 batch'lik koşuda ilerleme çubuğu log dosyasını
            # yüz binlerce karakter şişiriyor. Süre zaten model başına yazılıyor.
            "show_progress_bar": False,
        }
        if task:  # jina-v3 görev adaptörü
            kwargs["task"] = task
        return model.encode(payload, **kwargs)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(
            texts, self.spec.get("passage_prefix", ""), self.spec.get("passage_task")
        )

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(
            texts, self.spec.get("query_prefix", ""), self.spec.get("query_task")
        )

    def unload(self):
        """Sıradaki modele geçmeden VRAM'i boşalt."""
        import gc

        import torch

        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
