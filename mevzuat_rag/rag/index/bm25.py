"""BM25 sözlük araması (dense'in yakalayamadığı tam eşleşmeler için).

Türkçe sondan eklemeli: `madde`, `maddesi`, `maddesinde`, `maddelerden`
BM25 için dört ayrı kelimedir ve hiçbiri diğeriyle eşleşmez. Gerçek bir
gövdeleyici (Zemberek vb.) en doğrusu olurdu; burada kaba ama ucuz bir
yaklaşım kullanılıyor: kelimeyi ilk 6 karaktere kırpmak. Türkçede kök
çoğunlukla ilk hecelerde olduğu için bu, ek çeşitliliğinin büyük kısmını
tek biçime indirir.

Sayılar KIRPILMAZ: `5393`, `13/B` gibi ifadeler BM25'in asıl değer kattığı
yer ve kırpmak onları bozar.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_STEM_LEN = 6
_TOKEN_RE = re.compile(r"[0-9]+(?:/[0-9A-Za-zÇĞİÖŞÜçğıöşü]+)?|[a-zçğıöşü]+")

# Mevzuat metninde her yerde geçen, ayırt ediciliği olmayan kelimeler.
STOP = {
    "madde", "fıkra", "bent", "kanun", "yöneti", "hüküm", "bu", "ve", "ile",
    "veya", "için", "olan", "göre", "gibi", "her", "bir", "olarak", "dair",
    "ilişki", "hakkın", "belge", "sayılı",
}


def tr_lower(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower()


def tokenize(text: str) -> list[str]:
    out = []
    for tok in _TOKEN_RE.findall(tr_lower(text)):
        if tok[0].isdigit():
            out.append(tok)  # sayı/madde numarası olduğu gibi
            continue
        if len(tok) < 3:
            continue
        stem = tok[:_STEM_LEN]
        if stem in STOP:
            continue
        out.append(stem)
    return out


class BM25Index:
    def __init__(self, texts: list[str], ids: list[str]) -> None:
        if len(texts) != len(ids):
            raise ValueError("texts ve ids ayni uzunlukta olmali")
        self.ids = ids
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        best = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(self.ids[i], float(scores[i])) for i in best]


def rrf_fuse(
    rankings: list[list[str]], k: int = 60, weights: list[float] | None = None
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion.

    Ham skorlar toplanmaz: BM25 skoru sınırsız ve korpusa bağlı, kosinüs ise
    [-1,1]. Toplamak, ölçekleri farklı iki şeyi toplamak olur ve BM25 baskın
    çıkar. RRF yalnız SIRA kullanır, ölçek sorununu tamamen ortadan kaldırır.
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
