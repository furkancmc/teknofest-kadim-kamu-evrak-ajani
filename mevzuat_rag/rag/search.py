"""Arama servisi — Qwen ile yazı ajanları arasındaki katman.

Boru hattı (ölçümle seçildi, bkz. README):
    sorgu -> [dense bge-m3 | BM25] -> RRF -> bge-reranker-v2-m3 -> maddeler

Qwen tek sorgu değil 3-5 `rag_queries` üretiyor; `search()` liste alır, her
sorguyu ayrı çalıştırır, sonuçları RRF ile birleştirir ve AYNI MADDEYİ
tekrarlamaz.

Müdürlük bir FİLTRE değil, yalnız opsiyonel puan artırıcıdır ve varsayılan
olarak kapalıdır: Qwen'in `primary_unit` doğruluğu %87, yani her sekiz
evraktan birinde yanlış. Filtre olsaydı o evraklarda doğru mevzuat hiç
gelmezdi.

Terminalden denemek için:
    .venv/Scripts/python.exe -m rag.search
    .venv/Scripts/python.exe -m rag.search "sokak köpeği kaydı kimin görevi"
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from . import rerankers
from .config import CHUNKS_FILE, OUT_DIR, RERANK_BATCH_SIZE, RERANK_CANDIDATES
from .embed.models import Embedder
from .index.bm25 import BM25Index, rrf_fuse
from .scope import classify_scope

VECTOR_DIR = OUT_DIR / "vectors"
RETRIEVER = "bge-m3"  # MIT lisanslı ve tam boru hattında jina-v3'ten iyi
# Taban bge-reranker. rag/out/reranker_lora varsa onu kullan.
RERANKER = (
    "bge-m3-lora"
    if (OUT_DIR / "reranker_lora" / "adapter_config.json").is_file()
    else "bge-m3"
)
POOL = 50  # füzyondan önce her yoldan alınan aday
UNIT_BOOST = 0.15  # yalnız `unit` verilirse uygulanır
RERANK_RETRIEVAL_WEIGHT = 0.25  # holdout deneyinde seçildi; atıfta uygulanmaz

_ARTICLE_REF_RE = re.compile(
    r"(?:\bmadde(?:si|sinin|de|den)?\s*\d+(?:\s*/\s*[a-zçğıöşü0-9]+)?\b|"
    r"\b\d+(?:\s*/\s*[a-zçğıöşü0-9]+)?\s*(?:inci|ıncı|uncu|üncü|\.?)\s*"
    r"madde(?:si|sinin|de|den)?\b)",
    re.IGNORECASE,
)


def is_citation_query(query: str) -> bool:
    """Açık madde numarası isteyen sorguları kavramsal rotadan ayırır."""
    return bool(_ARTICLE_REF_RE.search(query))


def blend_reranker_with_retrieval(
    query: str,
    reranked_ids: list[str],
    retrieval_ids: list[str],
    weight: float = RERANK_RETRIEVAL_WEIGHT,
) -> list[str]:
    """Kavramsal sorguda cross-encoder ve retrieval sırasını birleştirir.

    Atıf sorguları mevcut %100 H@1 rotasında bırakılır — retrieval sırasını
    karıştırmak atıf H@1'ini 100'den 75'e düşürüyor (ölçüldü). Kavramsal
    ağırlık, 50 development / 50 holdout deneyinde H@10'u düşürmeden seçildi.
    """
    if weight <= 0 or is_citation_query(query):
        return reranked_ids
    return [
        cid
        for cid, _ in rrf_fuse(
            [reranked_ids, retrieval_ids],
            weights=[1.0, weight],
        )
    ]


@dataclass
class Provision:
    """Yazı ajanının atıf yapabileceği tek bir hüküm."""

    doc_id: str
    chunk_id: str
    kanun: str
    madde: str
    baslik: str
    metin: str  # tam madde metni (alıntı için)
    ozet: str  # ilk ~300 karakter
    sayfa: str
    kaynak_dosya: str
    birimler: list[str]
    # `skor` geriye uyumlu ana güven alanıdır ve GERÇEK cross-encoder
    # reranker skorunu taşır. RRF skoru güven değeri değildir; yalnız farklı
    # sıralamaları birleştirmek için kullanılır ve ayrıca raporlanır.
    skor: float
    reranker_skoru: float
    rrf_skoru: float
    siralama_skoru: float
    hangi_sorgudan: list[str] = field(default_factory=list)

    def to_dict(self, full_text: bool = True) -> dict:
        d = {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "kanun": self.kanun,
            "madde": self.madde,
            "baslik": self.baslik,
            "sayfa": self.sayfa,
            "kaynak_dosya": self.kaynak_dosya,
            "birimler": self.birimler,
            "skor": round(self.skor, 4),
            "reranker_skoru": round(self.reranker_skoru, 4),
            "rrf_skoru": round(self.rrf_skoru, 6),
            "siralama_skoru": round(self.siralama_skoru, 6),
            "hangi_sorgudan": self.hangi_sorgudan,
        }
        d["metin"] = self.metin if full_text else self.ozet
        return d


@dataclass
class SearchDecision:
    """Son aramanın neden çalıştığını veya neden geri çekildiğini açıklar."""

    cevaplanabilir: bool
    neden: str
    en_iyi_reranker_skoru: float | None = None
    skor_farki: float | None = None
    atlanan_sorgular: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cevaplanabilir": self.cevaplanabilir,
            "neden": self.neden,
            "en_iyi_reranker_skoru": (
                round(self.en_iyi_reranker_skoru, 4)
                if self.en_iyi_reranker_skoru is not None
                else None
            ),
            "skor_farki": round(self.skor_farki, 4) if self.skor_farki is not None else None,
            "atlanan_sorgular": self.atlanan_sorgular,
        }


class Searcher:
    """Ağır bileşenler bir kez yüklenir; sonraki aramalar hızlıdır."""

    def __init__(
        self,
        retriever: str = RETRIEVER,
        reranker: str = RERANKER,
        candidates: int = RERANK_CANDIDATES,
        retrieval_weight: float = RERANK_RETRIEVAL_WEIGHT,
    ) -> None:
        self.retriever = retriever
        self.reranker_name = reranker
        # Aday sayısı ve blend ağırlığı reranker'a bağlıdır: güçlü bir
        # sıralayıcı derin havuzu kazanca çevirir, zayıf olan çeldiriciye
        # yenilir. Bu yüzden ikisi de sabit değil, kurucu parametresi.
        self.candidates = candidates
        self.retrieval_weight = retrieval_weight
        self.chunks = [json.loads(l) for l in CHUNKS_FILE.open(encoding="utf-8")]
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self.order = json.loads((VECTOR_DIR / "chunk_ids.json").read_text(encoding="utf-8"))
        self.mat = np.load(VECTOR_DIR / f"{retriever}.npy")
        self.last_decision = SearchDecision(False, "not_searched")

    @cached_property
    def bm25(self) -> BM25Index:
        return BM25Index([c["text"] for c in self.chunks], [c["chunk_id"] for c in self.chunks])

    @cached_property
    def embedder(self) -> Embedder:
        return Embedder(self.retriever)

    @cached_property
    def reranker(self) -> rerankers.Reranker:
        return rerankers.get(self.reranker_name)

    def warmup(self) -> None:
        """Bütün ağır bileşenleri şimdi yükler.

        `reranker` yalnız sarmalayıcıyı kurar; asıl model ilk `score()`
        çağrısında iniyor. Ölçüm koşularında bu, ilk sorgunun süresine
        model yükleme maliyetini ekliyordu.
        """
        _ = self.embedder, self.bm25
        self.reranker.load()

    # --- tek sorgu --------------------------------------------------------

    def _candidates(self, query: str) -> list[str]:
        # Her yoldan alınan aday, reranker'a verilecek sayının en az iki katı
        # olmalı: RRF iki listeyi birleştirdiği için ilk `candidates` sonuç
        # tek bir yoldan gelebilir.
        pool = max(POOL, 2 * self.candidates)
        qvec = self.embedder.encode_queries([query])[0]
        sims = qvec @ self.mat.T
        dense = [self.order[j] for j in np.argsort(-sims)[:pool]]
        sparse = [cid for cid, _ in self.bm25.search(query, pool)]
        return [cid for cid, _ in rrf_fuse([dense, sparse])][: self.candidates]

    def _rerank(self, query: str, ids: list[str]) -> list[tuple[str, float]]:
        pairs = [(query, self.by_id[c]["text"]) for c in ids]
        scores = self.reranker.score(pairs, batch_size=RERANK_BATCH_SIZE)
        score_by_id = dict(zip(ids, scores))
        reranked_ids = sorted(ids, key=lambda cid: -score_by_id[cid])
        final_ids = blend_reranker_with_retrieval(
            query, reranked_ids, ids, weight=self.retrieval_weight
        )
        return [(cid, score_by_id[cid]) for cid in final_ids]

    # --- çok sorgu --------------------------------------------------------

    def search(
        self,
        queries: str | list[str],
        top_k: int = 8,
        unit: str | None = None,
        abstain: bool = True,
        min_reranker_score: float | None = None,
        ensure_query_coverage: bool = True,
        max_per_document: int | None = None,
    ) -> list[Provision]:
        if isinstance(queries, str):
            queries = [queries]
        queries = [q.strip() for q in queries if q and q.strip()]
        if not queries:
            self.last_decision = SearchDecision(False, "empty_query")
            return []

        skipped_queries: list[str] = []
        if abstain:
            scoped = [(q, classify_scope(q)) for q in queries]
            skipped_queries = [q for q, decision in scoped if not decision.in_scope]
            queries = [q for q, decision in scoped if decision.in_scope]
            if not queries:
                reasons = sorted({decision.reason for _, decision in scoped})
                self.last_decision = SearchDecision(
                    False,
                    reasons[0] if len(reasons) == 1 else "multiple_out_of_scope_reasons",
                    atlanan_sorgular=skipped_queries,
                )
                return []

        # Sorgu başına sıralı chunk listesi + hangi sorgunun getirdiği
        rankings: list[list[str]] = []
        best_score: dict[str, float] = {}
        source_q: dict[str, list[str]] = {}
        for q in queries:
            ranked = self._rerank(q, self._candidates(q))
            rankings.append([cid for cid, _ in ranked])
            for cid, sc in ranked:
                if sc > best_score.get(cid, -1e9):
                    best_score[cid] = sc
                source_q.setdefault(cid, []).append(q)

        fused = rrf_fuse(rankings)

        # Birim boost Top-K seçildikten sonra uygulanırsa dışarıda kalan bir
        # sonucu içeri taşıyamaz. Ham RRF ve gerçek sıralama skoru bu yüzden
        # bütün adaylar üzerinde önceden hesaplanır.
        raw_rrf = dict(fused)
        ranking_score: dict[str, float] = {}
        for cid, rrf_score in fused:
            c = self.by_id[cid]
            adjusted = rrf_score
            if unit and unit in c["units"]:
                adjusted += UNIT_BOOST * rrf_score
            ranking_score[cid] = adjusted

        ranked_candidates = sorted(ranking_score, key=lambda cid: -ranking_score[cid])

        # Qwen'in her alt sorgusu ayrı bir hukuki facet'i temsil eder. Salt
        # RRF, iki benzer alt sorgunun aynı belgeyi sonuçlarda çoğaltmasına ve
        # üçüncü facet'in kaybolmasına yol açabiliyor. Her alt sorgudan en iyi
        # benzersiz maddeyi önce havuza al; kalan yerleri birleşik sıralamayla
        # doldur.
        #
        # DENENDİ VE REDDEDİLDİ: garantiyi tek maddeden `top_k / sorgu sayısı`
        # kotasına çıkarmak. Gerekçe makuldü (dört facet 8 yere bölününce facet
        # başına iki sıra kalıyor) ama evrak setinde ölçüm tersini söyledi:
        # ilk 8'de bütün facet'leri yakalama %76.7'den %70.0'a düştü. Kotanın
        # zorla soktuğu ikinci-üçüncü maddeler, RRF'in daha iyi bulduğu
        # adayları listeden atıyor.
        forced: list[str] = []
        forced_articles: set[tuple[str, str]] = set()
        if ensure_query_coverage and len(rankings) > 1:
            for ranking in rankings:
                for cid in ranking:
                    c = self.by_id[cid]
                    key = (c["doc_id"], c["article_label"])
                    if key not in forced_articles:
                        forced.append(cid)
                        forced_articles.add(key)
                        break

        # Çok sorguda belge başına doğal kota: 2 sorguda 4, 3 sorguda 3,
        # 4-5 sorguda 2 madde. `0` verilirse kota tamamen kapatılabilir.
        document_cap = max_per_document
        if document_cap is None and len(rankings) > 1:
            document_cap = max(2, (top_k + len(rankings) - 1) // len(rankings))
        if document_cap == 0:
            document_cap = None

        selected: list[str] = []
        seen_article: set[tuple[str, str]] = set()
        doc_counts: dict[str, int] = {}

        def add_candidate(cid: str, enforce_cap: bool) -> bool:
            c = self.by_id[cid]
            key = (c["doc_id"], c["article_label"])
            if key in seen_article:
                return False
            if enforce_cap and document_cap is not None:
                if doc_counts.get(c["doc_id"], 0) >= document_cap:
                    return False
            selected.append(cid)
            seen_article.add(key)
            doc_counts[c["doc_id"]] = doc_counts.get(c["doc_id"], 0) + 1
            return True

        # Facet garantileri belge kotasından etkilenmez; aksi durumda aynı
        # kanunun farklı hükümlerini isteyen iki facet'ten biri kaybolabilir.
        for cid in forced:
            add_candidate(cid, enforce_cap=False)
        for cid in ranked_candidates:
            if len(selected) >= top_k:
                break
            add_candidate(cid, enforce_cap=True)
        # Küçük veya tek-belgeli korpuslarda kota listeyi eksik bırakmasın.
        for cid in ranked_candidates:
            if len(selected) >= top_k:
                break
            add_candidate(cid, enforce_cap=False)

        selected = sorted(selected[:top_k], key=lambda cid: -ranking_score[cid])

        # Aynı maddenin farklı parçaları yukarıda tek kayda indirildi. Yazı
        # ajanına tam madde metni ve üç ayrı skor türü gönderilir.
        out: list[Provision] = []
        for cid in selected:
            c = self.by_id[cid]
            reranker_score = best_score[cid]

            law = f"{c['law_no']} sayılı {c['canonical_title']}" if c["law_no"] else c["canonical_title"]
            body = c["raw_text"].strip()
            out.append(
                Provision(
                    doc_id=c["doc_id"],
                    chunk_id=cid,
                    kanun=law,
                    madde=c["article_label"],
                    baslik=c["article_title"] or "-",
                    metin=body,
                    ozet=" ".join(body.split())[:300],
                    sayfa=f"{c['page_start']}-{c['page_end']}" if c["page_start"] != c["page_end"] else str(c["page_start"]),
                    kaynak_dosya=c["source_files"][0],
                    birimler=c["units"],
                    skor=reranker_score,
                    reranker_skoru=reranker_score,
                    rrf_skoru=raw_rrf[cid],
                    siralama_skoru=ranking_score[cid],
                    hangi_sorgudan=source_q.get(cid, []),
                )
            )

        # Sonuçlar kullanıcının ve arayüzün gördüğü gerçek eşleşme (reranker)
        # skoruna göre en ilgiliden en az ilgiliye (en yüksekten en düşüğe) sıralanır.
        out.sort(key=lambda p: -p.skor)

        top_score = out[0].reranker_skoru if out else None
        second_score = out[1].reranker_skoru if len(out) > 1 else None
        margin = top_score - second_score if top_score is not None and second_score is not None else None
        if (
            abstain
            and min_reranker_score is not None
            and top_score is not None
            and top_score < min_reranker_score
        ):
            self.last_decision = SearchDecision(
                False, "low_reranker_score", top_score, margin, skipped_queries
            )
            return []

        self.last_decision = SearchDecision(
            bool(out),
            "retrieved_with_skipped_queries" if skipped_queries and out else (
                "retrieved" if out else "no_candidates"
            ),
            top_score,
            margin,
            skipped_queries,
        )
        return out


# --- terminal -------------------------------------------------------------


def _print(results: list[Provision], full: bool) -> None:
    if not results:
        print("  (sonuc yok)")
        return
    for i, p in enumerate(results, 1):
        print(f"\n{i}. {p.kanun}")
        print(f"   {p.madde} | {p.baslik}   (s.{p.sayfa})")
        print(f"   skor {p.skor:.4f} | birim: {', '.join(p.birimler)}")
        if full:
            for line in p.metin.split("\n"):
                print(f"      {line}")
        else:
            print(f"      {p.ozet}...")


def main() -> None:
    import sys

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    short = "--short" in sys.argv or "--summary" in sys.argv
    full = not short
    as_json = "--json" in sys.argv
    diagnostics = "--diagnostics" in sys.argv

    reranker = RERANKER
    if "--reranker" in sys.argv:
        reranker = sys.argv[sys.argv.index("--reranker") + 1]
        args = [a for a in args if a != reranker]

    print("yukleniyor...")
    s = Searcher(reranker=reranker)
    s.warmup()
    print(
        f"hazir: {len(s.chunks)} madde parcasi | "
        f"retriever={s.retriever} | reranker={s.reranker_name}\n"
    )

    def run(q: str) -> None:
        res = s.search(q, top_k=8)
        if as_json:
            payload = [p.to_dict(full) for p in res]
            if diagnostics:
                payload = {"karar": s.last_decision.to_dict(), "sonuclar": payload}
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            if not res and not s.last_decision.cevaplanabilir:
                print(f"  (cevap yok: {s.last_decision.neden})")
            else:
                _print(res, full)

    if args:
        run(" ".join(args))
        return

    print("Soru yaz, Enter'a bas. Cikmak icin bos birak veya Ctrl+C.")
    print("Birden fazla sorgu icin ` | ` ile ayir (Qwen'in urettigi gibi).\n")
    while True:
        try:
            q = input("soru> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        queries = [x.strip() for x in q.split("|") if x.strip()]
        res = s.search(queries, top_k=8)
        if as_json:
            payload = [p.to_dict(full) for p in res]
            if diagnostics:
                payload = {"karar": s.last_decision.to_dict(), "sonuclar": payload}
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            if not res and not s.last_decision.cevaplanabilir:
                print(f"  (cevap yok: {s.last_decision.neden})")
            else:
                _print(res, full)
        print()


if __name__ == "__main__":
    main()
