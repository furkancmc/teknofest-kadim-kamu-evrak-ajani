"""Cross-encoder reranker sarmalayıcıları.

Her reranker'ın kendi çağrı konvansiyonu var. bge-reranker klasik bir
sequence-classification modeli: (sorgu, belge) çiftini alır, tek logit verir.
Qwen3-Reranker ise causal LM: bir talimat şablonuna sarılır ve "yes"/"no"
token logitleri karşılaştırılır. Yanlış çağrılan model kötü model gibi
görünür, bu yüzden konvansiyonlar burada tek yerde toplanır.

`score()` her zaman "yüksek = daha ilgili" ham bir skor listesi döndürür;
skorların ölçekleri modeller arasında karşılaştırılabilir DEĞİLDİR, yalnız
aynı model içindeki sıralama anlamlıdır.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


# Qwen3-Reranker talimat alanı. Mevzuat aramasına uyarlanmış hâli, modelin
# varsayılan "web search" talimatından belirgin biçimde farklı davranıyor.
TR_LEGAL_INSTRUCTION = (
    "Verilen idari/hukuki soruyu doğrudan cevaplayan mevzuat maddesini bul"
)


@dataclass
class RerankerSpec:
    name: str
    hf_id: str
    kind: str  # "cross-encoder" | "qwen3"
    max_length: int = 512
    instruction: str | None = None
    extra: dict = field(default_factory=dict)


SPECS: dict[str, RerankerSpec] = {
    "bge-m3": RerankerSpec("bge-m3", "BAAI/bge-reranker-v2-m3", "cross-encoder", 512),
    "qwen3-0.6b": RerankerSpec(
        "qwen3-0.6b", "Qwen/Qwen3-Reranker-0.6B", "qwen3", 1024,
        instruction="Given a web search query, retrieve relevant passages that answer the query",
    ),
    "qwen3-0.6b-tr": RerankerSpec(
        "qwen3-0.6b-tr", "Qwen/Qwen3-Reranker-0.6B", "qwen3", 1024,
        instruction=TR_LEGAL_INSTRUCTION,
    ),
    "jina-v2": RerankerSpec(
        "jina-v2", "jinaai/jina-reranker-v2-base-multilingual", "cross-encoder", 512,
        extra={"trust_remote_code": True},
    ),
}


class Reranker:
    def __init__(self, spec: RerankerSpec) -> None:
        self.spec = spec
        self._model = None

    # --- yükleme ----------------------------------------------------------

    def load(self):
        if self._model is not None:
            return self._model
        if self.spec.kind == "cross-encoder":
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.spec.hf_id,
                max_length=self.spec.max_length,
                **self.spec.extra,
            )
        elif self.spec.kind == "qwen3":
            self._model = _Qwen3Reranker(self.spec)
        else:
            raise ValueError(f"bilinmeyen reranker türü: {self.spec.kind}")
        return self._model

    def unload(self) -> None:
        self._model = None
        torch.cuda.empty_cache()

    # --- skorlama ---------------------------------------------------------

    def score(self, pairs: list[tuple[str, str]], batch_size: int = 16,
              progress: bool = False) -> list[float]:
        model = self.load()
        if self.spec.kind == "qwen3":
            return model.score(pairs, batch_size=batch_size, progress=progress)
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=progress)
        return [float(s) for s in scores]


class _Qwen3Reranker:
    """Qwen3-Reranker: causal LM'in "yes" olasılığı ilgililik skoru olur.

    sentence-transformers'ın CrossEncoder'ı bu modeli destekliyor ama şablonu
    ve `yes/no` logit çıkarımını burada açıkça kurmak iki şey sağlıyor:
    talimat alanını değiştirebilmek ve hangi tokenin okunduğunu görebilmek.
    """

    PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements "
        'based on the Query and the Instruct provided. Note that the answer can '
        'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, spec: RerankerSpec) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.tok = AutoTokenizer.from_pretrained(spec.hf_id, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id, torch_dtype=torch.float16
        ).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.false_id = self.tok.convert_tokens_to_ids("no")
        self.true_id = self.tok.convert_tokens_to_ids("yes")
        self.prefix_ids = self.tok.encode(self.PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tok.encode(self.SUFFIX, add_special_tokens=False)
        self.body_budget = spec.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        self.instruction = spec.instruction or "Given a web search query, retrieve relevant passages that answer the query"

    def _format(self, query: str, doc: str) -> str:
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"

    @torch.inference_mode()
    def score(self, pairs: list[tuple[str, str]], batch_size: int = 8,
              progress: bool = False) -> list[float]:
        device = next(self.model.parameters()).device
        out: list[float] = []
        rng = range(0, len(pairs), batch_size)
        if progress:
            from tqdm import tqdm

            rng = tqdm(rng, desc=self.spec.name)
        for start in rng:
            batch = pairs[start : start + batch_size]
            texts = [self._format(q, d) for q, d in batch]
            enc = self.tok(
                texts,
                add_special_tokens=False,
                truncation=True,
                max_length=self.body_budget,
                return_attention_mask=False,
            )
            enc["input_ids"] = [
                self.prefix_ids + ids + self.suffix_ids for ids in enc["input_ids"]
            ]
            enc = self.tok.pad(enc, padding=True, return_tensors="pt").to(device)
            logits = self.model(**enc).logits[:, -1, :]
            pair_logits = torch.stack(
                [logits[:, self.false_id], logits[:, self.true_id]], dim=1
            ).float()
            probs = torch.nn.functional.log_softmax(pair_logits, dim=1)[:, 1].exp()
            out.extend(probs.tolist())
        return out


def get(name: str) -> Reranker:
    return Reranker(SPECS[name])
