"""Resmi birim–konu ve RAG anahtarları: maps/belediye_konu.json"""
from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def load_maps(path=None) -> dict:
    cands = []
    if path:
        cands.append(Path(path))
    cands.extend(
        [
            _HERE.parent / "maps" / "belediye_konu.json",  # depo yerleşimi: app/maps/
            _HERE / "maps" / "belediye_konu.json",
            _HERE / "belediye_konu.json",
        ]
    )
    for p in cands:
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "belediye_konu.json bulunamadı. maps/belediye_konu.json veya path ver."
    )


def _strip_rag_numbers(s: str) -> str:
    """topic_to_rag icindeki 5199/4982 gibi kanun no sizintisini kes."""
    return " ".join(w for w in str(s).split() if not w.isdigit())


_DATA = load_maps()
TOPIC_TO_UNIT = dict(_DATA["topic_to_unit"])
TOPIC_TO_RAG = {k: _strip_rag_numbers(v) for k, v in dict(_DATA["topic_to_rag"]).items()}
UNITS = dict(_DATA["units"])
SHARED_TOPICS = dict(_DATA.get("shared_topics") or {})

PROCESS_STATUS_CHOICES = [
    "INCELEMEDE",
    "TAMAMLANDI",
    "EKSIK_BILGI_BEKLENIYOR",
    "REDDEDILDI",
    "YONLENDIRILDI",
]
