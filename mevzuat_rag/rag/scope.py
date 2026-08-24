"""Mevzuat korpusunun açıkça cevaplayamayacağı sorguları ayırır.

Bu modül bir hukuk/konu sınıflandırıcısı değildir. Yalnız yerel ve canlı veri
gerektirdiği açık olan telefon, web adresi, yakın tarihli kurul kararı ve
güncel/future tarife tutarı gibi yüksek kesinlikli örneklerde geri çekilir.
Belirsiz durumda retrieval çalışmaya devam eder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from .index.bm25 import tr_lower


@dataclass(frozen=True)
class ScopeDecision:
    in_scope: bool
    reason: str


_LOCAL = re.compile(
    r"\b(?:bu belediye(?:nin)?|belediyemiz(?:de|in)?|ilçe(?:deki|mizde)?|"
    r"mahalle(?:deki|mizde)?|kurumumuz(?:un)?)\b"
)
_CONTACT = re.compile(
    r"\b(?:telefon(?: numarası)?\w*|e[ -]?posta\w*|internet adres\w*|"
    r"web site\w*|url|hangi adres\w*)\b"
)
_LIVE_SERVICE = re.compile(r"\b(?:nöbetçi|bugün açık|yarın açık)\b")
_RECENT = re.compile(
    r"\b(?:geçen hafta\w*|bu hafta\w*|dün|bugün|az önce|son toplantı)\b"
)
_EVENT = re.compile(r"\b(?:toplantı|karar|gündem|ihale sonucu|etkinlik)\w*\b")
_AMOUNT = re.compile(
    r"\b(?:tarife|ücret|bedel|metrekare değer|rayiç değer)\w*\b.*"
    r"\b(?:ne kadar|kaç (?:tl|lira)|tutar)\w*\b|"
    r"\b(?:ne kadar|kaç (?:tl|lira)|tutar)\w*\b.*"
    r"\b(?:tarife|ücret|bedel|metrekare değer|rayiç değer)\w*\b"
)
_YEAR = re.compile(r"\b(20\d{2})\b")


def classify_scope(queries: str | list[str], current_year: int | None = None) -> ScopeDecision:
    if isinstance(queries, str):
        queries = [queries]
    text = " ".join(tr_lower(q) for q in queries if q and q.strip())
    if not text:
        return ScopeDecision(False, "empty_query")

    local = bool(_LOCAL.search(text))

    if _CONTACT.search(text) and (local or _LIVE_SERVICE.search(text)):
        return ScopeDecision(False, "local_contact_or_navigation_data")

    if _LIVE_SERVICE.search(text) and (local or _CONTACT.search(text)):
        return ScopeDecision(False, "live_service_data")

    if _RECENT.search(text) and _EVENT.search(text):
        return ScopeDecision(False, "recent_local_event_data")

    if _AMOUNT.search(text):
        years = [int(y) for y in _YEAR.findall(text)]
        year = current_year if current_year is not None else date.today().year
        if local or "güncel" in text or any(y >= year for y in years):
            return ScopeDecision(False, "current_or_local_tariff_data")

    return ScopeDecision(True, "legislation_retrieval")
