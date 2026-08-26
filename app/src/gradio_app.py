import json
import re
import traceback

import gradio as gr

# Sadece iskelet. Metin rengi HTML'de inline; global span/p EZME (beyaz-ustune-beyaz yapar).
WHITE_CSS = """
.gradio-container { background: #e8eef3 !important; max-width: none !important; width: 100% !important; }
footer { display: none !important; }
.masa-aside { background: #ffffff !important; border: 1px solid #c5d0d8 !important; border-radius: 12px !important; }
.masa-col { background: #ffffff !important; border: 1px solid #c5d0d8 !important; border-radius: 12px !important; }
button.primary, .gr-button-primary {
  background: #2b80b9 !important; color: #ffffff !important; border: 0 !important;
  font-weight: 700 !important; border-radius: 10px !important;
}
textarea, input[type="text"] {
  background: #ffffff !important; color: #1a1a1a !important;
  border: 1px solid #8a97a3 !important;
}
#tick-madde label, #tick-surec label,
#tick-madde .wrap label, #tick-surec .wrap label {
  display: flex !important; align-items: center !important; gap: 12px !important;
  width: 100% !important; min-height: 48px !important; margin: 0 0 8px !important;
  padding: 12px 14px !important; border-radius: 10px !important;
  background: #ffffff !important; color: #111111 !important;
  border: 1px solid #6b7780 !important; cursor: pointer !important;
  font-size: 14px !important; font-weight: 600 !important; line-height: 1.35 !important;
}
#tick-madde label span, #tick-surec label span,
#tick-madde label p, #tick-surec label p {
  color: #111111 !important; opacity: 1 !important;
}
#tick-madde label:has(input:checked), #tick-surec label:has(input:checked) {
  background: #2b80b9 !important; border-color: #2b80b9 !important;
}
#tick-madde label:has(input:checked) span, #tick-surec label:has(input:checked) span,
#tick-madde label:has(input:checked), #tick-surec label:has(input:checked) {
  color: #ffffff !important;
}
#tick-madde input[type="checkbox"], #tick-surec input[type="radio"] {
  width: 22px !important; height: 22px !important; min-width: 22px !important;
  accent-color: #2b80b9 !important; cursor: pointer !important;
  flex-shrink: 0 !important;
}
#tick-madde button, #tick-surec button {
  background: #ffffff !important; color: #111111 !important;
  min-height: 48px !important; border: 1px solid #6b7780 !important;
  border-radius: 10px !important; margin-bottom: 8px !important;
}
.ink, .ink * { color: #111111 !important; }
.ink { background: #ffffff !important; }
#ocr-panel, #ocr-panel .ink, #ocr-panel .html-container {
  max-height: none !important; overflow: visible !important;
}
.ocr-full { white-space: pre-wrap !important; color: #111111 !important;
  max-height: none !important; overflow: visible !important; word-break: break-word; }
.rag-score { white-space: nowrap !important; overflow: visible !important;
  min-width: 14em; display: inline-block; }
.rag-preview { white-space: pre-wrap; max-height: 11em; overflow: auto; }
.rag-full { margin-top: 8px; }
.rag-full > summary { color: #2b80b9 !important; font-weight: 700 !important;
  cursor: pointer; list-style: none; }
.rag-full > summary::-webkit-details-marker { display: none; }
.rag-sum-close { display: none; }
.rag-full[open] > summary .rag-sum-open { display: none; }
.rag-full[open] > summary .rag-sum-close { display: inline; }
.rag-card:has(details[open]) .rag-preview { display: none; }
.rag-full-body { white-space: pre-wrap; margin-top: 8px; }
"""

JS_LIGHT = """
() => {
  document.documentElement.classList.remove('dark');
  document.body.classList.remove('dark');
}
"""


def _h(title, inner):
    return (
        f'<div class="ink" style="background:#ffffff;border:1px solid #c5d0d8;border-radius:12px;'
        f'padding:12px 14px;margin:0 0 10px;color:#111111;font-family:Inter,Segoe UI,sans-serif">'
        f'<div style="font-size:11px;font-weight:700;color:#2b80b9;margin-bottom:8px">{title}</div>'
        f'<div class="ink" style="color:#111111;font-size:13px;line-height:1.45">{inner}</div></div>'
    )


def _esc(s) -> str:
    return str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def agent_board_html() -> str:
    st = EVRAK_STATE if isinstance(globals().get("EVRAK_STATE"), dict) else {}
    flags = st.get("flags") or {}
    a = st.get("analysis") if isinstance(st.get("analysis"), dict) else {}
    topic = a.get("primary_topic") or "—"
    n_ocr = len(st.get("ocr_text") or "")
    n_q = len(st.get("rag_queries") or [])
    n_p = len(st.get("provisions") or [])
    clerk = st.get("clerk") or {}
    draft = st.get("draft") or ""
    done = {x.get("ajan") for x in (st.get("trace") or []) if isinstance(x, dict)}

    def card(title, body):
        return (
            f'<div class="ink" style="flex:1;min-width:150px;background:#ffffff;border:1px solid #8a97a3;'
            f'border-radius:10px;padding:10px 12px;color:#111111">'
            f'<div style="color:#2b80b9;font-weight:800;font-size:12px">{_esc(title)}</div>'
            f'<div class="ink" style="color:#111111;font-size:13px;margin-top:6px;line-height:1.4">{body}</div></div>'
        )

    oku = f"bitti · OCR {n_ocr} kr" if "Okuyucu" in done else "bekliyor"
    eks = "eksik var" if flags.get("eksik_bilgi") else "eksik yok"
    ana = f"bitti · {_esc(topic)} · {eks}" if "Analiz" in done else "bekliyor"
    mev = f"bitti · {n_q} sorgu, {n_p} madde" if "Mevzuat" in done else "bekliyor"
    mem = _esc(clerk.get("note") or "HITL: madde işaretle + süreç")
    yaz = "bitti · taslak hazır" if draft else "bekliyor"
    row = (
        '<div style="display:flex;gap:8px;flex-wrap:nowrap">'
        + card("1. Okuyucu", oku)
        + card("2. Analiz", ana)
        + card("3. Mevzuat", mev)
        + card("Memur", mem)
        + card("4. Yazıcı", yaz)
        + "</div>"
    )
    return _h("Ajan masası", row)


EMPTY_BOARD = agent_board_html()
EMPTY_RAG = _h("Mevzuat", "Henüz yok. Belgeyi işle.")
EMPTY_LETTER = (
    '<div style="background:#fffdf8;border:1px solid #c4b89a;border-radius:4px;padding:20px;'
    'color:#1a1a1a;font-family:Times New Roman,Times,serif;font-size:15px;line-height:1.55;'
    'min-height:340px">Resmî yazı burada. JSON buraya yazılmaz.</div>'
)


def _prov_to_dict(p) -> dict:
    if isinstance(p, dict):
        metin = p.get("metin") or p.get("ozet") or ""
        skor = p.get("skor") if p.get("skor") is not None else p.get("reranker_skoru")
        return {"kanun": p.get("kanun") or "", "madde": p.get("madde") or "",
                "baslik": p.get("baslik") or "", "metin": metin, "skor": skor}
    return {
        "kanun": getattr(p, "kanun", "") or "",
        "madde": getattr(p, "madde", "") or "",
        "baslik": getattr(p, "baslik", "") or "",
        "metin": getattr(p, "metin", None) or getattr(p, "ozet", "") or "",
        "skor": getattr(p, "skor", None) or getattr(p, "reranker_skoru", None),
    }


def _prov_label(d: dict, i: int) -> str:
    return f"{i}. {d.get('kanun','')} — {d.get('madde','')} | {str(d.get('baslik') or '')[:70]}"


def provisions_cards_html(plist: list) -> str:
    if not plist:
        return _h("Mevzuat", "Bulunamadı veya Amaç/Kapsam/Tanım elendi.")
    lim = 280
    cards = []
    for i, d in enumerate(plist, 1):
        metin = str(d.get("metin") or "").strip() or "—"
        govde = (
            f'<div class="ink rag-preview" style="color:#111111;font-size:13px;'
            f'line-height:1.5">{_esc(metin)}</div>'
        )
        if len(metin) > lim:
            govde += (
                f'<details class="rag-full">'
                f'<summary><span class="rag-sum-open">Tam metni göster</span>'
                f'<span class="rag-sum-close">Kapat</span></summary>'
                f'<div class="ink rag-full-body" style="color:#111111;font-size:13px;'
                f'line-height:1.5">{_esc(metin)}</div></details>'
            )
        cards.append(
            f'<div class="rag-card ink" style="background:#f4f7f9;border:1px solid #8a97a3;'
            f'border-radius:10px;padding:12px 14px;margin:0 0 10px;color:#111111">'
            f'<div class="rag-score" style="color:#2b80b9;font-weight:800">'
            f'{i}. sıra</div>'
            f'<div class="ink" style="font-weight:700;color:#111111">'
            f'{_esc(d.get("kanun"))} — {_esc(d.get("madde"))}</div>'
            f'<div class="ink" style="color:#111111;margin-bottom:8px">{_esc(d.get("baslik") or "-")}</div>'
            f'{govde}</div>'
        )
    return _h("Mevzuat maddeleri", "".join(cards))


def letter_html(text: str) -> str:
    body = _esc((text or "").strip()) or "Resmî yazı boş."
    return (
        f'<div style="background:#fffdf8;border:1px solid #c4b89a;border-radius:4px;padding:22px 24px;'
        f'color:#111111;font-family:Times New Roman,Times,serif;font-size:15px;line-height:1.55;'
        f'white-space:pre-wrap;min-height:360px">{body}</div>'
    )


def _fmt_info_html(val) -> str:
    """key_information / missing_information: liste veya sözlük."""
    items = []
    if val is None or val == "":
        return "—"
    if isinstance(val, str):
        s = val.strip()
        return _esc(s) if s and s.lower() not in {"yok", "-", "yok."} else "—"
    if isinstance(val, dict):
        for k, v in val.items():
            vs = "" if v is None else str(v).strip()
            if vs and vs.lower() not in {"yok", "-", "yok."}:
                items.append((str(k), vs))
    elif isinstance(val, list):
        for it in val:
            if isinstance(it, dict):
                t = str(it.get("type") or it.get("key") or "").strip()
                v = str(it.get("value") or it.get("val") or "").strip()
                if t or (v and v.lower() not in {"yok", "-", "yok."}):
                    items.append((t or "bilgi", v or "—"))
            else:
                s = str(it).strip()
                if s and s.lower() not in {"yok", "-", "yok."}:
                    items.append(("", s))
    else:
        return _esc(str(val))
    if not items:
        return "—"
    bits = []
    for k, v in items:
        if k:
            bits.append(
                f'<div class="ink" style="margin:2px 0 0 8px;color:#111111">'
                f'• <b>{_esc(k)}</b>: {_esc(v)}</div>'
            )
        else:
            bits.append(
                f'<div class="ink" style="margin:2px 0 0 8px;color:#111111">• {_esc(v)}</div>'
            )
    return "".join(bits)


def analysis_html(analysis: dict) -> str:
    if not isinstance(analysis, dict) or analysis.get("_parse_error"):
        return _h("Analiz", "Yok.")
    simple = [
        ("Belge türü", analysis.get("document_type")),
        ("Gönderen", analysis.get("sender_type")),
        ("Konu", analysis.get("primary_topic")),
        ("Talep (requested_action)", analysis.get("requested_action")),
        ("Özet", analysis.get("summary")),
    ]
    bits = "".join(
        f'<div class="ink" style="margin:0 0 6px;color:#111111"><b>{_esc(k)}:</b> {_esc(v or "—")}</div>'
        for k, v in simple
    )
    bits += (
        f'<div class="ink" style="margin:8px 0 4px;color:#111111"><b>Ana bilgiler (key_information):</b></div>'
        + _fmt_info_html(analysis.get("key_information"))
        + f'<div class="ink" style="margin:8px 0 4px;color:#111111"><b>Eksik bilgi (missing_information):</b></div>'
        + _fmt_info_html(analysis.get("missing_information"))
    )
    return _h("Analiz", bits)


def response_type_html(rt: str) -> str:
    t = (rt or "").strip() or "—"
    return _h("Yazı türü (response_type)", f'<div class="ink" style="font-weight:700;color:#111111">{_esc(t)}</div>')


EMPTY_RESP = response_type_html("")


def ocr_html(text: str) -> str:
    return (
        '<div id="ocr-panel" class="ink" style="background:#ffffff;border:1px solid #c5d0d8;'
        'border-radius:12px;padding:12px 14px;margin:0 0 10px;color:#111111">'
        '<div style="font-size:11px;font-weight:700;color:#2b80b9;margin-bottom:8px">OCR</div>'
        f'<div class="ocr-full">{_esc(text or "—")}</div></div>'
    )


def status_html(text: str) -> str:
    return _h("Durum", _esc(text or "—"))


def lookup_unit(analysis: dict) -> str:
    topic = str((analysis or {}).get("primary_topic") or "").strip()
    return TOPIC_TO_UNIT.get(topic, "Yazı İşleri Müdürlüğü")


def _cite_label(d: dict) -> str:
    """Kanun adı + Madde no + madde başlığı (hardcode atıfın havada kalmaması için)."""
    kanun = (d.get("kanun") or "").strip()
    madde = (d.get("madde") or "").strip()
    baslik = (d.get("baslik") or "").strip()
    if baslik in {"-", "—", "–", "yok", "Yok"}:
        baslik = ""
    if madde and not re.search(r"(?i)\bmadde\b", madde) and re.search(r"\d+", madde):
        madde = f"Madde {madde}"
    bits = [x for x in (kanun, madde) if x]
    if baslik:
        blob = " ".join(bits).lower()
        if baslik.lower() not in blob and len(baslik) <= 90:
            bits.append(f"({baslik})")
    return " ".join(bits).strip()


def _strip_dative_unit(s: str) -> str:
    t = (s or "").strip()
    for a, b in (
        ("Müdürlüğüne", "Müdürlüğü"),
        ("MÜDÜRLÜĞÜNE", "MÜDÜRLÜĞÜ"),
        ("Başkanlığına", "Başkanlığı"),
        ("BAŞKANLIĞINA", "BAŞKANLIĞI"),
        ("müdürlüğüne", "müdürlüğü"),
        ("başkanlığına", "başkanlığı"),
    ):
        if t.endswith(a):
            return t[: -len(a)] + b
    return t


def _citizen_tokens(analysis: dict) -> list:
    """Başvuran kişi/adres kelimeleri. key_information hem liste hem sözlük olabilir."""
    toks = []
    ki = (analysis or {}).get("key_information") if isinstance(analysis, dict) else None
    pairs = []
    if isinstance(ki, list):
        for it in ki:
            if isinstance(it, dict):
                pairs.append((str(it.get("type") or ""), str(it.get("value") or "")))
    elif isinstance(ki, dict):
        for k, v in ki.items():
            pairs.append((str(k), str(v or "")))
    want = ("person", "ad", "soyad", "isim", "basvuran", "kisi",
            "location", "address", "adres", "mahalle", "konum")
    for t, v in pairs:
        tl = t.lower()
        vv = v.strip()
        if not vv or vv.lower() in {"yok", "-", "yok."}:
            continue
        if any(x in tl for x in want):
            toks.append(vv)
            for w in re.split(r"[,\s/]+", vv):
                if len(w) > 2:
                    toks.append(w)
    return toks


_EK_HEAD = re.compile(
    r"(?i)^\s*ek(?:ler| listesi)?\s*:?\s*(.*)$"
)


def _is_ek_item_line(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return False
    if t.startswith(("-", "•", "*", "–")) and re.search(
        r"(?i)(dilekçe|sayfa|evrak|fotokopi|başvuru)", t
    ):
        return True
    m = _EK_HEAD.match(t)
    if not m:
        return False
    rest = (m.group(1) or "").strip()
    return (not rest) or bool(re.search(r"(?i)(dilekçe|sayfa|evrak|fotokopi|başvuru)", rest))


def _strip_trailing_ek(text: str) -> str:
    """Modelin yazı sonuna yapıştırdığı sahte Ek / dilekçe listesini kes."""
    lines = (text or "").rstrip().splitlines()
    while lines and (not lines[-1].strip() or _is_ek_item_line(lines[-1])):
        lines.pop()
    close_i = None
    for i, ln in enumerate(lines):
        low = ln.strip().lower()
        if low.startswith(("saygılarımla", "saygılar", "saygilar", "bilgilerinize", "gereğini", "geregini")):
            close_i = i
        elif "rica ederim" in low or "arz ederim" in low:
            close_i = i
    if close_i is not None:
        head, tail = lines[: close_i + 1], lines[close_i + 1 :]
        out_tail, skipping = [], False
        for ln in tail:
            if _is_ek_item_line(ln) or (skipping and not ln.strip()):
                skipping = True
                continue
            if skipping and ln.strip().startswith(("-", "•", "*", "–")):
                continue
            skipping = False
            out_tail.append(ln)
        lines = head + out_tail
    return "\n".join(lines).rstrip()


def polish_official_letter(letter: str, payload: dict, analysis: dict, sender_unit: str) -> str:
    """Belediye → vatandaş: yanlış antet/hitap ve vatandaş imzasını kes.

    ensure_legislation_in_draft'tan ÖNCE çağrılmalı; yoksa atıf
    kapanış/imza satırlarına yapışır.
    """
    text = (letter or "").strip()
    sender = _strip_dative_unit(sender_unit or (payload or {}).get("target_unit") or "")
    dest = _strip_dative_unit((payload or {}).get("target_unit") or "")
    lines = text.splitlines()
    i = 0
    while i < min(len(lines), 8):
        ln = lines[i].strip()
        up = ln.upper()
        if (not ln) or up in {"T.C.", "TC"} or up.endswith("BAŞKANLIĞINA") or up.endswith("MÜDÜRLÜĞÜNE"):
            i += 1
            continue
        break
    body = "\n".join(lines[i:]).lstrip()
    head = "\n".join(lines[:i])
    if re.search(r"(BAŞKANLIĞINA|MÜDÜRLÜĞÜNE)", head, re.I) or not re.match(r"(?i)^\s*T\.C\.", text):
        text = f"T.C.\n{sender}\n\n{body}".strip() if sender else f"T.C.\n\n{body}".strip()

    # Antette birimin parantezli tekrarını at: "(Veteriner İşleri Müdürlüğü)"
    slow = sender.lower()
    cleaned, prev = [], None
    for ln in text.splitlines():
        s = ln.strip()
        bare = s.strip("()").strip().lower()
        if s.startswith("(") and s.endswith(")") and bare == slow:
            continue
        if prev is not None and s and s == prev:
            continue
        cleaned.append(ln)
        if s:
            prev = s
    text = "\n".join(cleaned)

    # Vatandaş imza bloğunu sondan kes (isim + adres + telefon).
    toks = [t.lower() for t in _citizen_tokens(analysis if isinstance(analysis, dict) else {}) if len(t) > 2]
    phone = re.compile(r"(?<!\d)(0\s*)?5\d[\d\s]{8,}")
    addr = re.compile(r"(?i)(mahallesi|caddesi|sokak|sok\.|no:|telefon|/[A-ZÇĞİÖŞÜ])")
    out_lines = text.rstrip().splitlines()
    while out_lines:
        ln = out_lines[-1].strip()
        if not ln:
            out_lines.pop()
            continue
        low = ln.lower()
        # Kapanış selamı geldiyse imza bloğu bitmiştir, dur.
        if low.startswith(("saygılar", "saygilar", "bilgilerinize", "gereğini", "geregini")):
            break
        if toks and any(tok in low for tok in toks):
            out_lines.pop()
            continue
        if phone.search(ln) or addr.search(ln):
            out_lines.pop()
            continue
        break
    text = "\n".join(out_lines).rstrip()
    text = _strip_trailing_ek(text)

    if (payload or {}).get("process_status") == "YONLENDIRILDI" and dest:
        if dest.lower() not in text.lower():
            text = text.rstrip() + f"\n\nBaşvurunuz {dest} birimine yönlendirilmiştir.\n"
    return text


def _is_closing_line(ln: str) -> bool:
    low = (ln or "").strip().lower()
    if not low:
        return False
    if low.startswith(("saygılarımla", "saygılar", "saygilar", "bilgilerinize", "gereğini", "geregini")):
        return True
    return "rica ederim" in low or "arz ederim" in low


def _insert_cite_into_body(draft: str, sentence: str) -> str:
    """Atıfı yazı sonuna ekleme; kapanıştan önceki son gövde cümlesine kat."""
    text = (draft or "").rstrip()
    sent = (sentence or "").strip()
    if sent and sent[-1] not in ".!?":
        sent += "."
    if not sent:
        return (draft or "")
    if not text:
        return sent + "\n"
    lines = text.splitlines()
    close_i = next((i for i, ln in enumerate(lines) if _is_closing_line(ln)), None)
    body_end = close_i if close_i is not None else len(lines)
    j = body_end - 1
    while j >= 0 and not lines[j].strip():
        j -= 1
    if j < 0:
        return text + "\n\n" + sent + "\n"
    prev = lines[j].rstrip()
    plow = prev.lower()
    if plow.startswith(("sayın", "ilgi", "konu:", "t.c")) or plow in {"t.c.", "tc"}:
        lines.insert(j + 1, "")
        lines.insert(j + 2, sent)
    else:
        if prev[-1:] not in ".!?":
            prev += "."
        lines[j] = prev + " " + sent
    return "\n".join(lines).rstrip() + "\n"


def ensure_legislation_in_draft(draft: str, chosen: list) -> str:
    if not chosen:
        return draft or ""
    t = (draft or "").lower()
    hit = False
    for d in chosen:
        kanun = (d.get("kanun") or "").lower()
        nums = re.findall(r"\d+", str(d.get("madde") or ""))
        if nums and nums[0] in t and (
            (kanun and kanun[:16] in t)
            or ("yönetmelik" in kanun and "yönetmelik" in t)
            or ("kanun" in kanun and "kanun" in t)
        ):
            hit = True
            break
    if hit:
        return draft
    cites = ", ".join(_cite_label(d) for d in chosen[:2] if _cite_label(d))
    if not cites:
        return draft or ""
    return _insert_cite_into_body(
        draft, f"İşlemler {cites} uyarınca yürütülmüştür."
    )


def intake_handler(image):
    empty_cb = gr.update(choices=[], value=[])
    empty = (
        EMPTY_BOARD, ocr_html("Önce görsel yükle."), analysis_html({}), "{}",
        "", "Yazı İşleri Müdürlüğü", EMPTY_RAG, empty_cb,
        "INCELEMEDE", gr.update(value="", visible=False), "", "", None, None, [],
        EMPTY_RESP,
    )
    if image is None:
        return empty
    def st(msg):
        print(msg, flush=True)
    try:
        ocr_text, analysis, queries, provisions = run_intake(image, status_cb=st)
    except Exception:
        err = traceback.format_exc()
        print(err, flush=True)
        return (
            EMPTY_BOARD, ocr_html(err), analysis_html({}), "{}",
            "", "Yazı İşleri Müdürlüğü", _h("Hata", _esc(err)), empty_cb,
            "INCELEMEDE", gr.update(value="", visible=False), "", "", None, None, [],
            EMPTY_RESP,
        )
    plist = [_prov_to_dict(p) for p in (provisions or [])]
    labels = [_prov_label(d, i) for i, d in enumerate(plist, 1)]
    unit = lookup_unit(analysis)
    default_sel = labels[:2] if labels else []
    default_proc = "EKSIK_BILGI_BEKLENIYOR" if _has_missing(analysis if isinstance(analysis, dict) else {}) else "INCELEMEDE"
    return (
        agent_board_html(),
        ocr_html(ocr_text),
        analysis_html(analysis if isinstance(analysis, dict) else {}),
        json.dumps(analysis, ensure_ascii=False, indent=2),
        " | ".join(queries or []) or "(sorgu yok)",
        unit,
        provisions_cards_html(plist),
        gr.update(choices=labels, value=default_sel),
        default_proc, gr.update(value="", visible=False), "", "",
        ocr_text, analysis, plist,
        EMPTY_RESP,
    )


def _toggle_yon(st):
    return gr.update(visible=(st == "YONLENDIRILDI"))


def draft_handler(selected, process_status, performed_raw, result_info, unit, yon_unit, ocr_s, analysis_s, plist):
    if not analysis_s:
        return "{}", EMPTY_LETTER, "{}", EMPTY_BOARD, EMPTY_RESP
    actions = [x.strip() for x in (performed_raw or "").split("\n") if x.strip()]
    selected = selected or []
    chosen = []
    if plist:
        for i, d in enumerate(plist, 1):
            if _prov_label(d, i) in selected:
                chosen.append(d)
    status = process_status or "INCELEMEDE"
    dest = (yon_unit or "").strip()
    if status == "YONLENDIRILDI" and dest:
        target = dest
    else:
        target = unit or lookup_unit(analysis_s)
    payload_in = {
        "document_type": analysis_s.get("document_type", ""),
        "sender_type": analysis_s.get("sender_type", ""),
        "primary_topic": analysis_s.get("primary_topic", ""),
        "requested_action": analysis_s.get("requested_action", ""),
        "key_information": analysis_s.get("key_information", {}),
        "missing_information": analysis_s.get("missing_information", []),
        "summary": analysis_s.get("summary", ""),
        "process_status": status,
        "performed_actions": actions,
        "result_information": (result_info or "").strip(),
        "target_unit": target,
        "selected_legislation": [
            {k: x.get(k) for k in ("kanun", "madde", "baslik", "metin")}
            for x in chosen
        ],
    }
    try:
        raw = resmi_yazi_uret(json.dumps(payload_in, ensure_ascii=False))
    except Exception:
        err = traceback.format_exc()
        print(err, flush=True)
        return json.dumps(payload_in, ensure_ascii=False, indent=2), letter_html("HATA:\n" + err), json.dumps(payload_in, ensure_ascii=False, indent=2), EMPTY_BOARD, EMPTY_RESP
    parsed = try_parse_json(raw) or {}
    letter = extract_official_letter(raw)
    if payload_in["process_status"] == "EKSIK_BILGI_BEKLENIYOR" and isinstance(parsed, dict):
        parsed["response_type"] = "EKSIK_BILGI_BELGE_TAMAMLAMA_YAZISI"
    letter = polish_official_letter(letter, payload_in, analysis_s, sender_unit=unit or lookup_unit(analysis_s))
    letter = ensure_legislation_in_draft(letter, chosen)
    out_show = {
        "response_type": parsed.get("response_type") if isinstance(parsed, dict) else "",
        "target_unit": payload_in["target_unit"],
        "process_information": parsed.get("process_information") if isinstance(parsed, dict) else "",
        "draft": letter,
    }
    return (
        json.dumps(out_show, ensure_ascii=False, indent=2),
        letter_html(letter),
        json.dumps(payload_in, ensure_ascii=False, indent=2),
        agent_board_html(),
        response_type_html(out_show.get("response_type") or ""),
    )


def rag_query_handler(query):
    q = (query or "").strip()
    if not q:
        return _h("Mevzuat", "Sorgu yaz.")
    try:
        provisions = searcher.search([q], top_k=RAG_TOP_K)
        provisions = filter_boilerplate_provisions(provisions)[:RAG_TOP_K]
    except Exception:
        return _h("Hata", _esc(traceback.format_exc()))
    return provisions_cards_html([_prov_to_dict(p) for p in (provisions or [])])


# Radio değeri enum (mantık için); ekranda düzgün Türkçe etiket.
PROCESS_STATUS_LABELS = {
    "INCELEMEDE": "İnceleniyor",
    "TAMAMLANDI": "Tamamlandı",
    "EKSIK_BILGI_BEKLENIYOR": "Eksik bilgi bekleniyor",
    "REDDEDILDI": "Reddedildi",
    "YONLENDIRILDI": "Yönlendirildi",
}
PROC_STATUS_CHOICES = [
    (PROCESS_STATUS_LABELS.get(v, v), v) for v in PROCESS_STATUS_CHOICES
]


_theme = gr.themes.Base(primary_hue="blue", secondary_hue="teal", neutral_hue="slate")
try:
    _theme = _theme.set(
        body_background_fill="#e8eef3",
        body_background_fill_dark="#e8eef3",
        body_text_color="#1a1a1a",
        body_text_color_dark="#1a1a1a",
        block_background_fill="#ffffff",
        block_background_fill_dark="#ffffff",
        block_label_text_color="#1a1a1a",
        block_label_text_color_dark="#1a1a1a",
        block_title_text_color="#1a1a1a",
        block_title_text_color_dark="#1a1a1a",
        input_background_fill="#ffffff",
        input_background_fill_dark="#ffffff",
        input_text_color="#1a1a1a",
        input_text_color_dark="#1a1a1a",
        button_primary_text_color="#ffffff",
        button_primary_background_fill="#2b80b9",
    )
except Exception:
    pass

_kw = dict(title="Belediye Evrak Masası", theme=_theme, css=WHITE_CSS)
try:
    demo_ctx = gr.Blocks(js=JS_LIGHT, head='<meta name="color-scheme" content="light">', fill_width=True, **_kw)
except TypeError:
    try:
        demo_ctx = gr.Blocks(js=JS_LIGHT, head='<meta name="color-scheme" content="light">', **_kw)
    except TypeError:
        demo_ctx = gr.Blocks(**_kw)

with demo_ctx as demo:
    agent_md = gr.HTML(EMPTY_BOARD)
    st_ocr = gr.State("")
    st_analysis = gr.State(None)
    st_plist = gr.State([])
    with gr.Row():
        with gr.Column(elem_classes=["masa-col"], scale=4, min_width=320):
            gr.HTML('<div style="color:#1a1a1a;font-weight:800;font-size:16px">Belge</div>')
            img_in = gr.Image(type="pil", label="Görsel", height=220)
            btn_in = gr.Button("Belgeyi işle (OCR + analiz + RAG)", variant="primary")
            ocr_out = gr.HTML(ocr_html(""), elem_id="ocr-panel")
            analysis_md = gr.HTML(analysis_html({}))
            with gr.Accordion("Ham JSON (analiz)", open=False):
                analysis_out = gr.Code(language="json", lines=6)
        with gr.Column(elem_classes=["masa-col"], scale=5, min_width=360):
            gr.HTML('<div style="color:#1a1a1a;font-weight:800;font-size:16px">Memur + mevzuat</div>')
            unit_out = gr.Textbox(label="Hedef birim", lines=1)
            rag_q_out = gr.Textbox(label="RAG sorguları", lines=2)
            rag_md = gr.HTML(EMPTY_RAG)
            rag_cb = gr.CheckboxGroup(choices=[], label="Yazıya gidecek maddeler", elem_id="tick-madde")
            proc = gr.Radio(PROC_STATUS_CHOICES, value="INCELEMEDE", label="Süreç", elem_id="tick-surec")
            yon_unit = gr.Textbox(
                label="Yönlendirilecek birim",
                lines=1,
                visible=False,
                placeholder="Örn. Park ve Bahçeler Müdürlüğü",
            )
            actions_in = gr.Textbox(label="Aksiyonlar (satır satır)", lines=3)
            result_in = gr.Textbox(label="Süreç sonucu", lines=2)
        with gr.Column(elem_classes=["masa-col"], scale=5, min_width=360):
            gr.HTML('<div style="color:#1a1a1a;font-weight:800;font-size:16px">Taslak</div>')
            btn_draft = gr.Button("Taslak üret", variant="primary")
            resp_type_md = gr.HTML(EMPTY_RESP)
            draft_out = gr.HTML(EMPTY_LETTER)
            with gr.Accordion("Ham JSON (yazı değil)", open=False):
                writer_in_json = gr.Code(language="json", lines=6)
                writer_json = gr.Code(language="json", lines=6)
    with gr.Accordion("Serbest mevzuat", open=False):
        rag_q = gr.Textbox(label="Sorgu", lines=2)
        rag_btn = gr.Button("Ara")
        rag_direct_out = gr.HTML()
        rag_btn.click(rag_query_handler, inputs=rag_q, outputs=rag_direct_out)
    proc.change(_toggle_yon, inputs=proc, outputs=yon_unit)
    btn_in.click(
        intake_handler, inputs=img_in,
        outputs=[
            agent_md, ocr_out, analysis_md, analysis_out, rag_q_out,
            unit_out, rag_md, rag_cb, proc, yon_unit, actions_in, result_in,
            st_ocr, st_analysis, st_plist, resp_type_md,
        ],
    )
    btn_draft.click(
        draft_handler,
        inputs=[rag_cb, proc, actions_in, result_in, unit_out, yon_unit, st_ocr, st_analysis, st_plist],
        outputs=[writer_json, draft_out, writer_in_json, agent_md, resp_type_md],
    )

demo.queue(default_concurrency_limit=1)
demo.launch(share=True, debug=False)
