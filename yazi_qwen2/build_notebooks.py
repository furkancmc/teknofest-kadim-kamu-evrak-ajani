"""Colab notebook'larini (fine-tune + evaluate) programatik olarak insa eder.

UYARI: `01_finetune_qwen25_7b.ipynb` ve `02_evaluate_qwen25_7b.ipynb` artik
dogrudan duzenleniyor. Bu betigi calistirmak o dosyalarin uzerine yazar;
juri/GitHub metinleri kaybolur. Yalnizca sifirdan uretmek icin kullanin.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


# =====================================================================
# NOTEBOOK 1: FINE-TUNING
# =====================================================================

ft_cells = []

ft_cells.append(md("""# Qwen2.5-7B-Instruct Fine-Tuning — Belediye Evrak Karar Destek Modeli (Qwen 2)

Bu notebook, `qwen2_train_updated.jsonl` ve `qwen2_val_updated.jsonl` veri setleriyle
`Qwen2.5-7B-Instruct` modelini **Unsloth + QLoRA** kullanarak Google Colab'da fine-tune eder.
Ayarlar **Colab Pro (A100/L4, 40GB VRAM)** icin optimize edilmistir (4096 token context,
efektif batch=32); T4'e (ucretsiz katman) donerseniz `MAX_SEQ_LENGTH=2048` ve
`per_device_train_batch_size=2` kullanin.

**Once yapmaniz gerekenler:**
1. Colab'da: `Calisma Zamani > Calisma zamani turunu degistir > A100 GPU` secin (Colab Pro'da
   `Ek RAM` / `Premium GPU` secenegini de acmaniz onerilir).
2. Asagidaki 3 dosyayi Google Drive'iniza yukleyin (orn. `MyDrive/qwen2_egitim_seti/` klasorune):
   - `qwen2_train_updated.jsonl`
   - `qwen2_val_updated.jsonl`
   - `qwen2_test_updated.jsonl` (bu notebook'ta kullanilmiyor, degerlendirme notebook'unda lazim)
"""))

ft_cells.append(md("## 1) Kurulum"))

ft_cells.append(code("""%%capture
!pip install -q -U unsloth
!pip install -q -U --no-deps trl==0.9.6 peft accelerate bitsandbytes
"""))

ft_cells.append(md("## 2) Google Drive baglama ve veri yollari"))

ft_cells.append(code("""from google.colab import drive
drive.mount('/content/drive')

# Kendi klasor yolunuza gore duzenleyin
DATA_DIR = "/content/drive/MyDrive/qwen2_egitim_seti"
TRAIN_PATH = f"{DATA_DIR}/qwen2_train_updated.jsonl"
VAL_PATH   = f"{DATA_DIR}/qwen2_val_updated.jsonl"
OUTPUT_DIR = f"{DATA_DIR}/qwen2_5_7b_finetuned"

import os
assert os.path.exists(TRAIN_PATH), f"Bulunamadi: {TRAIN_PATH}"
assert os.path.exists(VAL_PATH), f"Bulunamadi: {VAL_PATH}"
print("Veri dosyalari hazir.")
"""))

ft_cells.append(md("## 3) Modeli 4-bit (QLoRA) yukle"))

ft_cells.append(code("""from unsloth import FastLanguageModel
import torch

MAX_SEQ_LENGTH = 4096   # Colab Pro (A100/L4, 40GB) icin; T4 (16GB) kullaniyorsaniz 2048'e dusurun
DTYPE = None            # otomatik (bf16 destekleniyorsa bf16, yoksa fp16)
LOAD_IN_4BIT = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = DTYPE,
    load_in_4bit = LOAD_IN_4BIT,
)
"""))

ft_cells.append(md("""## 4) LoRA adaptorlerini ekle

Parametreler proje dokumantasyonundaki (`PROJE_VE_DATASET_DOKUMANTASYONU.md`) onerilere gore:
`r=32`, `alpha=64`, tum linear katmanlar hedefleniyor."""))

ft_cells.append(code("""model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
"""))

ft_cells.append(md("""## 5) Veri setini yukle ve ChatML formatina donustur

Veri setindeki her satir zaten `{"messages": [{"role": "system", ...}, {"role": "user", ...},
{"role": "assistant", ...}]}` seklinde ChatML formatinda. Qwen2.5'in kendi chat template'i
`tokenizer.apply_chat_template` ile bu yapiyi doğru sekilde metne cevirir."""))

ft_cells.append(code("""from datasets import load_dataset

train_ds = load_dataset("json", data_files=TRAIN_PATH, split="train")
val_ds   = load_dataset("json", data_files=VAL_PATH, split="train")

def formatting_func(example):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

train_ds = train_ds.map(formatting_func)
val_ds   = val_ds.map(formatting_func)

print(f"Train: {len(train_ds)} kayit, Val: {len(val_ds)} kayit")
print("--- Ornek formatlanmis kayit ---")
print(train_ds[0]["text"][:800])
"""))

ft_cells.append(md("""## 6) Egitim ayarlari ve SFTTrainer

`train_on_responses_only` ile loss SADECE assistant (draft/karar) turlerinde hesaplanir;
sistem promptu ve kullanici JSON'u uzerinden loss alinmaz (dogru davranis budur)."""))

ft_cells.append(code("""from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

training_args = SFTConfig(
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    gradient_accumulation_steps = 8,     # efektif batch = 32 (Colab Pro A100 icin)
    warmup_ratio = 0.05,
    num_train_epochs = 3,
    learning_rate = 2e-4,                # QLoRA icin dokumantasyon onerisi
    lr_scheduler_type = "cosine",
    logging_steps = 20,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    seed = 3407,
    output_dir = "outputs",
    save_strategy = "epoch",
    eval_strategy = "epoch",
    bf16 = torch.cuda.is_bf16_supported(),
    fp16 = not torch.cuda.is_bf16_supported(),
    report_to = "none",
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    packing = False,
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_ds,
    eval_dataset = val_ds,
    args = training_args,
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\\n",
    response_part = "<|im_start|>assistant\\n",
)
"""))

ft_cells.append(md("## 7) Egitimi baslat"))

ft_cells.append(code("""trainer_stats = trainer.train()
print(trainer_stats)
"""))

ft_cells.append(md("## 8) Modeli kaydet (LoRA adaptoru + birlestirilmis 16-bit model)"))

ft_cells.append(code("""import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sadece LoRA adaptoru (kucuk, hizli, Drive'a kolay yuklenir)
model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")

# Birlestirilmis (merged) 16-bit tam model - degerlendirme notebook'unda dogrudan kullanmak icin
model.save_pretrained_merged(
    f"{OUTPUT_DIR}/merged_16bit", tokenizer, save_method="merged_16bit"
)

print("Kaydedildi:", OUTPUT_DIR)
"""))

ft_cells.append(md("""## Sonraki adim

Fine-tuning bittikten sonra **02_evaluate.ipynb** notebook'unu acip
`{OUTPUT_DIR}/merged_16bit` yolunu vererek test seti uzerinde yuzdesel dogruluk
raporunu alabilirsiniz."""))

with open(os.path.join(HERE, "01_finetune_qwen25_7b.ipynb"), "w", encoding="utf-8") as f:
    json.dump(notebook(ft_cells), f, ensure_ascii=False, indent=1)

print("01_finetune_qwen25_7b.ipynb yazildi.")


# =====================================================================
# NOTEBOOK 2: DEGERLENDIRME (YUZDESEL DOGRULUK RAPORU)
# =====================================================================

ev_cells = []

ev_cells.append(md("""# Qwen2.5-7B Fine-Tune Sonuclarini Degerlendirme — Yuzdesel Dogruluk Raporu

Bu notebook, fine-tune edilmis modeli `qwen2_test_updated.jsonl` test seti uzerinde calistirir
ve asagidaki metrikleri **yuzde (%)** olarak raporlar:

1. **JSON Gecerlilik Orani** - modelin cevabi parse edilebilir, 7 alani da dolu bir JSON mu?
2. **Alan Bazli Dogruluk** - `target_unit`, `response_type`, `process_status` alanlari test
   setindeki (altin/ground-truth) degerle birebir eslesiyor mu?
3. **Bicim/Kural Uyum Orani** - taslak (`draft`) minimum kelime sayisi, dogru arz/rica kapanisi
   kurallarina uyuyor mu?
4. **Mevzuat Sadakati (RAG Faithfulness)** - `selected_legislation` doluysa, model taslakta bu
   mevzuata gercekten atif yapmis mi (halusinasyon var mi)?
5. **Genel Dogruluk Skoru** - yukaridaki metriklerin agirlikli ortalamasi, TEK bir yuzde olarak.
"""))

ev_cells.append(md("## 1) Kurulum"))

ev_cells.append(code("""%%capture
!pip install -q -U unsloth
!pip install -q -U --no-deps trl==0.9.6 peft accelerate bitsandbytes
"""))

ev_cells.append(md("## 2) Drive baglama, model ve test verisini yukle"))

ev_cells.append(code("""from google.colab import drive
drive.mount('/content/drive')

DATA_DIR = "/content/drive/MyDrive/qwen2_egitim_seti"
TEST_PATH = f"{DATA_DIR}/qwen2_test_updated.jsonl"

# 01_finetune notebook'unda kaydedilen birlestirilmis (merged) model yolu.
# LoRA adaptoru + orijinal modeli ayri kullanmak isterseniz asagidaki
# 'MODEL_PATH' yerine base model adini ve ADAPTER_PATH'i ayarlayin.
MODEL_PATH = f"{DATA_DIR}/qwen2_5_7b_finetuned/merged_16bit"

import os
assert os.path.exists(TEST_PATH), f"Bulunamadi: {TEST_PATH}"
print("Test verisi hazir:", TEST_PATH)
"""))

ev_cells.append(code("""from unsloth import FastLanguageModel
import torch

MAX_SEQ_LENGTH = 4096   # finetune notebook'uyla ayni deger olmali

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,
    load_in_4bit = True,   # inference'i hizlandirmak icin 4-bit yukleniyor
)
FastLanguageModel.for_inference(model)  # Unsloth'un 2x hizli inference modu
"""))

ev_cells.append(md("## 3) Test setini yukle"))

ev_cells.append(code("""import json

test_records = []
with open(TEST_PATH, encoding="utf-8") as f:
    for line in f:
        test_records.append(json.loads(line))

print(f"Test seti: {len(test_records)} kayit")
"""))

ev_cells.append(md("""## 4) Model cevabi uretme ve JSON parse fonksiyonlari

`--limit` ile once kucuk bir ornekte (orn. 30 kayit) hizli test yapip, sorun yoksa
tum test setinde (482 kayit) calistirmanizi oneririz (T4'te ~482 kayit ~30-45 dk surebilir)."""))

ev_cells.append(code("""import re

REQUIRED_KEYS = [
    "performed_actions", "target_unit", "response_type",
    "process_status", "result_information", "process_information", "draft",
]

def generate_response(system_content, user_content, max_new_tokens=1024):
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # deterministik degerlendirme icin greedy
            temperature=1.0,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def try_parse_json(raw_text):
    text = raw_text.strip()
    # Bazen model markdown code-fence icine JSON dondurebilir, temizle
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    try:
        return json.loads(text)
    except Exception:
        # metindeki ilk { ile son } arasini almayi dene
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return None
        return None
"""))

ev_cells.append(md("## 5) Bicim/kural kontrolu (arz-rica kapanisi, mevzuat atfi, min kelime)"))

ev_cells.append(code("""CLOSING_RICA = ("rica ederim", "rica ederiz")
CLOSING_ARZ = ("arz ederim", "arz ederiz")
CLOSING_ARZ_RICA = ("arz ve rica ederim", "arz ve rica ederiz")

def check_closing(draft: str, sender_type: str) -> bool:
    lower = draft.lower()
    has_rica = any(p in lower for p in CLOSING_RICA)
    has_arz = any(p in lower for p in CLOSING_ARZ)
    has_combo = any(p in lower for p in CLOSING_ARZ_RICA)
    sender_type = (sender_type or "").upper()
    if sender_type in ("VATANDAS", "OZEL_KURULUS"):
        return has_rica and not has_arz
    elif sender_type == "KAMU_KURUMU":
        return has_combo
    else:
        return has_arz and not has_rica

def check_legislation_citation(draft: str, legislation: list) -> bool:
    if not legislation:
        return True  # atif zorunlu degil
    for leg in legislation:
        law_num = re.search(r"\\d+", leg.get("law_name") or "")
        art_num = re.search(r"\\d+", leg.get("article") or "")
        if (law_num and law_num.group(0) in draft) or (art_num and art_num.group(0) in draft):
            return True
    return False

def check_min_words(draft: str, min_words: int = 180) -> bool:
    return len(draft.split()) >= min_words
"""))

ev_cells.append(md("## 6) Degerlendirme dongusu"))

ev_cells.append(code("""from tqdm.auto import tqdm

LIMIT = None   # once orn. 30 ile test edin, sonra None yaparak tum test setinde calistirin

records_to_eval = test_records[:LIMIT] if LIMIT else test_records

results = []
for rec in tqdm(records_to_eval):
    msgs = {m["role"]: m["content"] for m in rec["messages"]}
    system_content = msgs["system"]
    user_content = msgs["user"]
    gold = json.loads(msgs["assistant"])
    user_data = json.loads(user_content)

    raw_output = generate_response(system_content, user_content)
    pred = try_parse_json(raw_output)

    row = {
        "sender_type": user_data.get("sender_type"),
        "legislation": user_data.get("selected_legislation") or [],
        "gold": gold,
        "pred": pred,
        "raw_output": raw_output,
    }
    results.append(row)

print(f"{len(results)} kayit uzerinde uretim tamamlandi.")
"""))

ev_cells.append(md("## 7) Metrikleri hesapla ve yuzdesel raporu yazdir"))

ev_cells.append(code("""n = len(results)

json_valid = 0
json_valid_complete = 0   # gecerli JSON + 7 alan da dolu
target_unit_correct = 0
response_type_correct = 0
process_status_correct = 0
closing_correct = 0
legislation_faithful = 0
min_words_ok = 0

evaluable_for_fields = 0  # JSON parse olan kayitlar (alan karsilastirmasi icin)

for row in results:
    pred = row["pred"]
    gold = row["gold"]

    if pred is not None:
        json_valid += 1
        has_all_keys = all(k in pred and str(pred[k]).strip() not in ("", "[]") for k in REQUIRED_KEYS)
        if has_all_keys:
            json_valid_complete += 1

        evaluable_for_fields += 1
        if str(pred.get("target_unit", "")).strip().lower() == str(gold.get("target_unit", "")).strip().lower():
            target_unit_correct += 1
        if pred.get("response_type") == gold.get("response_type"):
            response_type_correct += 1
        if pred.get("process_status") == gold.get("process_status"):
            process_status_correct += 1

        draft = str(pred.get("draft", ""))
        if draft:
            if check_closing(draft, row["sender_type"]):
                closing_correct += 1
            if check_legislation_citation(draft, row["legislation"]):
                legislation_faithful += 1
            if check_min_words(draft):
                min_words_ok += 1

def pct(x, total=n):
    return round(100 * x / total, 2) if total else 0.0

report = {
    "toplam_kayit": n,
    "json_gecerlilik_orani_%": pct(json_valid),
    "json_tam_alanli_orani_%": pct(json_valid_complete),
    "target_unit_dogruluk_%": pct(target_unit_correct),
    "response_type_dogruluk_%": pct(response_type_correct),
    "process_status_dogruluk_%": pct(process_status_correct),
    "kapanis_kural_uyum_%": pct(closing_correct),
    "mevzuat_sadakati_%": pct(legislation_faithful),
    "min_kelime_uyum_%": pct(min_words_ok),
}

# Genel dogruluk skoru: temel metriklerin agirlikli ortalamasi
weights = {
    "json_tam_alanli_orani_%": 0.15,
    "target_unit_dogruluk_%": 0.20,
    "response_type_dogruluk_%": 0.15,
    "process_status_dogruluk_%": 0.15,
    "kapanis_kural_uyum_%": 0.15,
    "mevzuat_sadakati_%": 0.20,
}
genel_skor = sum(report[k] * w for k, w in weights.items())
report["GENEL_DOGRULUK_SKORU_%"] = round(genel_skor, 2)

print("=" * 60)
print(" QWEN2.5-7B FINE-TUNE DEGERLENDIRME RAPORU")
print("=" * 60)
for k, v in report.items():
    print(f"{k:35s}: {v}")
print("=" * 60)
"""))

ev_cells.append(md("""## 8) (Opsiyonel) Hatali ornekleri incele

Raporun ardindan modelin nerede hata yaptigini gormek icin, `pred` alani `None` olan
veya alan uyusmazligi olan kayitlari asagidaki gibi filtreleyip inceleyebilirsiniz."""))

ev_cells.append(code("""hatali_ornekler = [
    r for r in results
    if r["pred"] is None or r["pred"].get("target_unit") != r["gold"].get("target_unit")
][:5]

for r in hatali_ornekler:
    print("-" * 40)
    print("GOLD target_unit:", r["gold"].get("target_unit"))
    print("PRED target_unit:", (r["pred"] or {}).get("target_unit"))
    print("RAW OUTPUT (ilk 400 karakter):", r["raw_output"][:400])
"""))

ev_cells.append(code("""# Raporu Drive'a JSON olarak kaydet
import json as _json
with open(f"{DATA_DIR}/degerlendirme_raporu.json", "w", encoding="utf-8") as f:
    _json.dump(report, f, ensure_ascii=False, indent=2)
print("Kaydedildi:", f"{DATA_DIR}/degerlendirme_raporu.json")
"""))

with open(os.path.join(HERE, "02_evaluate_qwen25_7b.ipynb"), "w", encoding="utf-8") as f:
    json.dump(notebook(ev_cells), f, ensure_ascii=False, indent=1)

print("02_evaluate_qwen25_7b.ipynb yazildi.")
