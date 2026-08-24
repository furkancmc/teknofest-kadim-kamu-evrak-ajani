"""
11 dataset_XXX.jsonl dosyasini (1100 kayit) birlestirip, primary_topic bazinda
stratifiye edilmis train/val/test ayrimi yapar ve Qwen chat-format (messages)
SFT verisine cevirir.

Cikti (finetune_data/ klasorunde):
  train.jsonl, val.jsonl, test.jsonl   -> {"messages": [...]} formatinda, egitime hazir
  train_raw.jsonl, val_raw.jsonl, test_raw.jsonl -> orijinal document_text+labels (referans/debug icin)
  split_stats.txt -> topic bazinda dagilim raporu
"""

import json
import glob
import os
import random
from collections import defaultdict

random.seed(50607080)  # FINAL round -- her round'da degistiriliyor, ayni test setine tekrar tekrar
                        # "gore ayarlanmis" (test-set overfitting) olmamak icin.

OUT_DIR = "finetune_data"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# kalan ~0.10 test'e gider

SYSTEM_PROMPT_INFERENCE = """Sen bir belediye evrak analiz asistanısın. Sana verilen resmi evrakı analiz ederek tam olarak şu 7 alanı içeren geçerli bir JSON üret:

{
  "document_type": "",
  "sender_type": "",
  "primary_topic": "",
  "requested_action": "",
  "key_information": [],
  "missing_information": [],
  "summary": ""
}

Kurallar:
- document_type yalnızca şu değerlerden biri olabilir: TALEP_DILEKCE_BASVURUSU, SIKAYET_IHBAR_BASVURUSU, BILGI_TALEBI, ITIRAZ_IDARI_BASVURU, KURUMLAR_ARASI_RESMI_YAZI
- sender_type yalnızca şu değerlerden biri olabilir: VATANDAS, KAMU_KURUMU, OZEL_KURULUS, YARGI_MERCII
- primary_topic, evrağın konusuna ve ilgili belediye birimine göre en uygun sınıfı yansıtmalıdır.
- key_information yalnızca belgede açıkça bulunan, işlemin anlaşılması için önemli bilgileri içermelidir (belgede olmayan bilgi eklenmez).
- missing_information, belgeyi işleme almak için gereken ama belgede bulunmayan bilgileri listeler; belge yeterliyse boş dizi ([]) olmalıdır.
- requested_action, göndereni tarafından istenen işlemi kısa ve somut şekilde belirtmelidir.
- summary, 1-2 cümlelik tarafsız bir özet olmalıdır.
- Mevzuat, kaynak, öncelik gibi ek alanlar ÜRETME. Sadece yukarıdaki 7 alanı döndür.
- Sadece geçerli JSON döndür, başka hiçbir açıklama veya markdown ekleme."""


def to_messages(record: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_INFERENCE},
            {"role": "user", "content": record["document_text"]},
            {"role": "assistant", "content": json.dumps(record["labels"], ensure_ascii=False)},
        ]
    }


def main():
    by_topic = defaultdict(list)
    total_loaded = 0

    for fn in sorted(glob.glob("dataset_*.jsonl")):
        with open(fn, encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
        for rec in recs:
            topic = rec["labels"]["primary_topic"]
            by_topic[topic].append(rec)
            total_loaded += 1

    train, val, test = [], [], []
    stats_lines = []

    for topic, recs in sorted(by_topic.items()):
        recs = recs[:]
        random.shuffle(recs)
        n = len(recs)

        if n <= 2:
            n_val, n_test = 0, 0
        else:
            n_val = max(1, round(n * VAL_RATIO))
            n_test = max(1, round(n * (1 - TRAIN_RATIO - VAL_RATIO)))
            # train'de en az 1 kayit kalsin
            while n_val + n_test >= n:
                if n_test > 0:
                    n_test -= 1
                elif n_val > 0:
                    n_val -= 1
                else:
                    break

        topic_val = recs[:n_val]
        topic_test = recs[n_val:n_val + n_test]
        topic_train = recs[n_val + n_test:]

        train.extend(topic_train)
        val.extend(topic_val)
        test.extend(topic_test)

        stats_lines.append(f"{topic:45s} toplam={n:4d}  train={len(topic_train):4d}  val={len(topic_val):3d}  test={len(topic_test):3d}")

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    # ---- kaydet ----
    def save(name, records):
        raw_path = os.path.join(OUT_DIR, f"{name}_raw.jsonl")
        chat_path = os.path.join(OUT_DIR, f"{name}.jsonl")
        with open(raw_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(chat_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(to_messages(r), ensure_ascii=False) + "\n")

    save("train", train)
    save("val", val)
    save("test", test)

    with open(os.path.join(OUT_DIR, "split_stats.txt"), "w", encoding="utf-8") as f:
        f.write(f"Toplam kayit: {total_loaded}\n")
        f.write(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}\n")
        f.write(f"Toplam topic sayisi: {len(by_topic)}\n\n")
        f.write("\n".join(stats_lines))

    print(f"Toplam: {total_loaded}  Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    print(f"Cikti: {OUT_DIR}/ klasorunde (train/val/test .jsonl + _raw.jsonl + split_stats.txt)")


if __name__ == "__main__":
    main()
