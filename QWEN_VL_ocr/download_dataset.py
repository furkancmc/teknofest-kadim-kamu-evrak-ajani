"""OCR kaynak metnini Hugging Face'den indirir.

Derleme: erdem-erdem/Turkish-Law-Documents-700k-clustered
Icerik: Yargitay ve Danistay'in kamuya acik karar metinleri
  (karararama.yargitay.gov.tr, kararara.danistay.gov.tr).
Asil mahkeme dosyasi / tarama PDF'si indirilmez; duz metin kayitlari gelir.
Sonraki adim: generate_dataset_1000.py ile HTML/PNG olarak yeniden basmak.
"""
from datasets import load_dataset

dataset = load_dataset("erdem-erdem/Turkish-Law-Documents-700k-clustered")
dataset.save_to_disk("./turkish_law_dataset")

print(dataset)
print(dataset["train"][0])
