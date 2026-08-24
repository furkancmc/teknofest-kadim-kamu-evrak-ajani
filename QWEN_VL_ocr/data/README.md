# OCR örnek görseller (A–D)

Tam 1000’lik eğitim seti repoda yoktur (`dataset_1000.zip` Drive’dadır). Burada **her bozulma seviyesinden bir örnek** vardır; dosya adındaki harf grubu, sayı orijinal örnek kimliğidir.

| Dosya | Grup | Tarama kalitesi |
|---|---|---|
| `A_0003.png` + `.txt` | A | Temiz baskı, bozma yok |
| `B_0077.png` + `.txt` | B | Çok az bozulma |
| `C_0065.png` + `.txt` | C | Belirgin bozukluk |
| `D_0064.png` + `.txt` | D | Bariz kötü tarama |

`.txt` dosyası o görselin OCR yer gerçeğidir (yeniden basılmış sayfa metni).

## Kaynak

Metinler Hugging Face derlemesi [`erdem-erdem/Turkish-Law-Documents-700k-clustered`](https://huggingface.co/datasets/erdem-erdem/Turkish-Law-Documents-700k-clustered) üzerinden alınır.

Birincil kurumlar (kamuya açık karar arama):

- Yargıtay — https://karararama.yargitay.gov.tr/
- Danıştay — https://kararara.danistay.gov.tr/

`generate_dataset_1000.py` düz metni HTML antetli sayfa olarak **yeniden basar**. Bu klasördeki PNG’ler taranmış asıl mahkeme evrakı değildir; sentetik sayfa görüntüsüdür. Belediye dilekçesi / CİMER / EBYS verisi yoktur.
