# OCR örnek görselleri (A–D)

Bu klasörde, OCR eğitim kümesindeki **dört bozulma seviyesinin her birinden bir örnek** bulunur. Dosya adındaki harf bozulma grubunu, sayı ise örneğin kaynak kümedeki kimliğini gösterir. 1000 görselden oluşan tam küme depoda yer almaz; `generate_dataset_1000.py` ile yeniden üretilebilir.

| Dosya | Grup | Tarama kalitesi |
|---|---|---|
| `A_0003.png` + `.txt` | A | Temiz baskı, bozulma yok |
| `B_0077.png` + `.txt` | B | Hafif bulanıklık, küçük kontrast kayması |
| `C_0065.png` + `.txt` | C | Belirgin bulanıklık, kontrast düşüşü, gürültü |
| `D_0064.png` + `.txt` | D | Düşük çözünürlük, ağır gürültü, eğrilik |

Her PNG'nin yanındaki `.txt` dosyası, o görselin doğru metin karşılığıdır (yer gerçeği).

## Kaynak

Metinler, Hugging Face üzerindeki açık [`erdem-erdem/Turkish-Law-Documents-700k-clustered`](https://huggingface.co/datasets/erdem-erdem/Turkish-Law-Documents-700k-clustered) derlemesinden alınmıştır. Derleme, kamuya açık karar arama sistemlerinden toplanan düz metinlerden oluşur:

- Yargıtay — https://karararama.yargitay.gov.tr/
- Danıştay — https://kararara.danistay.gov.tr/

`generate_dataset_1000.py` bu düz metinleri antetli birer belge sayfası olarak yeniden basar. Dolayısıyla buradaki görseller taranmış resmî evrak değil, eğitim için üretilmiş sentetik belge görüntüleridir.
