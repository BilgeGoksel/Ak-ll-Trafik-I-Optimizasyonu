# GERÇEK ZAMANLI ARAÇ SAYIMI İLE TRAFiK YOĞUNLUĞUNA GÖRE 
# AKILLI TRAFİK IŞIĞI SÜRE TAHMİNİ

Bu projenin amacı, gerçek zamanlı video akışlarından araç sayımı yaparak trafik yoğunluğunu
belirlemek ve bu bilgiye dayanarak trafik ışığı sürelerini otomatik olarak ayarlayan bir sistem
geliştirmektir. Sistem, özellikle yoğun saatlerde trafik sıkışıklığını azaltmayı, düşük yoğunlukta ise
boş ışık beklemeyi önlemeyi hedeflemektedir.

## Kullanılan Yöntem ve Araçlar

Bu sistem, üç ana bileşenden oluşmaktadır:

**Araç Sayımı:** YOLOv8 ile video karelerinde otomobil, otobüs, kamyon gibi araç sınıfları
algılanarak araç sayısı elde edilir.

**Yoğunluk Sınıflandırması:** Araç sayısına göre trafik yoğunluğu “Low”, “Mid” veya “High”
şeklinde sınıflandırılır.

**Tahmin Modeli:** Karar ağacı tabanlı bir model ile bu yoğunluk etiketine karşılık gelen trafik
ışığı süreleri (kırmızı/yeşil) tahmin edilir.

▪ YOLOv8 Tabanlı Araç Tespiti

Araç tespiti için “YOLOv8n” (You Only Look Once) nesne algılama modeli kullanılmıştır.
veri_uretici.py ve arac_sayim.py dosyaları sayesinde, videolardaki her karede yer alan araç sayısı
belirlenmiştir. Araç sınıfları (car, bus, truck) filtrelenerek Vehicle_Count sütunu oluşturulmuştur.

Veri üretimi veri_uretici.py dosyasında tanımlanan YOLO modeli ile gerçekleştirilmiştir. Her
karedeki araç sayısı ve yoğunluk etiketi bir CSV dosyasına aktarılmıştır.
Her karedeki araç sayısı aşağıdaki kurala göre etiketlenmiştir:
- Araç sayısı < 5 → “Low”
- 5 ≤ Araç sayısı ≤ 15 → “Mid”
- Araç sayısı > 15 → “High”
Bu etiketler, yoğunluk sınıfını belirtmek üzere Density_Label olarak kaydedilmiştir.

▪ Makine Öğrenmesi ile Yoğunluk Tahmini

model_egitim.py dosyasında, yoğunluk etiketleri sayısallaştırılmış (Low: 0, Mid: 1, High: 2) ve
Decision Tree Classifier modeli eğitilmiştir. Model, ortalama araç sayısına göre trafik yoğunluğunu
tahmin etmek üzere yapılandırılmıştır. Eğitim verisi traffic_light_summary.csv dosyasından
alınmıştır.
Modelin performansı doğruluk (accuracy) metriği ile değerlendirilmiş ve sonuçlar başarılı
bulunmuştur.

▪ Simülasyon ve Görselleştirme

simulasyon.py dosyası, modelin gerçek zamanlı simülasyonunu yapar. Video karelerinde tespit
edilen araç sayısına göre:
- Yoğunluk etiketi belirlenir,
- Eğitilmiş modelden tahmin edilen yoğunluk alınır,
- Bu yoğunluğa göre kırmızı ve yeşil ışık süreleri atanır.
Simülasyon sırasında trafik ışığı animasyonu, araç sayısı, yoğunluk etiketi ve ışık süreleri OpenCV
ile kullanıcıya görsel olarak sunulmaktadır.

## Sonuçlar ve Değerlendirme
Model, rule-based olarak etiketlenmiş yoğunluk sınıflarını yüksek doğrulukla tahmin etmeyi
başarmıştır. YOLOv8 modelinin kullanılması, gerçek zamanlı performans elde etmeye olanak
sağlamıştır. Ancak küçük model (yolov8n) bazı küçük/uzaktaki araçları kaçırabilmekte, bu nedenle
conf parametresi 0.3'e düşürülerek tespit hassasiyeti artırılmıştır.
