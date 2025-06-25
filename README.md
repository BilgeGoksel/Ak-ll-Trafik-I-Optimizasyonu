# GERÇEK ZAMANLI ARAÇ SAYIMI İLE TRAFĠK YOĞUNLUĞUNA GÖRE 
# AKILLI TRAFİK IŞIĞI SÜRE TAHMİNİ
Bu projenin amacı, gerçek zamanlı video akışlarından araç sayımı yaparak trafik yoğunluğunu
belirlemek ve bu bilgiye dayanarak trafik ışığı sürelerini otomatik olarak ayarlayan bir sistem
geliştirmektir. Sistem, özellikle yoğun saatlerde trafik sıkışıklığını azaltmayı, düşük yoğunlukta ise
boş ışık beklemeyi önlemeyi hedeflemektedir.
## 3. Kullanılan Yöntem ve Araçlar
Bu sistem, üç ana bileşenden oluşmaktadır:
**Araç Sayımı:** YOLOv8 ile video karelerinde otomobil, otobüs, kamyon gibi araç sınıfları
algılanarak araç sayısı elde edilir.
**Yoğunluk Sınıflandırması:** Araç sayısına göre trafik yoğunluğu “Low”, “Mid” veya “High”
şeklinde sınıflandırılır.
**Tahmin Modeli:** Karar ağacı tabanlı bir model ile bu yoğunluk etiketine karşılık gelen trafik
ışığı süreleri (kırmızı/yeşil) tahmin edilir.
