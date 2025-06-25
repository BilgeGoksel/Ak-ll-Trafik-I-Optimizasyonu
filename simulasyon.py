import cv2
import os
import pandas as pd
import joblib
from ultralytics import YOLO
import time
import numpy as np # For drawing text background

# --- 1. Modelleri ve Ayarları Yükle ---

# YOLO modelini yükle (araç tespiti için)
# Bu modelin 'yolov8n.pt' adıyla script'in çalıştığı dizinde olması gerekir.
try:
    model_yolo = YOLO('yolov8n.pt')
except Exception as e:
    print(f"Hata: YOLO modeli yüklenemedi. 'yolov8n.pt' dosyasının mevcut olduğundan emin olun. Hata: {e}")
    exit()

# Eğitilmiş trafik yoğunluğu modelini yükle
# Bu modelin 'trafik_yogunluk_modeli.joblib' adıyla script'in çalıştığı dizinde olması gerekir.
try:
    model_density = joblib.load("trafik_yogunluk_modeli.joblib")
except Exception as e:
    print(f"Hata: Yoğunluk tahmin modeli yüklenemedi. 'trafik_yogunluk_modeli.joblib' dosyasının mevcut olduğundan emin olun. Hata: {e}")
    exit()

# Video klasörü ve desteklenen uzantılar
video_folder_path = "videos"
supported_extensions = ('.mp4', '.avi', '.mov', '.mkv')

# Yoğunluk etiketleri ve tersine haritalama (model_egitim.py'deki gibi)
label_map = {"Düşük": 0, "Orta": 1, "Yüksek": 2}
reverse_map = {0: "Düşük", 1: "Orta", 2: "Yüksek"}

# --- 2. Trafik Işığı Durum Yönetimi ---

# Trafik ışığı fazları
LIGHT_RED = 0
LIGHT_YELLOW = 1
LIGHT_GREEN = 2

# Trafik ışığı renkleri (OpenCV BGR formatında)
COLOR_RED = (0, 0, 255)      # Kırmızı
COLOR_YELLOW = (0, 255, 255) # Sarı
COLOR_GREEN = (0, 255, 0)    # Yeşil
COLOR_DARK = (50, 50, 50)    # Sönük ışık rengi

# Trafik ışığı süreleri (saniye cinsinden)
# Bu süreler, yoğunluğa göre dinamik olarak ayarlanacaktır.
# Sabit sarı ışık süresi
YELLOW_LIGHT_DURATION = 3

# Toplam döngü süresi (Örnek olarak, bu değer ayarlanabilir)
TOTAL_CYCLE_TIME = 45 # seconds

# Başlangıç durumu
current_light_state = LIGHT_RED
time_in_current_state = 0.0 # Mevcut fazda geçen süre
current_light_duration = 0 # Mevcut fazın toplam süresi

# --- 3. Yardımcı Fonksiyonlar ---

def get_light_durations(avg_vehicle_count):
    """
    Ortalama araç sayısına göre trafik yoğunluğunu ve ışık sürelerini belirler.
    Bu kısım, yogunluk_tahmini.py ve model_egitim.py mantığını birleştirir.
    """
    # Yoğunluk tahmini (model_egitim.py'den yüklenen model ile)
    try:
        density_label_numeric = model_density.predict([[avg_vehicle_count]])[0]
        density = reverse_map[density_label_numeric]
    except Exception as e:
        print(f"Uyarı: Yoğunluk tahmini yapılamadı, varsayılan 'Orta' kullanılıyor. Hata: {e}")
        density = "Orta"
        density_label_numeric = label_map["Orta"]

    # yogunluk_tahmini.py'deki kırmızı ışık süresi mantığı
    if density == "Düşük":
        red_duration = 30
    elif density == "Orta":
        red_duration = 20
    else: # Yüksek
        red_duration = 10

    # Yeşil ışık süresini toplam döngü süresinden hesapla
    green_duration = TOTAL_CYCLE_TIME - red_duration - YELLOW_LIGHT_DURATION

    # Negatif süre olmaması için kontrol
    if green_duration < 0:
        green_duration = 5 # Minimum yeşil ışık süresi

    return density, red_duration, green_duration

def draw_traffic_light(frame, light_state, red_color, yellow_color, green_color):
    """
    Kare üzerine trafik ışığı animasyonunu çizer.
    """
    # Işık direği ve kutusu için basit çizimler
    pole_x, pole_y = frame.shape[1] - 80, 50
    cv2.rectangle(frame, (pole_x - 10, pole_y), (pole_x + 10, pole_y + 150), (100, 100, 100), -1) # Direk
    cv2.rectangle(frame, (pole_x - 20, pole_y - 30), (pole_x + 20, pole_y + 160), (30, 30, 30), -1) # Kutu

    # Işıklar
    cv2.circle(frame, (pole_x, pole_y + 20), 15, red_color, -1)    # Kırmızı
    cv2.circle(frame, (pole_x, pole_y + 75), 15, yellow_color, -1) # Sarı
    cv2.circle(frame, (pole_x, pole_y + 130), 15, green_color, -1) # Yeşil

def draw_text_with_background(img, text, org, fontFace, fontScale, color, thickness, bg_color, padding=5):
    """
    Arka planlı metin çizmek için yardımcı fonksiyon.
    """
    (text_width, text_height), baseline = cv2.getTextSize(text, fontFace, fontScale, thickness)
    x, y = org
    cv2.rectangle(img, (x - padding, y - text_height - baseline - padding),
                  (x + text_width + padding, y + padding), bg_color, -1)
    cv2.putText(img, text, org, fontFace, fontScale, color, thickness, cv2.LINE_AA)


# --- 4. Ana Simülasyon Döngüsü ---

if not os.path.isdir(video_folder_path):
    print(f"Hata: '{video_folder_path}' klasörü bulunamadı. Lütfen videolarınızı bu klasöre yerleştirin.")
    exit()

print(f"'{video_folder_path}' klasöründeki videolar işlenecek ve simüle edilecek...")

for filename in os.listdir(video_folder_path):
    if filename.lower().endswith(supported_extensions):
        video_path = os.path.join(video_folder_path, filename)
        print(f"\n--- Simülasyon Başlatılıyor: {video_path} ---")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Hata: {video_path} video dosyası açılamadı. Atlanıyor.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay_ms = int(1000 / fps) if fps > 0 else 30 # Kareler arası bekleme süresi

        frame_number = 0
        last_frame_time = time.time() # Son karenin işlenme zamanı

        # İlk yoğunluk ve süreleri belirle
        # İlk karedeki araç sayımına göre veya varsayılan bir değerle başlayabiliriz.
        # Burada simülasyonun başında varsayılan bir değerle başlıyoruz.
        current_density, red_duration_for_scenario, green_duration_for_scenario = get_light_durations(5) # Başlangıçta düşük yoğunluk varsayımı
        current_light_duration = red_duration_for_scenario # İlk faz kırmızı

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # --- A. YOLO Tahmini ve Araç Sayımı ---
            results = model_yolo(frame, conf=0.5, classes=[2, 3, 5, 7], verbose=False) # Otomobil, motosiklet, otobüs, kamyon
            vehicle_count = len(results[0].boxes.cls)

            # --- B. Yoğunluk Tahmini ve Işık Süreleri Belirleme ---
            # Her karede yoğunluğu yeniden hesaplayabiliriz veya belirli aralıklarla yapabiliriz.
            # Simülasyonun dinamik olması için her karede yapalım.
            current_density, red_duration_for_scenario, green_duration_for_scenario = get_light_durations(vehicle_count)

            # --- C. Trafik Işığı Durumunu Güncelleme ---
            current_time = time.time()
            delta_time = current_time - last_frame_time
            last_frame_time = current_time

            time_in_current_state += delta_time

            if time_in_current_state >= current_light_duration:
                # Faz değişimi
                time_in_current_state = 0 # Süreyi sıfırla

                if current_light_state == LIGHT_RED:
                    current_light_state = LIGHT_GREEN
                    current_light_duration = green_duration_for_scenario
                elif current_light_state == LIGHT_GREEN:
                    current_light_state = LIGHT_YELLOW
                    current_light_duration = YELLOW_LIGHT_DURATION
                elif current_light_state == LIGHT_YELLOW:
                    current_light_state = LIGHT_RED
                    current_light_duration = red_duration_for_scenario

            # --- D. Görselleştirme ---

            # Tespit edilen araçların etrafına kutu çiz
            for r in results[0].boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Yeşil kutu
                cv2.putText(frame, f"{model_yolo.names[int(class_id)]} {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Trafik ışığı renklerini ayarla
            red_light_color = COLOR_DARK
            yellow_light_color = COLOR_DARK
            green_light_color = COLOR_DARK

            if current_light_state == LIGHT_RED:
                red_light_color = COLOR_RED
            elif current_light_state == LIGHT_YELLOW:
                yellow_light_color = COLOR_YELLOW
            elif current_light_state == LIGHT_GREEN:
                green_light_color = COLOR_GREEN

            draw_traffic_light(frame, current_light_state, red_light_color, yellow_light_color, green_light_color)

            # Bilgileri ekrana yazdır
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_color = (255, 255, 255) # Beyaz
            bg_color = (0, 0, 0, 0.6) # Yarı saydam siyah

            # Araç Sayısı
            draw_text_with_background(frame, f"Arac Sayisi: {vehicle_count}", (10, 30), font, font_scale, text_color, thickness, bg_color)
            # Yoğunluk
            draw_text_with_background(frame, f"Yogunluk: {current_density}", (10, 60), font, font_scale, text_color, thickness, bg_color)
            # Kırmızı Işık Süresi
            draw_text_with_background(frame, f"Kirmizi Isik Suresi: {red_duration_for_scenario}s", (10, 90), font, font_scale, text_color, thickness, bg_color)
            # Yeşil Işık Süresi
            draw_text_with_background(frame, f"Yesil Isik Suresi: {green_duration_for_scenario}s", (10, 120), font, font_scale, text_color, thickness, bg_color)

            # Mevcut Işık Durumu ve Kalan Süre
            light_name = ""
            if current_light_state == LIGHT_RED: light_name = "Kirmizi"
            elif current_light_state == LIGHT_YELLOW: light_name = "Sari"
            elif current_light_state == LIGHT_GREEN: light_name = "Yesil"

            remaining_time = max(0, int(current_light_duration - time_in_current_state))
            draw_text_with_background(frame, f"Mevcut Isik: {light_name} ({remaining_time}s)", (10, 150), font, font_scale, text_color, thickness, bg_color)


            # Pencereyi göster
            cv2.imshow('Akilli Trafik Isigi Simülasyonu', frame)

            # Çıkış için 'q' tuşuna basılıp basılmadığını kontrol et
            if cv2.waitKey(frame_delay_ms) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f" '{filename}' videosu için simülasyon tamamlandı.")

print("\n--- Tüm videolar işlendi ve simülasyon tamamlandı. ---")
