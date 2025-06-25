import os
import cv2
import csv
from ultralytics import YOLO

# YOLO modeli yükle
model = YOLO("yolov8n.pt")  

# Yoğunluk sınıflandırması için eşikler
def get_density_label(count):
    if count < 5:
        return "Low"
    elif count <= 15:
        return "Mid"
    else:
        return "High"

# Klasörler
video_folder = "videos"
output_folder = "output_csv"
os.makedirs(output_folder, exist_ok=True)


extensions = ('.mp4', '.avi', '.mov', '.mkv')

print(f" '{video_folder}' klasöründeki videolar işleniyor...")

for filename in os.listdir(video_folder):
    if not filename.lower().endswith(extensions):
        continue

    path = os.path.join(video_folder, filename)
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print(f"Hata: {path} açılamadı.")
        continue

    csv_name = os.path.splitext(filename)[0] + "_veri.csv"
    csv_path = os.path.join(output_folder, csv_name)

    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Video", "Frame", "Vehicle_Count", "Density_Label"])

        frame_id = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model(frame, conf=0.3, classes=[2, 3, 5, 7])  # araç sınıfları
            vehicle_count = len(results[0].boxes.cls)

            label = get_density_label(vehicle_count)
            writer.writerow([filename, frame_id, vehicle_count, label])
            frame_id += 1

    cap.release()
    print(f" Veri çıkarıldı: {csv_path}")

print("\n Tüm videolar işlendi.")
