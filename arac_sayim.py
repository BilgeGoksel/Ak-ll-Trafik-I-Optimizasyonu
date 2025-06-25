from ultralytics import YOLO
import cv2
import sys
import os
import csv

# YOLO modelini yükle
model = YOLO('yolov8n.pt')

video_folder_path = "videos"
output_folder = "output_csv"
os.makedirs(output_folder, exist_ok=True)

supported_extensions = ('.mp4', '.avi', '.mov', '.mkv')

if not os.path.isdir(video_folder_path):
    print(f"Hata: '{video_folder_path}' klasörü bulunamadı.")
    sys.exit()

print(f"'{video_folder_path}' klasöründeki videolar işlenecek...")

for filename in os.listdir(video_folder_path):
    if filename.lower().endswith(supported_extensions):
        video_path = os.path.join(video_folder_path, filename)
        print(f"\n--- İşleniyor: {video_path} ---")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Hata: {video_path} video dosyası açılamadı.")
            continue

        # Video FPS bilgisi alınır
        fps = cap.get(cv2.CAP_PROP_FPS)

        # CSV dosyası oluştur
        csv_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_count.csv")
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Video", "Frame", "Timestamp (s)", "Vehicle_Count"])

            frame_number = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # YOLO tahmini
                results = model(frame, conf=0.5, classes=[2, 3, 5, 7])
                vehicle_count = len(results[0].boxes.cls)

                timestamp = frame_number / fps
                writer.writerow([filename, frame_number, round(timestamp, 2), vehicle_count])

                frame_number += 1

        cap.release()
        print(f" '{filename}' için veri kaydedildi: {csv_path}")

# Pencereler için güvenli kapatma
try:
    cv2.destroyAllWindows()
except:
    pass

print("\n--- Tüm videolar işlendi ve CSV dosyaları kaydedildi. ---")
