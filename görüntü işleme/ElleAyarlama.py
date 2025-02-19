from datetime import timedelta, datetime
import cv2
from cap_from_youtube import cap_from_youtube
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from collections import defaultdict
import numpy as np

# Global değişkenler
drawing = False  # Fare ile alan çizimi
ix, iy = -1, -1  # Başlangıç koordinatları
rois = []  # Birden fazla seçilen alanı saklamak için liste
current_roi = None  # Geçici ROI

# Fare olaylarını işleme fonksiyonu
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, current_roi, rois
    if event == cv2.EVENT_LBUTTONDOWN:  # Fare sol tık ile başlama
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:  # Fare hareket ediyorsa geçici ROI'yi güncelle
        if drawing:
            current_roi = (ix, iy, x - ix, y - iy)
    elif event == cv2.EVENT_LBUTTONUP:  # Sol tuş bırakıldığında ROI'yi kaydet
        drawing = False
        current_roi = (ix, iy, x - ix, y - iy)
        rois.append(current_roi)  # ROI'yi listeye ekle
        print(f"ROI eklendi: {current_roi}")

# OpenCV ile pencere oluştur ve fare olaylarını bağla
cv2.namedWindow('YOLO Detection', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('YOLO Detection', draw_rectangle)

# Video üzerinden frame'leri oku ve YOLO ile işlem yap
while True:
    # Frame'i oku
    ret, frame = cap.read()
    if not ret:
        break

    # Seçilen tüm alanları çiz
    for roi in rois:
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Mavi dikdörtgen

    # YOLO modelini tüm frame üzerinde çalıştır
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    # Tespit edilen kutuları ve izleme kimliklerini al
    boxes = results[0].boxes.xywh.cpu()  # Tespit edilen kutuların koordinatları
    track_ids = results[0].boxes.id.int().cpu().tolist()  # İzlenen nesnelerin kimlikleri
    vehicle_types = results[0].boxes.cls.numpy()  # Tespit edilen nesnelerin sınıfları
    names = results[0].names  # Sınıf isimleri

    # Annotator'ı başlat
    annotator = Annotator(frame, line_width=2, font_size=10)

    # ROI'ler içinde tespit edilen araçları işleyin
    for box, track_id, vehicle_type in zip(boxes, track_ids, vehicle_types):
        class_label = names[vehicle_type]  # Sınıf adı

        if class_label in allowed_classes:
            x, y, w, h = box
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            # Seçilen alanlarda kontrol
            for roi in rois:
                roi_x, roi_y, roi_w, roi_h = roi
                if roi_x <= x1 <= roi_x + roi_w and roi_y <= y1 <= roi_y + roi_h:
                    # Araç bu ROI içinde ise Annotator ile çiz
                    color = class_colors.get(class_label, (255, 255, 255))  # Varsayılan renk beyaz
                    annotator.box_label([x1, y1, x2, y2], f'{class_label} ID: {track_id}', color=color)

    # Annotated frame'i al
    annotated_frame = annotator.result()

    # Annotated frame'i göster
    cv2.imshow("YOLO Detection", annotated_frame)

    # 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Seçilen tüm ROI'leri yazdır
print("Seçilen ROI'ler:")
for i, roi in enumerate(rois):
    print(f"ROI {i+1}: {roi}")

# Video akışını serbest bırak ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()
