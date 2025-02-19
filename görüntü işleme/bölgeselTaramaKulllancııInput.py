from datetime import timedelta
import cv2
from cap_from_youtube import cap_from_youtube
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from collections import defaultdict
import numpy as np

# Global değişkenler
drawing = False  # Fare ile alan çizimi
ix, iy = -1, -1  # Başlangıç koordinatları
roi = None  # Seçilen alan

# Fare olaylarını işleme fonksiyonu
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, roi
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi = (ix, iy, x - ix, y - iy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi = (ix, iy, x - ix, y - iy)

# YouTube videosunu al
youtube_url = 'https://youtu.be/MNn9qKG2UFI'
start_time = timedelta(seconds=5)  # YouTube videosunun başlangıç zamanı
cap = cap_from_youtube(youtube_url, 'best', start=start_time)
cap.set(cv2.CAP_PROP_FPS, 30)

# Pretrained YOLO modelini yükle
model = YOLO(r"C:\Users\erena\Desktop\Yolo\carDetectionTrain\model\yolo11l.pt").cuda()

allowed_classes = ['car', 'bus', 'motorcycle', 'truck']

# Her sınıf için renkleri tanımlayın
class_colors = {
    'car': (255, 0, 0),  # Kırmızı
    'bus': (0, 255, 0),  # Yeşil
    'motorcycle': (0, 0, 255),  # Mavi
    'truck': (0, 255, 255)  # Sarı
}

# İzleme geçmişini tutmak için bir defaultdict oluştur
track_history = defaultdict(lambda: [])

# OpenCV ile pencere oluştur
cv2.namedWindow('YOLO Detection', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('YOLO Detection', draw_rectangle)  # Fare olaylarını bağla

# Video üzerinden frame'leri oku ve YOLO ile işlem yap
while True:
    # Frame'i oku
    ret, frame = cap.read()
    if not ret:
        break

    # Seçilen alanı çizin (eğer varsa)
    if roi:
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Mavi dikdörtgen

    # YOLO modelini tüm frame üzerinde çalıştır
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    # Tespit edilen kutuları ve izleme kimliklerini al
    boxes = results[0].boxes.xywh.cpu()  # Tespit edilen kutuların koordinatları
    track_ids = results[0].boxes.id.int().cpu().tolist()  # İzlenen nesnelerin kimlikleri
    vehicle_types = results[0].boxes.cls.numpy()  # Tespit edilen nesnelerin sınıfları
    names = results[0].names  # Sınıf isimleri

    # Araç sayısını bul
    vehicle_count = 0

    # Annotator'ı başlat
    annotator = Annotator(frame, line_width=2, font_size=10)

    # İzleme çizgilerini güncelle ve Annotator ile kutuları çiz
    for box, track_id, vehicle_type in zip(boxes, track_ids, vehicle_types):
        class_label = names[vehicle_type]  # class name

        if class_label in allowed_classes:
                
            x, y, w, h = box
            
            # Kutunun sol üst köşesini hesapla
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            # Seçilen alan içindeyse
            if roi:
                roi_x, roi_y, roi_w, roi_h = roi
                if roi_x <= x1 <= roi_x + roi_w and roi_y <= y1 <= roi_y + roi_h:
                    # İzleme bilgilerini güncelle
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y merkez noktaları

                    # 30 frame'den fazla izlenen izleri sil (30 frame tutacak şekilde)
                    if len(track) > 30:
                        track.pop(0)

                    # Sınıf için renk seçimi
                    color = class_colors.get(class_label, (255, 255, 255))  # Varsayılan renk beyaz

                    # Annotator ile tespit edilen kutuları ve kimlikleri çiz
                    annotator.box_label([x1, y1, x2, y2], f'{class_label} ID: {track_id}', color=color)

    # Annotated frame'i al
    annotated_frame = annotator.result()

    # Annotated frame'i göster
    cv2.imshow("YOLO Detection", annotated_frame)

    # 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Video akışını serbest bırak ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()
