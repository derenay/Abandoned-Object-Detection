from datetime import timedelta
import cv2
import torch
import numpy as np
from collections import defaultdict
from cap_from_youtube import cap_from_youtube
from ultralytics.utils.plotting import Annotator  # Çizim için kullanılmaya devam ediliyor

# SORT izleyicisini içe aktarın (pip install sort-python veya ilgili paketi kullanın)
from sort import Sort

# YouTube videosunu al
youtube_url = 'https://youtu.be/KBsqQez-O4w'
start_time = timedelta(seconds=5)  # Video başlangıç zamanı
cap = cap_from_youtube(youtube_url, 'best', start=start_time)
cap.set(cv2.CAP_PROP_FPS, 30)

# Pretrained RT-DETR modelini yükle (modelin API'sı, çıktı formatı ve yükleme yöntemi farklı olabilir)
# Aşağıdaki satır örnek olup, kendi modelinize göre düzenlemeniz gerekebilir.
model = torch.hub.load('your_rt_detr_repo', 'rt_detr', pretrained=True)
model.cuda()
model.eval()

allowed_classes = ['car', 'bus', 'motorcycle', 'truck']

# Her sınıf için renk tanımlamaları
class_colors = {
    'car': (255, 0, 0),         # Kırmızı
    'bus': (0, 255, 0),         # Yeşil
    'motorcycle': (0, 0, 255),  # Mavi
    'truck': (0, 255, 255)      # Sarı
}

# SORT izleyicisini başlat
tracker = Sort()

# İzleme geçmişini ve benzersiz izleme ID'lerini tutmak için
track_history = defaultdict(lambda: [])
unique_track_ids = set()

# OpenCV penceresi oluştur
cv2.namedWindow('RT-DETR Detection', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # RT-DETR için ön işleme (örneğin, BGR -> RGB dönüşümü, boyutlandırma vs.)
    # Burada, modelin beklentisine göre dönüştürme yapılmalıdır.
    input_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).cuda()  # (1, 3, H, W)

    with torch.no_grad():
        results = model(input_tensor)
    # Beklenen çıktı: results bir sözlük; örn.:
    # results = {
    #     'boxes': tensor([...]),  # [N, 4] formatında [x1, y1, x2, y2]
    #     'scores': tensor([...]),
    #     'labels': tensor([...]),
    #     'names': {0: 'person', 1: 'car', ...}  # sınıf indekslerinden isimlere dönüşüm
    # }
    boxes = results['boxes'].cpu().numpy()      # [N, 4]
    scores = results['scores'].cpu().numpy()      # [N]
    labels = results['labels'].cpu().numpy()      # [N]
    names = results['names']                      # Sınıf isimleri sözlüğü

    # Tespit edilen nesnelerden sadece allowed_classes listesindeki nesneleri seçiyoruz.
    detections = []
    detection_classes = []  # Her tespitin sınıf bilgisini saklamak için
    for box, score, label in zip(boxes, scores, labels):
        class_name = names[label]
        if class_name in allowed_classes:
            x1, y1, x2, y2 = box
            detections.append([x1, y1, x2, y2, score])
            detection_classes.append(class_name)
    detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))

    # SORT izleyicisini güncelle (detections formatı: [x1, y1, x2, y2, score])
    # Çıktı: her izlenen nesne için [x1, y1, x2, y2, track_id]
    tracks = tracker.update(detections)

    # Annotator başlat
    annotator = Annotator(frame, line_width=2, font_size=10)

    # İzlenen her nesne için; burada izleyici sınıf bilgisi sağlamadığı için,
    # her track için tespit edilen kutular arasında merkez mesafesi en yakın olanın sınıfını atıyoruz.
    for trk in tracks:
        x1, y1, x2, y2, track_id = trk
        # Track merkezini hesapla
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        assigned_class = None
        min_dist = float('inf')
        for det, cls in zip(detections, detection_classes):
            bx1, by1, bx2, by2, score = det
            bcx = (bx1 + bx2) / 2
            bcy = (by1 + by2) / 2
            dist = np.sqrt((cx - bcx) ** 2 + (cy - bcy) ** 2)
            if dist < min_dist:
                min_dist = dist
                assigned_class = cls

        # İzleme geçmişini güncelle
        track_history[track_id].append((cx, cy))
        if len(track_history[track_id]) > 30:
            track_history[track_id].pop(0)

        unique_track_ids.add(track_id)

        # Kutuyu ve ID bilgisini çiz
        color = class_colors.get(assigned_class, (255, 255, 255))
        annotator.box_label([int(x1), int(y1), int(x2), int(y2)],
                            f'{assigned_class} ID: {int(track_id)}',
                            color=color)

    # Toplam araç sayısını ekrana yazdır
    total_vehicles_text = f"Total amount of vehicles: {len(unique_track_ids)}"
    cv2.putText(frame, total_vehicles_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    annotated_frame = annotator.result()
    cv2.imshow("RT-DETR Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
