from datetime import timedelta
import cv2
from cap_from_youtube import cap_from_youtube
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

# YouTube videosunu al
youtube_url = 'https://youtu.be/MNn9qKG2UFI'
start_time = timedelta(seconds=5)  # YouTube videosunun başlangıç zamanı
cap = cap_from_youtube(youtube_url, 'best', start=start_time)
cap.set(cv2.CAP_PROP_FPS, 30)

# Pretrained YOLO modelini yükle
model = YOLO(r"C:\Users\erena\Desktop\Yolo\carDetectionTrain\model\best1.pt").cuda()

# İzleme geçmişini tutmak için bir defaultdict oluştur
track_history = defaultdict(lambda: [])

# OpenCV ile pencere oluştur
cv2.namedWindow('YOLO Detection', cv2.WINDOW_NORMAL)

# Video üzerinden frame'leri oku ve YOLO ile işlem yap
while True:
    # Frame'i oku
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLO modelini frame üzerinde çalıştır
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    # Tespit edilen kutuları ve izleme kimliklerini al
    boxes = results[0].boxes.xywh.cpu()  # Tespit edilen kutuların koordinatları
    track_ids = results[0].boxes.id.int().cpu().tolist()  # İzlenen nesnelerin kimlikleri

    # Frame üzerine tespit edilen nesneleri ve izleme bilgilerini çiz
    annotated_frame = results[0].plot()

    # İzleme çizgilerini çiz
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))  # x, y merkez noktaları

        # 60 frame'den fazla izlenen izleri sil (60 frame tutacak şekilde)
        if len(track) > 30:
            track.pop(0)

        # İzleme çizgilerini çizmek için kullanılabilir (isteğe bağlı)
        # points = np.array(track, np.int32).reshape((-1, 1, 2))
        # cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

    # Annotated frame'i göster
    cv2.imshow("YOLO Detection", annotated_frame)

    # 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Video akışını serbest bırak ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()