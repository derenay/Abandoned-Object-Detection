from datetime import timedelta
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from collections import defaultdict

# Canlı yayın veya video dosyası URL'si
video_url = 'https://content.tvkur.com/l/c77i5e384cnrb6mlji10/master.m3u8'  # Örneğin: canlı yayın URL'si veya 'video.mp4'

# YOLO modelini yükle
model = YOLO(r"C:\Users\erena\Desktop\Yolo\carDetectionTrain\model\yolo11m-seg.pt").cuda()
allowed_classes = ['car', 'bus', 'motorcycle', 'truck']

# Video akışını aç
cap = cv2.VideoCapture(video_url)

# Her şerit için poligonlar
lane1_polygon = np.array([(128, 993), (933, 991), (855, 362), (641, 348)], np.int32)  # Üstteki gelen şerit
lane2_polygon = np.array([(935, 988), (1824, 955), (1054, 364), (853, 362)], np.int32)  # Üstteki giden şerit
    # Yan gelen sağ

# Şeritler için araç sayımı
lane_counts = [0, 0, 0, 0]  # Lane1, Lane2, Lane3, Lane4 için sayaçlar

# İzlenen araç kimliklerini ayrı tutmak için listeler
unique_track_ids = [set(), set(), set(), set()]

# OpenCV ile pencere oluştur
cv2.namedWindow('YOLO Detection', cv2.WINDOW_NORMAL)

# Video üzerinden frame'leri oku ve YOLO ile işlem yap
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video akışı sona erdi veya açılamadı.")
        break

    # Şerit poligonlarını çizin
    cv2.polylines(frame, [lane1_polygon], isClosed=True, color=(255, 0, 0), thickness=2)  # Mavi
    cv2.polylines(frame, [lane2_polygon], isClosed=True, color=(0, 255, 0), thickness=2)  # Yeşil
    

    # YOLO modelini frame üzerinde çalıştır
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    # Eğer sonuçlar geçerli değilse, devam et
    if not results or results[0] is None or results[0].boxes.id is None:
        continue

    # Tespit edilen kutuları ve izleme kimliklerini al
    boxes = results[0].boxes.xywh.cpu()  # Tespit edilen kutuların koordinatları
    track_ids = results[0].boxes.id.int().cpu().tolist()  # İzlenen nesnelerin kimlikleri
    vehicle_types = results[0].boxes.cls.numpy()  # Tespit edilen nesnelerin sınıfları
    names = results[0].names  # Sınıf isimleri

    # Annotator'ı başlat
    annotator = Annotator(frame, line_width=2, font_size=10)

    # Şeritler için araç tespiti
    for box, track_id, vehicle_type in zip(boxes, track_ids, vehicle_types):
        class_label = names[vehicle_type]  # Sınıf adı

        if class_label in allowed_classes:
            x, y, w, h = box
            center_x, center_y = int(x), int(y)  # Araç merkez noktası

            # Şeritlerde kontrol yap
            for i, (lane_polygon, unique_ids) in enumerate(zip(
                [lane1_polygon, lane2_polygon],
                unique_track_ids
            )):
                if cv2.pointPolygonTest(lane_polygon, (center_x, center_y), False) >= 0:
                    # Araç bu şeritte ve daha önce görülmemişse sayacı artır
                    if track_id not in unique_ids:
                        unique_ids.add(track_id)
                        lane_counts[i] += 1

                    # Annotator ile kutu çizimi
                    annotator.box_label(
                        [int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)],
                        f'{class_label} ID: {track_id}',
                        color=(255, 255, 255)
                    )

    # Annotated frame'i al
    annotated_frame = annotator.result()

    # Şerit sayacı yazdır
    lane_texts = [
        f"Lane1: {lane_counts[0]}",
        f"Lane2: {lane_counts[1]}",
        f"Lane3: {lane_counts[2]}",
        f"Lane4: {lane_counts[3]}"
    ]
    for i, text in enumerate(lane_texts):
        cv2.putText(annotated_frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Annotated frame'i göster
    cv2.imshow("YOLO Detection", annotated_frame)

    # 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Video akışını serbest bırak ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()

# Şerit sonuçlarını yazdır
print(f"Lane1 araç sayısı: {lane_counts[0]}")
print(f"Lane2 araç sayısı: {lane_counts[1]}")
print(f"Lane3 araç sayısı: {lane_counts[2]}")
print(f"Lane4 araç sayısı: {lane_counts[3]}")
