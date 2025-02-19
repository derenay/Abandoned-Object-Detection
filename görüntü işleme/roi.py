import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Video kaynağı (Canlı yayın veya video dosyası URL'si)
video_url = 'https://content.tvkur.com/l/c77i6cfbb2nj4i0fr7s0/master.m3u8'

# YOLO modelini yükle
model = YOLO(r"C:\Users\erena\Desktop\Yolo\carDetectionTrain\model\yolo11m-seg.pt").cuda()
allowed_classes = ['car', 'bus', 'motorcycle', 'truck']

# Video akışını aç
cap = cv2.VideoCapture(video_url)

# Nokta seçiminde kullanılan değişkenler
points = []  # Seçilen noktalar
polygon_done = False  # Poligon çizimi tamamlandı mı

# Fare olaylarını işleme fonksiyonu
def select_points(event, x, y, flags, param):
    global points, polygon_done
    if event == cv2.EVENT_LBUTTONDOWN:  # Sol tık ile nokta ekleme
        points.append((x, y))
        print(f"Nokta eklendi: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:  # Sağ tık ile poligon çizimini tamamla
        polygon_done = True
        print("Poligon tamamlandı.")

# OpenCV ile pencere oluştur ve fare olaylarını bağla
cv2.namedWindow('YOLO Detection', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('YOLO Detection', select_points)

# Video üzerinden frame'leri oku ve YOLO ile işlem yap
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video akışı sona erdi veya açılamadı.")
        break

    # Seçilen noktaları çizin
    for point in points:
        cv2.circle(frame, point, 5, (0, 255, 0), -1)  # Yeşil noktalar

    # Eğer poligon tamamlandıysa, poligonu çizin
    if len(points) > 1 and polygon_done:
        polygon = np.array(points, np.int32)
        cv2.polylines(frame, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)  # Mavi poligon

    # YOLO modelini çalıştır
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

    # Eğer sonuçlar geçerli değilse, devam et
    if not results or results[0] is None or results[0].boxes.id is None:
        print("Bu frame'de tespit edilen nesne yok.")
    else:
        # Annotator'ı başlat
        annotator = Annotator(frame, line_width=2, font_size=10)
        
        # Tespit edilen kutuları ve kimlikleri al
        boxes = results[0].boxes.xywh.cpu()  # Tespit edilen kutuların koordinatları
        track_ids = results[0].boxes.id.int().cpu().tolist()  # İzlenen nesnelerin kimlikleri
        vehicle_types = results[0].boxes.cls.numpy()  # Tespit edilen nesnelerin sınıfları
        names = results[0].names  # Sınıf isimleri
        
        for box, track_id, vehicle_type in zip(boxes, track_ids, vehicle_types):
            class_label = names[vehicle_type]  # Sınıf adı
            
            if class_label in allowed_classes:
                x, y, w, h = box
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)

                # Eğer poligon tamamlandıysa, ROI içinde olup olmadığını kontrol et
                if polygon_done:
                    roi_polygon = np.array(points, np.int32)
                    center_x, center_y = int(x), int(y)  # Araç kutusunun merkezi
                    if cv2.pointPolygonTest(roi_polygon, (center_x, center_y), False) >= 0:
                        # ROI içinde olan araçlar
                        annotator.box_label([x1, y1, x2, y2], f'{class_label} ID: {track_id}', color=(0, 255, 255))  # Sarı kutular

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
