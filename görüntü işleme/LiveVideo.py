from datetime import timedelta
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from collections import defaultdict

# Canlı yayın veya video dosyası URL'si
video_url = 'https://content.tvkur.com/l/c77i5e384cnrb6mlji10/master.m3u8'  # Örneğin: 'https://content.tvkur.com/l/c77i6qj84cnrb6mlji5g/master.m3u8' veya 'video.mp4'

# Video akışını aç
cap = cv2.VideoCapture(video_url)

# Pretrained YOLO modelini yükle
model = YOLO(r"C:\Users\erena\Desktop\Yolo\carDetectionTrain\model\yolo11m-seg.pt").cuda()

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

# Toplam araç sayısını takip etmek için bir set (küme) oluştur
unique_track_ids = set()

# OpenCV ile pencere oluştur
cv2.namedWindow('YOLO Detection', cv2.WINDOW_NORMAL)

# Video üzerinden frame'leri oku ve YOLO ile işlem yap
while True:
    # Frame'i oku
    ret, frame = cap.read()
    if not ret:
        print("Video akışı sona erdi veya açılamadı.")
        break
    
    # YOLO modelini frame üzerinde çalıştır
    results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")

    # Eğer sonuçlar geçerli değilse, devam et
    if not results or results[0] is None or results[0].boxes.id is None:
        print("Bu frame'de tespit edilen nesne yok.")
    else:
        # Tespit edilen kutuları ve izleme kimliklerini al
        boxes = results[0].boxes.xywh.cpu()  # Tespit edilen kutuların koordinatları
        track_ids = results[0].boxes.id.int().cpu().tolist()  # İzlenen nesnelerin kimlikleri
        vehicle_types = results[0].boxes.cls.numpy()  # Tespit edilen nesnelerin sınıfları
        names = results[0].names  # Sınıf isimleri
        
        # Annotator'ı başlat
        annotator = Annotator(frame, line_width=2, font_size=10)

        # İzleme çizgilerini güncelle ve Annotator ile kutuları çiz
        for box, track_id, vehicle_type in zip(boxes, track_ids, vehicle_types):
            class_label = names[vehicle_type]  # Sınıf adı

            if class_label in allowed_classes:
                # Yeni track_id'yi kontrol et ve ekle
                if track_id not in unique_track_ids:
                    unique_track_ids.add(track_id)

                x, y, w, h = box
                
                # Kutunun sol üst köşesini hesapla
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)

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

    # Ekrana toplam geçen araç sayısını yazdır
    total_vehicles_text = f"Total amount of vehicles: {len(unique_track_ids)}"
    cv2.putText(annotated_frame, total_vehicles_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Annotated frame'i göster
    cv2.imshow("YOLO Detection", annotated_frame)

    # 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Video akışını serbest bırak ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()
