from ultralytics import YOLO
import cv2
import yt_dlp
import time
from collections import defaultdict

# Youtube video URL'si
youtube_url = 'https://youtu.be/jct8UYI54-0'

ydt_opts = {
    'format': 'best[ext=mp4]',
    'quiet': True
}

with yt_dlp.YoutubeDL(ydt_opts) as ydt:
    info_dict = ydt.extract_info(youtube_url, download=False)
    video_url = info_dict['url']
 
cap = cv2.VideoCapture(video_url)

model = YOLO('yolo11m.pt').cuda() 

# Her track id için geçmiş koordinatları tutan sözlük
track_history = defaultdict(lambda: [])
# Hedeflenen nesne sınıfları
target_classes = {"backpack", "suitcase", "handbag"}

# Her nesnenin statik kalmaya başladığı zamanı kaydedeceğimiz sözlük
static_timer = dict()
# Terk edilmiş (abandoned) nesneleri tutan set
abandoned_objects_list = set()

# Hareket eşik değeri (piksel cinsinden)
MOVE_THRESHOLD = 5
# Sabit kalma süresi eşik değeri (saniye cinsinden)
TIME_THRESHOLD = 20

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video bitti")
        break
    
    # BoT-SORT tracker için 'botsort.yaml' kullanılarak YOLO tespiti
    results = model.track(frame, persist=True, tracker="botsort.yaml")
    
    boxes = results[0].boxes.xywh.cpu().numpy()
    track_ids = results[0].boxes.id.int().cpu().tolist()
    types = results[0].boxes.cls.cpu().numpy()
    names = results[0].names
    
    for (box, track_id, cls) in zip(boxes, track_ids, types):
        class_label = names[int(cls)]
        
        # Bounding box hesaplama
        x, y, w, h = box            
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        
        # Mevcut koordinatı al
        current_center = (float(x), float(y))
        # Geçmiş koordinat listesinden son koordinatı al
        if track_history[track_id]:
            prev_center = track_history[track_id][-1]
            distance = ((current_center[0] - prev_center[0])**2 + (current_center[1] - prev_center[1])**2) ** 0.5
        else:
            distance = 0
        
        # Geçmişe ekle
        track_history[track_id].append(current_center)
        
        # Hedef sınıflardan ise statik kontrolünü yap
        if class_label in target_classes:
            if distance < MOVE_THRESHOLD:
                # Nesne yeterince az hareket ediyorsa, timer başlat veya devam ettir
                if track_id not in static_timer:
                    static_timer[track_id] = time.time()
                else:
                    elapsed = time.time() - static_timer[track_id]
                    if elapsed >= TIME_THRESHOLD:
                        abandoned_objects_list.add(track_id)
            else:
                # Nesne hareket ediyorsa; timer sıfırlanır ve abandoned listesinden çıkarılır
                if track_id in static_timer:
                    static_timer.pop(track_id)
                if track_id in abandoned_objects_list:
                    abandoned_objects_list.remove(track_id)
        else:
            # Diğer sınıflar için isteğe bağlı: timer reset
            if track_id in static_timer:
                static_timer.pop(track_id)
            if track_id in abandoned_objects_list:
                abandoned_objects_list.remove(track_id)
        
        # Label ve renk belirleme: terk edilmiş nesneler kırmızı, normal olanlar yeşil
        if track_id in abandoned_objects_list:
            label = f"{class_label} abandoned id:{track_id}"
            color = (0, 0, 255)
        else:
            label = f"{class_label} id:{track_id}"
            color = (0, 255, 0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imshow("Yolo 11 Youtube detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

def abanoded_objects(id: int) -> None:
    abandoned_objects_list.add(id)
