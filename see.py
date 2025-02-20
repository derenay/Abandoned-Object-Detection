from ultralytics import YOLO
import cv2
import yt_dlp
import time
from collections import defaultdict
import math

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

# Her track id için geçmiş koordinatları saklıyoruz
track_history = defaultdict(lambda: [])

# Sadece takip edeceğimiz hedef sınıflar
target_classes = {"backpack", "suitcase", "handbag"}

# Nesnenin sabit kalma zamanını (timer) tutan sözlük: track_id -> başlangıç zamanı
static_timer = dict()
# Şu anki terk edilmiş (abandoned) track id'leri
abandoned_objects_list = set()

# Hafıza: terk edilmiş nesneleri lokasyon bazında saklayıp, occlusion sonrası hatırlamak için
# Her eleman {'center': (x,y), 'class': class_label, 'last_seen': zaman}
abandoned_memory = []

# Parametreler
MOVE_THRESHOLD = 5             # Piksel cinsinden; bu değerin altında hareket statik kabul edilir
TIME_THRESHOLD = 20            # Saniye cinsinden; 20 saniye boyunca hareketsiz kalırsa abandoned
MEMORY_DISTANCE_THRESHOLD = 10 # Piksel; hafızadaki konum ile yeni tespit arasındaki mesafe
MEMORY_MAX_AGE = 180            # Hafızada tutulma süresi (saniye), bu süreden eski memory temizlenecek

def euclidean_distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video bitti")
        break

    # Hafıza temizleme: eski kayıtlar kaldırılıyor
    current_time = time.time()
    abandoned_memory = [mem for mem in abandoned_memory if current_time - mem['last_seen'] < MEMORY_MAX_AGE]

    
    results = model.track(frame, persist=True, tracker="botsort.yaml")
    
    boxes = results[0].boxes.xywh.cpu().numpy()
    track_ids = results[0].boxes.id.int().cpu().tolist()
    types = results[0].boxes.cls.cpu().numpy()
    names = results[0].names
    
    for (box, track_id, cls) in zip(boxes, track_ids, types):
        class_label = names[int(cls)]
        
        # Bounding box hesaplama (center, width, height formatından x1,y1,x2,y2'ye)
        x, y, w, h = box            
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        
        # Şu anki nesnenin merkezi
        current_center = (float(x), float(y))
        
        # Geçmiş koordinatlar üzerinden, son konumu alıp hareket mesafesi hesapla
        if track_history[track_id]:
            prev_center = track_history[track_id][-1]
            distance = euclidean_distance(current_center, prev_center)
        else:
            distance = 0
        
        # Geçmişe ekle
        track_history[track_id].append(current_center)
        
        # Sadece hedef sınıflar için statik kontrol ve hafıza kontrolü
        if class_label in target_classes:
            # Önce, hafıza kontrolü: Eğer yakın bir abandoned_memory kaydı varsa, direkt abandoned olarak işaretle
            near_memory = False
            for mem in abandoned_memory:
                if mem['class'] == class_label and euclidean_distance(current_center, mem['center']) < MEMORY_DISTANCE_THRESHOLD:
                    near_memory = True
                    # Hafızayı güncelle
                    mem['center'] = current_center
                    mem['last_seen'] = current_time
                    break
                    
            if near_memory:
                abandoned_objects_list.add(track_id)
                # Timer kullanmadan abandoned olarak işaretle
            else:
                # Timer tabanlı kontrol
                if distance < MOVE_THRESHOLD:
                    # Nesne yeterince az hareket ediyorsa timer başlat veya devam ettir
                    if track_id not in static_timer:
                        static_timer[track_id] = current_time
                    else:
                        elapsed = current_time - static_timer[track_id]
                        if elapsed >= TIME_THRESHOLD:
                            abandoned_objects_list.add(track_id)
                            # Hafızaya ekle veya güncelle
                            found = False
                            for mem in abandoned_memory:
                                if mem['class'] == class_label and euclidean_distance(current_center, mem['center']) < MEMORY_DISTANCE_THRESHOLD:
                                    mem['center'] = current_center
                                    mem['last_seen'] = current_time
                                    found = True
                                    break
                            if not found:
                                abandoned_memory.append({
                                    'center': current_center,
                                    'class': class_label,
                                    'last_seen': current_time
                                })
                else:
                    # Nesne hareket ediyorsa; timer sıfırlanır ve abandoned işareti kaldırılır
                    if track_id in static_timer:
                        static_timer.pop(track_id)
                    if track_id in abandoned_objects_list:
                        abandoned_objects_list.remove(track_id)
        else:
            # Hedef sınıf dışında ise, timer ve abandoned durumu resetlenir
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
