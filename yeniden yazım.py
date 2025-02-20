from ultralytics import YOLO
import cv2
import yt_dlp

from collections import defaultdict

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

track_history = defaultdict(lambda: [])

unique_track_ids = set()

abanoded_objects_list = set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video bitti")
    
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    
    
    boxes = results[0].boxes.xywh.cpu().numpy()
    track_ids = results[0].boxes.id.int().cpu().tolist()
    types = results[0].boxes.cls.cpu().numpy()
    names = results[0].names
    
    
    for (box, track_id, type) in zip(boxes, track_ids, types):
        
        class_label = names[type] 
        
        if track_id not in unique_track_ids:
            unique_track_ids.add(track_id)
        
        x, y, w, h = box
            
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        
        track = track_history[track_id]
        track.append((float(x), float(y)))
        
        
        
    
        
        label = f"{class_label}  id:{track_id}"
        
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 0)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    
        cv2.imshow("Yolo 11 Youtube detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()



def abanoded_objects(int:id) -> None:
    abanoded_objects_list.add(id)




