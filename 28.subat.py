from ultralytics import YOLO
import cv2
import yt_dlp
import time
import math


class AbandonedDetection():
    def __init__(
        self,
        model,
        general_classes:set,
        target_classes:set,
        video_url:str
        )-> None: 
        """sum

        Args:
            model (Yolo Model): It needs Yolo model
            general_classes (set): general detection classes
            target_classes (set): abandoned object classes
            video_url (str): Video url must be youtube video or live video(must be cv2 supported)
        """
        
        self.static_candidates:list = []  #{"initial_center": (x,y), "last_center": (x,y), "class": sınıf, "start_time": t, "last_seen": t, "abandoned": bool}
        self.general_classes = general_classes
        self.target_classes = target_classes
        self.model = model
        self.cap = self.start_live_or_mp4_video(video_url)
        self.persons_detected = [] #[[track_id,[x,y]],[track_id,[x,y]]]
        self.cand = []
        
    
    
    
    def euclidean_distance(self, p1:int, p2:int) -> float:
        """ Returns distance between two diffirent point 

        Args:
            p1 (int): _description_
            p2 (int): _description_

        Returns:
            float: Returns distance between two diffirent point 
        """
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])


    def find_candidate(self,candidates:list, cls:str, center)->int|None:
        # To do buraya id ile arama da eklenmeli ki kişiyi objeye bağladığımızda id ile işlem yapabilelim
        for cand in candidates:
            if self.euclidean_distance(cand["last_center"], center) < MATCH_THRESHOLD:
                return cand
        return None


    def find_closest_person(self, object:tuple ,MAX_DISTANCE=60) -> list: 
        """ Return closest person to detected object 

        Args:
            object (tuple): _description_

        Returns:
            list: Return closest person =[track_id, distance, center_point]
        """
        a = []
        try:
            for person_detected in self.persons_detected:
                a.append([person_detected[0], self.euclidean_distance(person_detected[1], object), person_detected[1]])
                #print(f"{object} {person_center[1]}")
                
            closest_person = min(a, key=lambda sublist: sublist[1]) # en yakın kişiyi buluan ve tüm kişi bilgisini(liste) veren yer
            
            if closest_person[1] < MAX_DISTANCE:
                return closest_person
            
            return None
            
        except Exception as e:
            print(e)
            
            
    def find_connected_person_object_distance(self, object:list) : 
 
        a = []
        try:
            for person_detected in self.persons_detected:
                if person_detected[0] == object[0]:
                    return person_detected[1]
            
        except Exception as e:
            print(e)
            
        
        
    def start_live_or_mp4_video(self, video_url):
        """video to cv2 VideoCapture

        Args:
            video_url (str): Must be live video or youtube video

        Returns:
            VideoCapture: returns cap
        """
        
        if video_url.find("youtu") == -1:
                
            return cv2.VideoCapture(video_url) 
        else:

            ydt_opts = {                                    
                'format': 'best[ext=mp4]',
                'quiet': True
            }

            with yt_dlp.YoutubeDL(ydt_opts) as ydt:
                info_dict = ydt.extract_info(video_url, download=False)
                video = info_dict['url']
                
            return cv2.VideoCapture(video)  

    def detection(
        self,
        MATCH_THRESHOLD:int,
        MOVE_THRESHOLD:int,
        TIME_THRESHOLD:int,
        CLEAR_THRESHOLD:int,
        PERSON_OBJECT_THRESHOLD:int
        ) -> None:
        """Starts whole abandoned detection process

        Args:
            MATCH_THRESHOLD (int): Distance (pixels) allowed for matching between the new detection and the candidate
            MOVE_THRESHOLD (int): Pixel; If the detected object moves away, it will be thrown from abandoned
            TIME_THRESHOLD (int): Seconds; If it remains inactive for more than this time, it will be marked as abandoned.
            CLEAR_THRESHOLD (int): Seconds; clear if candidate is not updated for a long time
            PERSON_OBJECT_THRESHOLD (int): Pixel; Specifies the distance between the person and the object, if they are within this distance, the two objects will never be abandoned.
        """
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        #frame counter
        frame_count = 0
        
        
        
        if fps < 30:
            skip_frame = 2
        elif 30 <= fps <= 60:
            skip_frame = 3
        else:
            skip_frame = 4
        print(f"skip frame: {skip_frame}\nfps: {fps}")
        
        frame_width = int(self.cap.get(3)) 
        frame_height = int(self.cap.get(4)) 
        size = (frame_width, frame_height) 
        
        save = cv2.VideoWriter('salam.mp4',  
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         fps, size) 
        
        #Frame detection
        while True:
     
            ret, frame = self.cap.read()
            if not ret:
                print("Video bitti")
                break


            current_time = time.time()
            
            #uzun süre güncellenmeyenleri listeden çıkar
            self.static_candidates = [cand for cand in self.static_candidates if current_time - cand["last_seen"] < CLEAR_THRESHOLD]
            
            if frame_count % skip_frame:
            
                results = self.model.track(frame, persist=True, tracker="botsort.yaml", verbose=False, conf=0.2)

                if not results or results[0] is None or results[0].boxes.id is None:
                    continue

                boxes = results[0].boxes.xywh.cpu().numpy()
                types = results[0].boxes.cls.cpu().numpy()
                names = results[0].names
                track_ids = results[0].boxes.id.int().cpu().tolist()  
            
                for (box, cls, track_id) in zip(boxes, types, track_ids):
                    class_label = names[int(cls)]
                    
                    if class_label in general_classes:
                    
                        x, y, w, h = box            
                        x1 = int(x - w / 2)
                        y1 = int(y - h / 2)
                        x2 = int(x + w / 2)
                        y2 = int(y + h / 2)
                        current_center = (float(x), float(y))

                        if class_label == "person" and (track_id in [person_center[0] for person_center in self.persons_detected]) is False:
                            self.persons_detected.append([track_id, current_center])
                        
                        
                        if class_label == "person" and (track_id in [person_center[0] for person_center in self.persons_detected]):
                            for person_center in self.persons_detected:
                                if person_center[0] == track_id:
                                    person_center[1] = current_center
                                            
                        abandoned = False
                        
                        if class_label in target_classes:
                
                            self.cand = self.find_candidate(self.static_candidates, class_label, current_center)
                            
                            if self.cand is not None:
                                
                                # Eğer candidate'nin ilk kaydedilen merkezi ile yeni tespit merkezi arasındaki fark MOVE_THRESHOLD'dan büyükse
                                if self.euclidean_distance(self.cand["initial_center"], current_center) > MOVE_THRESHOLD:
                                    self.static_candidates.remove(self.cand)
                                    self.cand["is_checked"] = False
                                else:
                                    self.cand["last_center"] = current_center
                                    self.cand["last_seen"] = current_time
                                    # Eğer belirlenen süreden (TIME_THRESHOLD) fazla hareketsiz kalmışsa abandoned olarak işaretle
                                    
                                    if self.cand["person"] is None:
                                        self.cand["person"] = self.find_closest_person(current_center) # bazen ilk obje tespit edilip sonrasında person tespit ediliyor eğer tespit edilemezse None geri dönüyor bunu düzeltmek için
                                        if self.cand['person'] is None:
                                            self.cand["abandoned"] = True
                                            abandoned = self.cand["abandoned"]
                                            
                                            
                                    if self.cand["person"] is not None:
                                        self.cand["person"][2] = self.find_connected_person_object_distance(self.cand["person"])
                                        self.cand["person"][1] = self.euclidean_distance(self.cand["last_center"], self.cand["person"][2])
                                        
                                        if self.cand["person"][1] > PERSON_OBJECT_THRESHOLD:
                                            if self.cand["is_checked"] == False:
                                                abondoned_start = current_time
                                                self.cand["is_checked"] = True
                                                
                                            if not self.cand["abandoned"] and (current_time - abondoned_start >= TIME_THRESHOLD):
                                                self.cand["is_checked"] = True
                                                self.cand["abandoned"] = True
                                            abandoned = self.cand["abandoned"]
                                        
                            else:
                                new_cand = {
                                    "track_id": track_id,
                                    "initial_center": current_center,
                                    "last_center": current_center,
                                    "class": class_label,
                                    "start_time": current_time,
                                    "last_seen": current_time,
                                    "abandoned": False,
                                    "is_checked": False,
                                    "person": self.find_closest_person(current_center) # burası en yakındaki insanları kaydetmesi lazım tek bir insanı değil belli bir dairedeki tüm insanları koyucaz
                                }
                                self.static_candidates.append(new_cand)
                        
                  
                        if class_label in target_classes and abandoned:
                            label = f"id: {self.cand['track_id']} {self.cand['class']} abandoned"
                            color = (0, 0, 255)  
                            
                        elif class_label == "person":
                            label = f"{class_label} Id:{track_id}"
                            color = (255,255,255)
                        else:
                            label = f"id: {track_id} {class_label}"
                            color = (0, 255, 0)  
                            
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                        #print(f"static candidates ----------{self.static_candidates} bitiş-------------------")

                        
            frame_count+=1
            # cv2.imshow("Yolo 11 Youtube detection", frame)
            
            save.write(frame) 


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    model = YOLO('yolo11m.pt').cuda()

    target_classes = {"backpack", "suitcase", "handbag"}
    general_classes = {"person","backpack", "suitcase", "handbag"}

    # https://youtu.be/fpTG4ELZ3bE https://youtu.be/jct8UYI54-0 https://youtu.be/ER0OdwuS6rg  https://youtu.be/3WNGZToVQGc
    # rtsp://admin:Admin123@10.4.0.112:554/Streaming/Channels/101
    
    youtube_url = 'https://youtu.be/fpTG4ELZ3bE'

    MATCH_THRESHOLD = 10         # Yeni tespit ile candidate arasında eşleşme için izin verilen mesafe (piksel)
    MOVE_THRESHOLD = 30          # Pixel;Tespit edilen nesne uzaklaşırsa abandoned ten atılır
    TIME_THRESHOLD = 5           # Saniye; bu süreden fazla hareketsiz kalırsa abandoned olarak işaretlenicek
    CLEAR_THRESHOLD = 5          # Saniye; candidate uzun süre güncellenmezse temizle
    PERSON_OBJECT_THRESHOLD = 90 # Pixel; Person ile object arasındaki mesafeyi belirtir eğer bu mesafe içindeyse iki nesne asla abandoned olmaz
    

    detection = AbandonedDetection(model=model, video_url=youtube_url, general_classes=general_classes, target_classes=target_classes)
    detection.detection(MATCH_THRESHOLD, MOVE_THRESHOLD, TIME_THRESHOLD, CLEAR_THRESHOLD, PERSON_OBJECT_THRESHOLD)

    # cv2.line(frame, (int(current_center[0]),int(current_center[1])), (int(a[2][0]),int(a[2][1])), color=(0,0,0))