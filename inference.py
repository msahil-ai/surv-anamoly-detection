import cv2
from cloud_memory_buffer import CloudStreamBuffer
import time
from collections import deque
from spatio_temp_clustering import CloudCrowdClusterer
from ultralytics import YOLO



def main():
    # 1. Initialize Heavyweight Models on GPU
    print("Loading Heavyweight YOLO model into GPU...")
    # Use YOLOv8-large or YOLOv11-large for maximum accuracy on cloud
    model = YOLO("yolov8n.pt") 
    
    # 2. Start Video Ingestion
    stream_url = "sample.mp4" # Replace with RTSP URL
    stream = CloudStreamBuffer(stream_url, buffer_seconds=10, fps=30).start()
    clusterer = CloudCrowdClusterer()
    
    # Tracking state dictionaries
    historical_trajectories = {} # track_id -> list of (x, y) over time
    static_bags = {} # bag_id -> time_stationary
    
    print("Inference Pipeline Started...")
    time.sleep(2) # Allow buffer to pre-fill

    while not stream.stopped:
        frame = stream.read_live()
        if frame is None:
            continue

        # ====================================================
        # A. HEAVY INFERENCE & BoT-SORT TRACKING
        # ====================================================
        # persist=True keeps tracking IDs across frames.
        # tracker="botsort.yaml" utilizes BoT-SORT for Re-ID and robust tracking (when in cloud)
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0, 24, 26, 28], verbose=False)
        # Classes: 0 (person), 24 (backpack), 26 (handbag), 28 (suitcase)

        current_persons = {}
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, classes):
                x, y, w, h = box
                
                # If it's a person, calculate velocity for clustering
                if cls == 0: 
                    if track_id not in historical_trajectories:
                        historical_trajectories[track_id] = deque(maxlen=30) # Store last 1 sec of positions
                    
                    historical_trajectories[track_id].append((x, y))
                    
                    # Calculate velocity if we have enough history
                    vx, vy = 0.0, 0.0
                    history = historical_trajectories[track_id]
                    if len(history) > 5:
                        vx = (history[-1][0] - history[0][0]) / len(history)
                        vy = (history[-1][1] - history[0][1]) / len(history)
                        
                    current_persons[track_id] = {'x': x, 'y': y, 'vx': vx, 'vy': vy}
                    
                # If it's luggage, check for static state
                elif cls in [24, 26, 28]:
                    # In a full implementation, you would check if 'x' and 'y' variance
                    # over the last 30 frames is below a movement threshold here.
                    pass 

        # ====================================================
        # B. DYNAMIC GROUP CLUSTERING
        # ====================================================
        active_groups = clusterer.cluster_groups(current_persons)
        
        # ====================================================
        # C. EVENT LOGIC (Pseudo-implementation)
        # ====================================================
        # If a bag triggers the "Static" state:
            # 1. Fetch historical_clip = stream.get_historical_clip()
            # 2. Identify the track_id of the person who dropped it.
            # 3. Check 'active_groups' to see if that person belongs to a family.
            # 4. Bind the bag's safe-zone logic to ALL members of that group.
            # 5. Start the 5-minute timer.

        # For visualization purposes only (Remove cv2.imshow in headless cloud deployments)
        annotated_frame = results[0].plot()
        cv2.imshow("Cloud GPU Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()