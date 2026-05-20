# config.py

class Config:
    # Model & Classes
    YOLO_MODEL = "yolov8l.pt"
    PERSON_CLASS = 0
    BAG_CLASSES = [24, 26, 28]  # backpack, handbag, suitcase

    # Thresholds
    GROUP_EPS = 150                 # DBSCAN radius for grouping (pixels)
    ASSOCIATION_DIST_THRESH = 200   # Max dist to associate bag to person (ownership)
    NEAR_GROUP_RADIUS = 150         # Dist to consider a group "attending" a bag
    ABANDON_TIME_THRESH = 3         # Secs a bag must be left alone to trigger alert
    STATIONARY_SPEED_THRESH = 3     # Max pixels/frame to be considered "stationary"
    STATIONARY_TIME_THRESH = 3      # Secs to consider person/bag "stationary"
    HISTORY_FRAMES = 30             # Number of frames to calculate speed (e.g., 1 sec at 30fps)

    # Ownership Rules
    MAX_BAGS_PER_PERSON = 3         # Max bags one person can own
    ASSOCIATION_TIME_THRESH = 3     # Secs a person must be near a bag to claim it