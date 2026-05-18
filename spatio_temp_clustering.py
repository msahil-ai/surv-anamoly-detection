import numpy as np
import cv2
import threading
import time
from collections import deque
from sklearn.cluster import DBSCAN

class CloudCrowdClusterer:
    def __init__(self, spatial_weight=1.0, velocity_weight=2.5):
        self.spatial_weight = spatial_weight
        self.velocity_weight = velocity_weight
        # eps and min_samples need tuning based on camera focal length and angle
        self.dbscan = DBSCAN(eps=2.0, min_samples=2)

    def cluster_groups(self, tracked_persons):
        """
        tracked_persons: dict mapping track_id -> {'x': float, 'y': float, 'vx': float, 'vy': float}
        """
        if len(tracked_persons) < 2:
            return {}

        features = []
        person_ids = []

        for p_id, data in tracked_persons.items():
            feature_vector = [
                data['x'] * self.spatial_weight, 
                data['y'] * self.spatial_weight, 
                data['vx'] * self.velocity_weight, 
                data['vy'] * self.velocity_weight
            ]
            features.append(feature_vector)
            person_ids.append(p_id)

        features_np = np.array(features)
        labels = self.dbscan.fit_predict(features_np)

        groups = {}
        for idx, label in enumerate(labels):
            if label != -1: # -1 indicates noise (a solo traveler)
                if label not in groups:
                    groups[label] = []
                groups[label].append(person_ids[idx])

        return groups