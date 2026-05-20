# clusterer.py
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
from config import Config

class SpatialClusterer:
    def __init__(self, eps=Config.GROUP_EPS, min_samples=1):
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    def cluster(self, items_dict):
        """Clusters items based on their center (x, y) coordinates."""
        if len(items_dict) < 1:
            return {}

        coords = []
        ids = []

        for item_id, data in items_dict.items():
            coords.append([data['x'], data['y']])
            ids.append(item_id)

        labels = self.dbscan.fit_predict(np.array(coords))

        groups = defaultdict(list)
        for i, label in enumerate(labels):
            if label != -1:  # -1 represents noise in DBSCAN
                groups[label].append(ids[i])

        return groups