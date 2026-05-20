# detector.py
import cv2
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO

from config import Config
from clusterer import SpatialClusterer
from visualizer import Visualizer

class BaggageDetectionSystem:
    def __init__(self, config=Config):
        self.config = config
        print("Loading YOLO Model...")
        self.model = YOLO(config.YOLO_MODEL)
        self.clusterer = SpatialClusterer(eps=config.GROUP_EPS)

        # State Tracking
        self.history = {'persons': defaultdict(lambda: deque(maxlen=config.HISTORY_FRAMES)),
                        'bags': defaultdict(lambda: deque(maxlen=config.HISTORY_FRAMES))}
        self.stationary_timers = {'persons': {}, 'bags': {}}

        # Association Tracking
        self.bag_owner = {}         
        self.bag_owner_group = {}   
        self.bag_unattended_timer = {} 

        # Trackers
        self.bag_id_map = {}               
        self.bag_potential_owners = {}     

    def _compute_speed(self, history_deque):
        if len(history_deque) < 5:
            return 0
        vx = history_deque[-1][0] - history_deque[0][0]
        vy = history_deque[-1][1] - history_deque[0][1]
        return np.sqrt(vx**2 + vy**2)

    def _dist(self, p1, p2):
        return np.linalg.norm(np.array([p1['x'], p1['y']]) - np.array([p2['x'], p2['y']]))

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        print("Pipeline running... Press Ctrl+C to stop in terminal.")

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = frame_count / fps

            # 1. Tracking
            results = self.model.track(frame, persist=True, tracker="botsort.yaml", conf=0.15,
                                       classes=[self.config.PERSON_CLASS] + self.config.BAG_CLASSES,
                                       verbose=False)

            tracked_persons = {}
            tracked_bags = {}

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes_xywh = results[0].boxes.xywh.cpu().numpy()
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                classes = results[0].boxes.cls.int().cpu().tolist()

                for xywh, xyxy, tid, cls in zip(boxes_xywh, boxes_xyxy, track_ids, classes):
                    x, y, _, _ = xywh
                    x1, y1, x2, y2 = map(int, xyxy)
                    data = {'x': x, 'y': y, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

                    if cls == self.config.PERSON_CLASS:
                        self.history['persons'][tid].append((x, y))
                        if tid not in self.stationary_timers['persons']:
                            self.stationary_timers['persons'][tid] = current_time

                        if self._compute_speed(self.history['persons'][tid]) >= self.config.STATIONARY_SPEED_THRESH:
                            self.stationary_timers['persons'][tid] = current_time
                            data['stationary'] = False
                        else:
                            data['stationary'] = (current_time - self.stationary_timers['persons'][tid]) >= self.config.STATIONARY_TIME_THRESH

                        tracked_persons[tid] = data

                    elif cls in self.config.BAG_CLASSES:
                        # Anti-Flicker ID Mapping
                        if tid in self.bag_id_map:
                            tid = self.bag_id_map[tid]
                        elif tid not in self.history['bags']:
                            for old_tid, old_hist in self.history['bags'].items():
                                if len(old_hist) > 0:
                                    old_x, old_y = old_hist[-1]
                                    if np.sqrt((x - old_x)**2 + (y - old_y)**2) < 80:
                                        self.bag_id_map[tid] = old_tid 
                                        tid = old_tid
                                        break

                        self.history['bags'][tid].append((x, y))
                        if tid not in self.stationary_timers['bags']:
                            self.stationary_timers['bags'][tid] = current_time

                        if self._compute_speed(self.history['bags'][tid]) >= self.config.STATIONARY_SPEED_THRESH:
                            self.stationary_timers['bags'][tid] = current_time
                            data['stationary'] = False
                        else:
                            data['stationary'] = (current_time - self.stationary_timers['bags'][tid]) >= self.config.STATIONARY_TIME_THRESH

                        tracked_bags[tid] = data

            # 2. Grouping
            groups = self.clusterer.cluster(tracked_persons)
            person_to_group = {pid: gid for gid, members in groups.items() for pid in members}

            # 3. Association & Behavior Check
            owner_bag_counts = defaultdict(int)
            for owner_id in self.bag_owner.values():
                owner_bag_counts[owner_id] += 1

            for bid, bag in tracked_bags.items():
                is_attended = False

                if bid not in self.bag_owner:
                    if bid not in self.bag_potential_owners:
                        self.bag_potential_owners[bid] = {}

                    current_potentials = {}
                    best_match = None
                    min_dist = float('inf')

                    for pid, person in tracked_persons.items():
                        dist = self._dist(person, bag)
                        if dist < self.config.ASSOCIATION_DIST_THRESH:
                            first_seen = self.bag_potential_owners[bid].get(pid, current_time)
                            current_potentials[pid] = first_seen

                            if (current_time - first_seen) >= self.config.ASSOCIATION_TIME_THRESH:
                                if owner_bag_counts[pid] < self.config.MAX_BAGS_PER_PERSON:
                                    if dist < min_dist:
                                        min_dist = dist
                                        best_match = pid

                    self.bag_potential_owners[bid] = current_potentials

                    if best_match is not None:
                        self.bag_owner[bid] = best_match
                        self.bag_owner_group[bid] = person_to_group.get(best_match, None)
                        is_attended = True
                        self.bag_potential_owners.pop(bid, None)

                else:
                    owner_id = self.bag_owner[bid]
                    owner_group = self.bag_owner_group[bid]

                    if owner_group is not None and owner_group in groups:
                        for group_member_id in groups[owner_group]:
                            if group_member_id in tracked_persons:
                                if self._dist(tracked_persons[group_member_id], bag) < self.config.NEAR_GROUP_RADIUS:
                                    is_attended = True
                                    break 
                    else:
                        if owner_id in tracked_persons:
                            if self._dist(tracked_persons[owner_id], bag) < self.config.NEAR_GROUP_RADIUS:
                                is_attended = True

                if not is_attended:
                    if bid not in self.bag_unattended_timer:
                        self.bag_unattended_timer[bid] = current_time
                else:
                    self.bag_unattended_timer.pop(bid, None)

            # 4. Rendering
            annotated = frame.copy()
            owner_to_bags = defaultdict(list)
            for bag_id, owner_id in self.bag_owner.items():
                owner_to_bags[owner_id].append(bag_id)

            for pid, p in tracked_persons.items():
                if pid in owner_to_bags:
                    owned_bags = owner_to_bags[pid]
                    bags_str = ",".join(map(str, owned_bags))
                    color = (0, 165, 255) 
                    label = f"ID: {pid} (Bag: {bags_str})"
                    
                    pt1 = (int(p['x']) - 10, p['y1'] - 35)
                    pt2 = (int(p['x']) + 10, p['y1'] - 35)
                    pt3 = (int(p['x']), p['y1'] - 20)
                    triangle_cnt = np.array([pt1, pt2, pt3])
                    cv2.drawContours(annotated, [triangle_cnt], 0, color, -1)
                else:
                    color = (0, 255, 0) if p.get('stationary') else (200, 200, 200)
                    label = f"ID: {pid}"

                cv2.rectangle(annotated, (p['x1'], p['y1']), (p['x2'], p['y2']), color, 2)
                cv2.putText(annotated, label, (p['x1'], p['y1'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            for bid, b in tracked_bags.items():
                cv2.rectangle(annotated, (b['x1'], b['y1']), (b['x2'], b['y2']), (255, 0, 0), 2)
                cv2.putText(annotated, f"Bag: {bid}", (b['x1'], b['y1'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            for gid, members in groups.items():
                active_members = [pid for pid in members if pid in tracked_persons]
                if len(active_members) > 1:
                    g_x1 = min(tracked_persons[pid]['x1'] for pid in active_members) - 20
                    g_y1 = min(tracked_persons[pid]['y1'] for pid in active_members) - 20
                    g_x2 = max(tracked_persons[pid]['x2'] for pid in active_members) + 20
                    g_y2 = max(tracked_persons[pid]['y2'] for pid in active_members) + 20

                    Visualizer.draw_dashed_rectangle(annotated, (g_x1, g_y1), (g_x2, g_y2), (0, 255, 255), 2)
                    cv2.putText(annotated, f"Group {gid}", (g_x1, g_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            for bid, bag in tracked_bags.items():
                if bag.get('stationary', False) and bid in self.bag_unattended_timer:
                    bag_center = (int(bag['x']), int(bag['y']))
                    time_left_alone = current_time - self.bag_unattended_timer[bid]

                    if time_left_alone >= self.config.ABANDON_TIME_THRESH:
                        Visualizer.draw_alert(annotated, bag_center[0], bag_center[1])
                        Visualizer.draw_dashed_circle(
                            annotated,
                            center=bag_center,
                            radius=self.config.NEAR_GROUP_RADIUS,
                            color=(255, 0, 255),
                            thickness=2
                        )

            out.write(annotated)

        cap.release()
        out.release()
        print(f"Done! Saved to {output_path}")