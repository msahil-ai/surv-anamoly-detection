# 🧳 Real-Time Abandoned Baggage Detection System (RT-ABDS)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green?style=flat-square&logo=opencv)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange?style=flat-square&logo=scikit-learn)

## 📖 Abstract
The **Real-Time Abandoned Baggage Detection System** is a computer vision pipeline designed to autonomously track pedestrians and luggage in video feeds, associate bags with their rightful owners, and trigger automated alerts when items are left unattended. By combining deep learning-based object detection (YOLOv8), multi-object tracking (BoT-SORT), and density-based spatial clustering (DBSCAN), the system mimics human situational awareness to reduce false positives in crowded environments.

---

## 📂 Project Structure

The codebase is highly modularized to separate configuration, core logic, and visualization.

    📦 RT-ABDS-Project
     ┣ 📜 config.py        # ⚙️ Hyperparameters, thresholds, and class definitions
     ┣ 📜 clusterer.py     # 🧠 Spatial grouping logic utilizing DBSCAN
     ┣ 📜 visualizer.py    # 🎨 UI/UX rendering (bounding boxes, dashed radii, alerts)
     ┣ 📜 detector.py      # 🚀 Core pipeline: Inference, kinematics, and association
     ┗ 📜 main.py          # 🏁 Execution entry point

---

## 🧠 Algorithmic Pipeline & Methodology

The system operates on a continuous 5-stage pipeline per frame:

### 1. Detection & Tracking (`YOLOv8` + `BoT-SORT`)
The system extracts bounding boxes for predefined classes (Persons, Backpacks, Handbags, Suitcases). `BoT-SORT` maintains temporal consistency by assigning unique IDs to objects across frames. A custom **Anti-Flicker ID Mapping** algorithm prevents stationary bags from being assigned new IDs when bounding boxes fluctuate slightly.

### 2. Kinematic Analysis
To differentiate between a moving passenger and a stationary bag, the system maintains a rolling history buffer of `N` frames. The instantaneous pixel velocity `v` is calculated to determine movement. If the velocity falls below a threshold for a set duration, the object's state transitions to "Stationary".

### 3. Spatial Grouping (`DBSCAN`)
Passengers rarely travel alone. To prevent false alarms when a passenger leaves a bag with a family member, the system applies **Density-Based Spatial Clustering of Applications with Noise (DBSCAN)**. 
* Individuals within an epsilon radius are clustered into groups. 
* A bag associated with *any* member of a cluster is considered safe as long as *at least one* cluster member is nearby.

### 4. Ownership Association
When an unowned bag enters the frame, the system computes the distance between the bag and all potential persons. 
* If a person stays within the association radius for a sustained period, they are formally "locked" as the owner.
* The system enforces a maximum bag limit per person to prevent a single individual from "owning" all luggage in a dense crowd.

### 5. Abandonment Evaluation
Once ownership is locked, the system continuously monitors the distance between the bag and the owner (or the owner's cluster). If the owner steps outside the "Safe Zone" radius, an unattended timer begins. 

If the timer exceeds the abandonment threshold, the system triggers the **ABANDONED BAG** alert and visualizes the breached safe zone.

---

## ⚙️ Configuration Parameters (`config.py`)

The system's behavior can be fine-tuned based on camera angle, focal length, and environment density.

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `GROUP_EPS` | `int` | `150` | DBSCAN radius for clustering people (pixels). |
| `ASSOCIATION_DIST_THRESH` | `int` | `200` | Max distance to associate a bag with a new owner. |
| `NEAR_GROUP_RADIUS` | `int` | `150` | The "Safe Zone" radius around a bag. |
| `ASSOCIATION_TIME_THRESH`| `int` | `3` | Seconds a person must stay near an orphan bag to claim it. |
| `ABANDON_TIME_THRESH` | `int` | `3` | Seconds a bag must be left alone to trigger the visual alert. |
| `MAX_BAGS_PER_PERSON` | `int` | `3` | Caps the number of items one track ID can own to prevent bias. |

---

## 🚀 Installation & Usage

### Prerequisites
Ensure you have Python 3.8+ installed. Install the required dependencies:

    pip install opencv-python numpy scikit-learn ultralytics lapx

*(Note: `lapx` is required for the BoT-SORT multi-object tracker).*

### Execution
1. Clone the repository and navigate to the project directory.
2. Place your target video file in the root directory (e.g., `sample2.mp4`).
3. Execute the pipeline:

    python main.py

* The system will automatically download the `yolov8l.pt` weights upon first execution.
* Press `Ctrl+C` in the terminal to interrupt the processing safely.
* The annotated output will be saved to `output_final-15-05-26.mp4`.

---

## 🔮 Future Enhancements
* **Perspective Transformation:** Mapping pixel distances to real-world metric distances using Homography to handle deep depth-of-field variations.
* **Re-Identification (ReID):** Integrating a feature-extraction backbone (e.g., OSNet) to re-associate owners with bags even if they leave and re-enter the camera frame.
