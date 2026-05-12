# AI Tennis Serve Prediction Model

## Overview

This project is a computer vision and machine learning pipeline that analyzes tennis serves using YOLOv8, trajectory tracking, and clustering techniques. The system processes tennis match frames from the TrackNet dataset, detects the tennis ball, tracks its movement across time, identifies bounce locations, and classifies serves as in or out.

The goal of the project was to combine supervised learning, unsupervised learning, and data analysis into a real-world sports analytics application.

---

# Features

* Tennis ball detection using YOLOv8
* Ball trajectory tracking across frames
* Missing-frame interpolation for smoother tracking
* Bounce detection using velocity analysis
* Serve location clustering with KMeans
* In vs Out serve classification
* Trajectory and clustering visualizations

---

# Tech Stack

## Languages

* Python

## Libraries and Frameworks

* PyTorch
* YOLOv8 (Ultralytics)
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* OpenCV

---

# Machine Learning Concepts Used

## Supervised Learning

Used YOLOv8 object detection trained on labeled tennis data to detect tennis balls in video frames.

## Unsupervised Learning

Used KMeans clustering to group serve bounce locations into estimated service box regions.

## Computer Vision

Processed image frames and extracted object coordinates from bounding boxes.

## Data Processing

Built ball trajectories using frame-by-frame positional tracking and interpolation.

---

# Dataset

This project uses the TrackNet tennis dataset.

Dataset structure:

Game -> Clip -> Frames

Each clip contains sequential image frames from a tennis rally or serve.

---

# How It Works

## 1. Object Detection

YOLOv8 detects objects in each frame.

Only tennis ball detections are kept.

Each detection is converted into an (x, y) center coordinate.

---

## 2. Trajectory Tracking

The detected positions are stored frame-by-frame to create a trajectory.

If the ball is not detected in certain frames, interpolation is used to estimate missing positions.

---

## 3. Bounce Detection

Bounce locations are estimated using velocity analysis.

The program detects when the vertical direction of the ball changes sharply, which indicates a bounce.

---

## 4. Clustering

KMeans clustering groups bounce locations into two clusters representing service box regions.

This allows the system to estimate court regions without manually hardcoding court boundaries.

---

## 5. In vs Out Classification

The distance from each bounce point to the nearest cluster center is calculated.

Serves closer to cluster centers are classified as IN.

Serves farther away are classified as OUT.

---

# Example Pipeline

Raw Frame -> YOLO Detection -> Ball Trajectory -> Bounce Detection -> KMeans Clustering -> IN/OUT Classification

---

# Results

The project successfully:

* Detected tennis ball positions across clips
* Generated ball trajectories
* Identified serve bounce locations
* Clustered serve locations into court regions
* Classified serves as in or out

The system also produced visualizations for:

* Ball trajectories
* Bounce locations
* Cluster centers
* Final serve classifications

---

# Challenges

Some of the biggest challenges during development included:

* Inconsistent ball detections across frames
* Clips starting after the serve was already hit
* Environment and dependency conflicts
* GPU/CUDA compatibility issues
* Handling missing data in trajectories

These challenges required debugging machine learning environments, handling noisy data, and improving trajectory processing.

---

# Future Improvements

Potential future improvements include:

* Court line detection using OpenCV
* Kalman filtering for smoother tracking
* Full video analysis instead of clips
* Better serve classification accuracy
* Real-time serve tracking
* Deep learning-based bounce prediction

---

# Skills Demonstrated

* Machine Learning
* Computer Vision
* Supervised Learning
* Unsupervised Learning
* Data Processing
* Data Visualization
* Python Development
* Model Inference
* Trajectory Analysis
* Debugging ML Environments
* High Performance Computing (HPC)

---

# Links To Datasets Used

*https://www.kaggle.com/datasets/sofuskonglevoll/tracknet-tennis
*https://app.roboflow.com/gabriels-workspace-5jcur/tennis-ball-detection-ci-project/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true

# Author

Gabriel Holfester

Computer Vision / Machine Learning Project
