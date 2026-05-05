#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Core libraries
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# YOLO
from ultralytics import YOLO

# Clustering
from sklearn.cluster import KMeans

model = YOLO("/home/gholfes/project/runs/detect/train4/weights/best.pt")

# Make the path start at the "tracknet" folder 
base_path = "/home/gholfes/project/tracknet/Dataset"

# Lists every item in "tracknet" and stores it in games
games = os.listdir(base_path)

all_data = {}


# In[ ]:


# Loop through each individual game 
for game in games:
    # Make the folder path to avoid string errors
    
    game_path = os.path.join(base_path, game)
    
    # Get all clips in each game 
    clips = os.listdir(game_path) 
    
    all_data[game] = {}
    
    # Go through each clip (rallies/sequences)
    for clip in clips[:5]:
        trajectory = []
        frame_id = 0   # reset per clip
        
        # Build path to clip to avoid errors 
        clip_path = os.path.join(game_path, clip)
        
        # Sorts the frames so each clip isn't chopped up
        frames = sorted(glob.glob(os.path.join(clip_path, "*.jpg")))
           
        # Go image by image and gather postions of balls 
        for f in frames:
            
            # Get boxes and confidence scores from a frame
            results = model(f)
            
            for r in results:
                boxes = r.boxes
                
                # Check if YOLO actually found something (players, tennis balls)
                if boxes is None or len(boxes) == 0:
                    continue
                
                # Takes care of YOLO accidently tracking multiple balls
                
                # Resets every frame so you only keep one detection
                best_box = None
                best_conf = 0
                
                # Loop through every detection (Players, Ball, Officals)
                for box in boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    
                    # Ignore everything but tennis balls (Tennis ball label == 2)
                    if cls_id != 2:
                        continue
                    
                    # Get the confidence of how sure yolo is something is a tennis ball
                    conf = float(box.conf[0])
                    
                    # Get the best detected ball per frame (if theres two being detected)
                    if conf > best_conf:
                        best_conf = conf
                        best_box = box
                
                # Only run if a ball is found
                if best_box is not None:
                    
                    # Get top left and bottom right coordinates and find the middle
                    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
                    
                    x = (x1 + x2) / 2
                    y = (y1 + y2) / 2
                    
                    # Add the middle and ID to trajectory
                    trajectory.append([frame_id, float(x), float(y)])
                                       
            # Increase frame ID
            frame_id += 1


    

        # Turn trajectory (X and Y values) into a excel style sheet 
        df = pd.DataFrame (trajectory, columns=["frame_id", "x", "y"])
        df = df.sort_values("frame_id").reset_index(drop=True)


        # Interpolate missing frames
        df = df.set_index("frame_id")
        df = df.reindex(range(df.index.min(), df.index.max()))

        df["x"] = df["x"].interpolate()
        df["y"] = df["y"].interpolate()

        df = df.reset_index()


        all_data[game][clip] = df


# In[12]:


for game in all_data:
    print(f"===== {game} =====")
    
    for clip in all_data[game]:
        
        df = all_data[game][clip]
        
        plt.figure()
        
        plt.plot(df["x"], df["y"])
        plt.gca().invert_yaxis()
        
        plt.title(f"{game} - {clip}")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        
        plt.show()


# In[8]:


all_bounces = {}

# Make sure we only use real game folders (ignore .ipynb_checkpoints)
games = [g for g in all_data if not g.startswith(".")]

for game in games:
    clips_bounce_pos = []
    
    # Id for each clip
    clip_id = 0
    
    for clip in all_data[game]:
        
        # Get dataframe for this clip
        df = all_data[game][clip]
        
        # Skip bad clips with too little data
        if len(df) < 5:
            clip_id += 1
            continue
        
        # Location of where the serve bounce was detected
        serve_bounce_x = 0.0
        serve_bounce_y = 0.0
        
        # Store all possible bounce candidates
        bounce_candidates = []
        
        # For every frame in clip find ALL possible bounces
        for i in range(5, len(df) - 1):
            y_prev = df.iloc[i - 1]["y"]
            y_current = df.iloc[i]["y"]
            y_after = df.iloc[i+1]["y"]
            
            velocity_before = y_current - y_prev
            velocity_after = y_after - y_current
            
            # If the velocity changes trajectory (falling → rising), this is a bounce candidate
            if velocity_before > 0 and velocity_after < 0:
                
                # Strength of bounce (bigger = more likely real bounce)
                strength = abs(velocity_before) + abs(velocity_after)
                
                bounce_candidates.append([i, strength])
        
        # If any bounce candidates are found pick the strongest one
        if len(bounce_candidates) > 0:
            
            # Get the index of the strongest bounce
            best_bounce = max(bounce_candidates, key=lambda x: x[1])
            bounce_index = best_bounce[0]
            
            # Get x and y position of that bounce
            serve_bounce_x = df.iloc[bounce_index]["x"]
            serve_bounce_y = df.iloc[bounce_index]["y"]
            
            clips_bounce_pos.append([clip_id, serve_bounce_x, serve_bounce_y])
        
        # If no bounce was found skip instead of adding garbage data
        clip_id += 1
        
    serve_df = pd.DataFrame (clips_bounce_pos, columns=["clip_id", "serve_x", "serve_y"])
    serve_df = serve_df.sort_values("clip_id").reset_index(drop=True)
    
    # Store per game
    all_bounces[game] = serve_df
        
                
                
            
        


# In[9]:


import matplotlib.pyplot as plt

for game in all_bounces:
    
    df = all_bounces[game]
    
    plt.figure()
    
    # Plot serve bounce locations
    plt.scatter(df["serve_x"], df["serve_y"])
    
    plt.gca().invert_yaxis()
    
    plt.title(f"Serve Bounce Locations - {game}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    
    plt.show()


# In[ ]:




