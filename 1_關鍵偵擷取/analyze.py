import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

# Fix for protobuf compatibility issue
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Ensure ffmpeg is in path (fix for WinError 2)
# Add C:\Users\ASUS\miniconda3\Library\bin to PATH
os.environ["PATH"] = r"C:\Users\ASUS\miniconda3\Library\bin" + os.pathsep + os.environ["PATH"]

import tensorflow as tf

# Check for GPU (optional, will use CPU if not available)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU Available: {len(gpus)} device(s) detected.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected. QQ")

from ultralytics import YOLO

# Add TransNetV2 inference directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
inference_dir = os.path.join(current_dir, "TransNetV2", "inference")
if os.path.exists(inference_dir):
    sys.path.append(inference_dir)

try:
    from transnetv2 import TransNetV2
except ImportError:
    pass # Handled inside functions if needed

def get_video_info(video_path):
    import ffmpeg
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            return None
        
        fps = eval(video_stream['r_frame_rate'])
        frames = int(video_stream.get('nb_frames', 0))
        return {'fps': fps, 'frames': frames}
    except Exception as e:
        print(f"Error probing video: {e}")
        return {'fps': 24.0, 'frames': 0}

def detect_scenes(video_path, threshold=0.5):
    print(f"\n[1/2] Detecting Scenes (TransNetV2)...")
    
    weights_path = os.path.join(inference_dir, "transnetv2-weights")
    if not os.path.exists(weights_path):
        print(f"Error: TransNetV2 weights not found at {weights_path}")
        return []

    try:
        model = TransNetV2(weights_path)
        video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(video_path)
        scenes = model.predictions_to_scenes(single_frame_predictions, threshold=threshold)
        
        results = []
        for i, (start_frame, end_frame) in enumerate(scenes):
            results.append({
                "type": "Scene Cut",
                "frame": start_frame,
                "end_frame": end_frame,
                "description": f"Scene {i+1} Start"
            })
        return results
    except Exception as e:
        print(f"Error in TransNetV2 analysis: {e}")
        return []

def detect_objects(video_path):
    print(f"\n[2/2] Detecting Objects (YOLOv8 + ByteTrack)...")
    
    # Load model
    try:
        model = YOLO("yolov8n.pt")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return []

    results = []
    active_tracks = {} # track_id -> last_seen_frame
    
    # Run tracking
    # stream=True for memory efficiency
    # persist=True is important for tracking across frames but loop calling track does it automatically if we iterate
    # Actually YOLO.track() returns a generator if stream=True
    
    # We will iterate through frames
    tracker_results = model.track(source=video_path, stream=True, tracker="bytetrack.yaml", verbose=False, persist=True)
    
    frame_idx = 0
    
    for r in tqdm(tracker_results, desc="Processing Frames"):
        if r.boxes and r.boxes.id is not None:
            # Get boxes, classes, and track IDs
            boxes = r.boxes
            ids = boxes.id.cpu().numpy().astype(int)
            clss = boxes.cls.cpu().numpy().astype(int)
            names = r.names
            
            for track_id, cls_id in zip(ids, clss):
                obj_name = names[cls_id]
                
                # Check if this is a new track
                if track_id not in active_tracks:
                    # New object appeared
                    results.append({
                        "type": "Object Appeared",
                        "frame": frame_idx,
                        "description": f"New {obj_name} (ID: {track_id})"
                    })
                
                # Update last seen
                active_tracks[track_id] = frame_idx
        
        frame_idx += 1
        
    # Optional: Detect disappearances? 
    # Valid "Disappearance" is hard to define without a buffer, 
    # but strictly "Track Ended" means we never saw it again.
    # We can post-process this if needed, but for now "Appeared" is the critical event requested.
    
    return results

def analyze_video(video_path, output_csv, threshold=0.5):
    if not os.path.exists(video_path):
        print(f"Error: File {video_path} not found.")
        return

    # 1. Get FPS
    info = get_video_info(video_path)
    fps = info['fps']
    print(f"Video Info: FPS={fps:.2f}")

    # 2. Run Scene Detection
    scene_events = detect_scenes(video_path, threshold)

    # 3. Run Object Detection
    object_events = detect_objects(video_path)

    # 4. Merge and Sort
    all_events = scene_events + object_events
    # Sort by frame number
    all_events.sort(key=lambda x: x['frame'])

    # 5. Export
    print(f"\nGenerating report to {output_csv}...")
    
    data = []
    for event in all_events:
        frame = event['frame']
        timestamp = frame / fps
        
        row = {
            "Frame": frame,
            "Time (s)": round(timestamp, 3),
            "Event Type": event['type'],
            "Description": event['description']
        }
        
        # Add extra info if scene
        if event['type'] == "Scene Cut":
            row["End Frame"] = event.get('end_frame', '')
            if 'end_frame' in event:
                row["Duration (s)"] = round((event['end_frame'] - frame) / fps, 3)
        else:
            row["End Frame"] = ""
            row["Duration (s)"] = ""
            
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # 1. Filter columns
    cols = ["Frame", "Time (s)", "Event Type"]
    df = df[cols]
    
    # 2. Drop duplicate times
    # This keeps the first occurrence. Since we added scene_events first and used stable sort,
    # Scene Cuts will be prioritized over Object events if they happen at the exact same frame.
    df = df.drop_duplicates(subset=['Time (s)'], keep='first')
    
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Analysis Complete! Exported unique events to {output_csv}.")
    
    # 6. Extract Frames
    output_dir = os.path.join(os.path.dirname(output_csv), "keyframes")
    extract_keyframes(video_path, output_csv, output_dir)

def extract_keyframes(video_path, csv_path, output_dir):
    print(f"\n[Extra] Extracting keyframes to {output_dir}...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(csv_path)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    count = 0
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting"):
        frame_num = int(row['Frame'])
        event_type = row['Event Type'].replace(" ", "_")
        
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            # Construct filename: frame_{number}_{type}.jpg
            filename = f"frame_{frame_num:06d}_{event_type}.jpg"
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, frame)
            count += 1
            
    cap.release()
    print(f"Successfully extracted {count} images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Video Analysis (Scene + Object)")
    parser.add_argument("video_path", help="Path to input video", nargs='?', default=None)
    parser.add_argument("--output", help="Output CSV path", default=None)
    parser.add_argument("--threshold", type=float, default=0.5, help="Scene detection threshold")
    
    args = parser.parse_args()
    
    video_path = args.video_path
    
    if video_path is None:
        # Auto-detect mp4 files in current directory
        files = [f for f in os.listdir('.') if f.lower().endswith('.mp4')]
        if not files:
            print("Error: No video_path provided and no .mp4 files found in current directory.")
            sys.exit(1)
        video_path = files[0]
        print(f"Auto-detected video: {video_path}")
    if args.output:
        output_csv = args.output
    else:
        base = os.path.splitext(video_path)[0]
        output_csv = f"{base}_analysis.csv"
        
    analyze_video(video_path, output_csv, args.threshold)
