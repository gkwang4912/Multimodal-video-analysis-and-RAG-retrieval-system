import cv2
import csv
import os
import glob

def time_to_msec(time_str):
    """Converts a time string HH:MM:SS.mmm to milliseconds."""
    try:
        if ':' not in time_str:
            return 0
        h, m, s = time_str.split(':')
        return int((int(h) * 3600 + int(m) * 60 + float(s)) * 1000)
    except ValueError:
        return 0

def extract_frames():
    video_path = 'test.mp4'
    csv_path = 'transcript.csv'
    output_folder = 'screenshots'

    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read CSV
    rows = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader) # Read header
            rows = list(reader)
    except UnicodeDecodeError:
         with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    updated_rows = []
    
    # Process rows
    print(f"Processing {len(rows)} rows...")
    
    for i, row in enumerate(rows):
        if len(row) < 3:
            updated_rows.append(row)
            continue
            
        start_time_str = row[0]
        end_time_str = row[1]
        
        start_msec = time_to_msec(start_time_str)
        end_msec = time_to_msec(end_time_str)
        
        # Define filenames
        start_filename = f"img_{i+1}_start.jpg"
        end_filename = f"img_{i+1}_end.jpg"
        start_filepath = os.path.join(output_folder, start_filename)
        end_filepath = os.path.join(output_folder, end_filename)
        
        # Capture Start Frame
        cap.set(cv2.CAP_PROP_POS_MSEC, start_msec)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(start_filepath, frame)
        else:
            print(f"Warning: Could not extract frame at {start_time_str}")

        # Capture End Frame
        cap.set(cv2.CAP_PROP_POS_MSEC, end_msec)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(end_filepath, frame)
        else:
            print(f"Warning: Could not extract frame at {end_time_str}")

        # Append new columns
        # Ensuring we don't duplicate if script runs multiple times on already modified file
        # But user format request implies overwriting or creating from scratch. 
        # The row currently has 3 columns. We append 2 more.
        new_row = [row[0], row[1], row[2], start_filename, end_filename]
        updated_rows.append(new_row)

    cap.release()

    # Define new header
    new_header = ['開始時間', '結束時間', '內容', '開始照片檔名', '結束照片檔名']

    # Write back to CSV
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(new_header)
            writer.writerows(updated_rows)
        print("Successfully updated transcript.csv and saved screenshots.")
    except Exception as e:
         print(f"Error writing CSV: {e}")

if __name__ == "__main__":
    extract_frames()
