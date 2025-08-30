import pandas as pd
import numpy as np
import ast, re
import cv2
import os
from datetime import datetime


def save_image(key, xyxy):
    
    source_dir = "/media/deathstar/8TBHDD/fileserver/dashcam"
    # Split the string by underscores
    parts = key.split('_')

    # Extract relevant pieces
    year = parts[0]
    month = parts[1][:2]
    day = parts[1][2:]
    timestamp = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"
    frame = int(parts[4])
    id_ = "_".join(parts[5:])

    # Build the file path
    dir_path = f"{source_dir}/{year}/{month}/{day}/{timestamp}.MP4"
    save_dir = f"/media/deathstar/324ab5fd-8cb6-4a27-bd56-e648a5fcdb7a/images/{timestamp}"
    # Output
    print("dir =", dir_path)
    print("frame =", frame)
    print("id =", id_)
    print(xyxy)
    os.makedirs(save_dir, exist_ok=True)
        # Open video file
    cap = cv2.VideoCapture(dir_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {dir_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    success, frame = cap.read()
    cap.release()
    if not success:
        raise ValueError(f"Failed to read frame {frame} from {dir_path}")
    x1, y1, x2, y2 = xyxy
    cropped = frame[y1:y2, x1:x2]
    # Save the image
    output_path = os.path.join(save_dir, f"{timestamp}_{id_}.png")
    cv2.imwrite(output_path, cropped)

def list_files(directory):
    file_keys = set([])
    for filename in os.listdir(directory):
        if re.search(".MP4", filename):
            file_keys.add(filename.rsplit('.MP4', 1)[0])
    file_keys_copy = file_keys.copy()
    for filename in file_keys_copy:
        save_dir = f"/media/deathstar/324ab5fd-8cb6-4a27-bd56-e648a5fcdb7a/images/{filename}"
        if os.path.exists(save_dir):
            file_keys.remove(filename)
    return sorted(list(file_keys))

def is_valid_date_structure(dir_name):
    try:
        datetime.strptime(dir_name, "%Y/%m/%d")
        return True
    except ValueError:
        return False
def parse_row(row):
    # Splitting by comma while ignoring commas within brackets
    parts = []
    temp = ""
    in_brackets = False
    for char in row:
        if char == '[':
            in_brackets = True
        if char == ']':
            in_brackets = False
        if char == ',' and not in_brackets:
            parts.append(temp.strip())
            temp = ""
        else:
            temp += char
    parts.append(temp.strip())  # Add the last part after the loop ends

    # Convert bracketed parts into Python lists using ast.literal_eval safely
    parts = [ast.literal_eval(part) if '[' in part else part for part in parts]
    return parts
# Safely calculate area only when xywh is a proper list
def calculate_area(xywh):
    if isinstance(xywh, (list, tuple)) and len(xywh) == 4:
        return xywh[2] * xywh[3]
    return np.nan  # or return 0 if you prefer
def list_directories(base_path):
    for root, dirs, files in os.walk(base_path):
        path_parts = root[len(base_path):].split(os.sep)
        normalized_path = "/".join(path_parts)
        temp_path = root[len(base_path):]        
        if is_valid_date_structure(temp_path):
            print(f"Valid directory structure found: {root}")
            file_path = root
            transcriptions = "/media/deathstar/8TB_2025/fileserver/dashcam/transcriptions"
            audio_dir = "/media/deathstar/8TB_2025/fileserver/dashcam/audio"
            
            key_list = list_files(file_path)

            total = len(key_list)
            current = 1
            print(f"{total} Videos left to process")
            for x in range(len(key_list)):
                print(key_list[x])
                if os.path.exists(f"{file_path}/{key_list[x]}_YOLOv8n.csv"):

                    # Initialize an empty list to store each row as a dictionary
                    data = []

                    # Open the file and process each line
                    with open(f"{file_path}/{key_list[x]}_YOLOv8n.csv", 'r') as file:
                        header = file.readline().strip().split(',')
                        for line in file:
                            parsed_line = parse_row(line.strip())
                            data.append(dict(zip(header, parsed_line)))

                    # Create a DataFrame from the list of dictionaries
                    df = pd.DataFrame(data)
                    if len(df) > 0:

                        # Calculate area
                        df['area'] = df['xywh'].apply(calculate_area)
                        df = df.dropna(subset=['area'])

                        # Sort by area descending
                        df_sorted = df.sort_values(by='area', ascending=False)
                        df_sorted = df_sorted[df_sorted['Key'].str.len() < 30]
                        print(df_sorted[['Key', 'vehicle_id', 'xywh', 'area']])
                        # Keep only the largest detection per vehicle_id
                        df_unique = df_sorted.drop_duplicates(subset='vehicle_id', keep='first')

                        # Done — print or save
                        print(df_unique[['Key', 'vehicle_id', 'xywh', 'area']])

                        for index, row in df_unique.iterrows():
                            save_image(row['Key'], row['xyxy'])

list_directories("/media/deathstar/8TBHDD/fileserver/dashcam/")
list_directories("/media/deathstar/8TB_2025/fileserver/dashcam/")