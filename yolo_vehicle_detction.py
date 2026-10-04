from auto_ingest_config import get_fileserver_path
import ultralytics
ultralytics.checks()
from collections import defaultdict
import os, re
from datetime import datetime
from moviepy.editor import *
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from PIL import Image
import uuid

from auto_ingest.backend import torch_device

# Dashcam clips carry a camera suffix: <key>_F.MP4 (front), <key>_R.MP4 (rear),
# <key>_FR.MP4 (front+rear stacked). Same convention as PATTERN in
# compress_dashcam.py / compress_dashcam2.py and clip_base_key() in
# auto_ingest/dashcam/yolo_embeddings.py. The digit/separator in front of the
# suffix is optional so the legacy 12_F.mp4 shape still matches.
# NOTE: the "." is escaped and the suffix is anchored to the end of the name.
# The old literal `_F.MP4` pattern had an unescaped "." (i.e. "any character"),
# so `CLIP_FXMP4` was accepted as a front-camera clip, and being unanchored and
# front-only it could never see a rear clip.
_CAMERA_SUFFIX_RE = re.compile(r"_(\d+)?[_-]?(FR|F|R)\.mp4$", re.IGNORECASE)

def list_files(directory):
    """Return the sorted base keys (<stem> minus the camera suffix) to process.

    The key is the *base* key, i.e. the stem with _F/_R/_FR removed, because
    that is what the rest of the chain is named after: the sidecar we write and
    skip ({key}_YOLOv8n.csv), the video we open ({key}.MP4, see below) and
    yolo_heatmap.list_files(), which rebuilds keys from {key}_YOLOv8n.csv and
    then requires {key}.MP4 to exist. Front and rear clips of one clip collapse
    onto a single key, so both cameras are detected instead of only the front.
    """
    file_keys = set([])
    for filename in os.listdir(directory):
        m = _CAMERA_SUFFIX_RE.search(filename)
        if not m:
            continue
        # group(2) is the camera token itself; everything before it (minus the
        # separator that joined them) is the base key.
        file_keys.add(filename[:m.start(2)].rstrip("_-"))
    file_keys_copy = file_keys.copy()
    for filename in file_keys_copy:
        if os.path.exists(f"{directory}/{filename}_YOLOv8n.csv"):
            file_keys.remove(filename)
    #     if os.path.exists(f"{directory}/{filename}_FR.MP4"):
    #         file_keys.remove(filename)
    return sorted(list(file_keys))

def is_valid_date_structure(dir_name):
    try:
        # If the directory name can be converted to a date, it's valid
        datetime.strptime(dir_name, "%Y/%m/%d")
        return True
    except ValueError:
        # If a ValueError is raised, the directory name is not a date
        return False

def list_directories(base_path):
    for root, dirs, files in os.walk(base_path):
        # print(root,dirs,files)
        # Split the path to analyze if it ends with a YYYY/MM/DD structure
        path_parts = root[len(base_path):].split(os.sep)
        # print(path_parts)
        # Rejoin with the correct separator to normalize across OSes
        normalized_path = "/".join(path_parts)
        temp_path = root[len(base_path):]        
        if is_valid_date_structure(temp_path):
            # print(normalized_path)
            print(f"Valid directory structure found: {root}")
            file_path = root

            key_list = list_files(file_path)

            total = len(key_list)
            current = 1
            print(f"{total} Videos left to process")
            # New version
            for x in range(len(key_list)):
                try:
                    # Dictionary to store tracking history with default empty lists
                    track_history = defaultdict(lambda: [])
                    import torch
                    device = torch_device()
                    if device != "cpu":
                        print(f"Running on the GPU ({device})")
                    else:
                        print("Running on the CPU")
                    model = YOLO("yolov8n-seg.pt", device=device)
                    # Load the YOLO model with segmentation capabilities
                    # model = YOLO("yolov8n-seg.pt")
                    # model = YOLO("yolov8n-seg.pt")


                    # Open the video file
                    from auto_ingest.cv2_vaapi import enable_vaapi
                    cap = cv2.VideoCapture(os.path.join(file_path, f"{key_list[x]}.MP4"))
                    enable_vaapi(cap)
                    
                    # Retrieve video properties: width, height, and frames per second
                    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

                    # Initialize video writer to save the output video with the specified properties
                    # out = cv2.VideoWriter("2024_0623_064739_F_TRACK.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
                    frame = 0
                    s = f"Key,vehicle_id,confidence,classification,xywh,xyxy,Frame\n"
                    while True:
                        frame +=1
                        # Read a frame from the video
                        ret, im0 = cap.read()
                        if not ret: 
                            print("Video frame is empty or video processing has been successfully completed.")
                            break

                        # Create an annotator object to draw on the frame
                        # annotator = Annotator(im0, line_width=2)

                        # Perform object tracking on the current frame
                        results = model.track(im0, persist=True)

                        # print(results)
                        # Check if tracking IDs and masks are present in the results
                        # print(len(results))
                        for result in results[0]:
                            if result.boxes.id is not None:
                                # if result.boxes.cls is not None and result.boxes.conf is not None and result.boxes.xyxy is not None:
                                id = int(result.boxes.cls.int().cpu().tolist()[0])
                                classification = result.names[id]
                                res = result.boxes.xyxy[0].int().cpu().tolist()
                                # print(f"{key_list[x]}_{frame}_{classification}_{result.boxes.id.int().cpu().tolist()[0]}, {result.boxes.id.int().cpu().tolist()[0]}, {classification}, {(result.boxes.conf.float()*100).tolist()[0]:.2f}%, {result.boxes.xywh.int().cpu().tolist()[0]}, {result.boxes.xyxy.int().cpu().tolist()[0]}, {frame}\n")
                                s += f"{key_list[x]}_{frame}_{classification}_{result.boxes.id.int().cpu().tolist()[0]}, {result.boxes.id.int().cpu().tolist()[0]}, {classification}, {(result.boxes.conf.float()*100).tolist()[0]:.2f}%, {result.boxes.xywh.int().cpu().tolist()[0]}, {result.boxes.xyxy.int().cpu().tolist()[0]}, {frame}\n"
                                print(f"{key_list[x]}_{frame}_{classification}_{result.boxes.id.int().cpu().tolist()[0]}")
                            if result.boxes.id is None:
                                if len(result.boxes.cls.int().cpu().tolist()) > 0:
                                    # print(result.boxes.cls.int().cpu().tolist())
                                    id = int(result.boxes.cls.int().cpu().tolist()[0])
                                    classification = result.names[id]
                                    res = result.boxes.xyxy[0].int().cpu().tolist()
                                    label_id = str(uuid.uuid4())
                                    # print(f"{key_list[x]}_{frame}_{classification}_{label_id}, {label_id}, {classification}, {(result.boxes.conf.float()*100).tolist()[0]:.2f}%, {result.boxes.xywh.int().cpu().tolist()[0]}, {result.boxes.xyxy.int().cpu().tolist()[0]}, {frame}\n")
                                    s += f"{key_list[x]}_{frame}_{classification}_{label_id}, {label_id}, {classification}, {(result.boxes.conf.float()*100).tolist()[0]:.2f}%, {result.boxes.xywh.int().cpu().tolist()[0]}, {result.boxes.xyxy.int().cpu().tolist()[0]}, {frame}\n"
                                    print(f"{key_list[x]}_{frame}_{classification}_{label_id}")
                        # Exit the loop if 'q' is pressed
                        # if cv2.waitKey(1) & 0xFF == ord("q"):
                        #     break
                    with open(os.path.join(file_path, f"{key_list[x]}_YOLOv8n.csv"), "w") as f:
                        print(s,file=f)
                    # Release the video writer and capture objects, and close all OpenCV windows
                    # out.release()
                    cap.release()
                finally:
                    current += 1


image_save_location = "/media/scott/NAS/324ab5fd-8cb6-4a27-bd56-e648a5fcdb7a/images/"
# base_directory = get_fileserver_path("dashcam")  # Adjust this path to your base directory
# list_directories(base_directory)
base_directory = get_fileserver_path("dashcam")
list_directories(base_directory)
