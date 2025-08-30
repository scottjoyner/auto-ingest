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

def list_files(directory):
    file_keys = set([])
    for filename in os.listdir(directory):
        if re.search("_R.MP4", filename):
            file_keys.add(filename.rsplit('.', 1)[0])
        if re.search("_F.MP4", filename):
            file_keys.add(filename.rsplit('.', 1)[0])
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
                    if torch.cuda.is_available():
                        device = torch.device("cuda:0")
                        print("Running on the GPU")
                        model = YOLO("yolov8n-seg.pt")
                        model.to(device)
                    else:
                        device = torch.device("cpu")
                        model = YOLO("yolov8n-seg.pt")
                        model.to(device)
                        print("Running on the CPU")
                    # Load the YOLO model with segmentation capabilities
                    # model = YOLO("yolov8n-seg.pt")
                    # model = YOLO("yolov8n-seg.pt")


                    # Open the video file
                    cap = cv2.VideoCapture(os.path.join(file_path, f"{key_list[x]}.MP4"))
                    
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


image_save_location = "/media/deathstar/324ab5fd-8cb6-4a27-bd56-e648a5fcdb7a/images/"
# base_directory = "/mnt/8TBHDD/fileserver/dashcam/"  # Adjust this path to your base directory
# list_directories(base_directory)
base_directory = "/mnt/8TB_2025/fileserver/dashcam/"
list_directories(base_directory)
