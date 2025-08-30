import pandas as pd
import ast, re
from moviepy.editor import *
from datetime import datetime
import easyocr
import matplotlib.pyplot as plt 

import numpy as np
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory


def list_files(directory):
    file_keys = set([])
    for filename in os.listdir(directory):
        if re.search("_YOLOv8n.csv", filename):
            file_keys.add(filename.rsplit('_YOLOv8n', 1)[0])
    file_keys_copy = file_keys.copy()   
    count = 0
    for filename in file_keys_copy:
        if not os.path.exists(f"{directory}/{filename}.MP4"):
            file_keys.remove(filename)
        if os.path.exists(f"{directory}/{filename}_heatmap.png"):
            file_keys.remove(filename)
    return sorted(list(file_keys))

def is_valid_date_structure(dir_name):
    try:
        # If the directory name can be converted to a date, it's valid
        datetime.strptime(dir_name, "%Y/%m/%d")
        return True
    except ValueError:
        # If a ValueError is raised, the directory name is not a date
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

def list_directories(base_path):
    for root, dirs, files in os.walk(base_path):
        print(root,dirs,files)
        # Split the path to analyze if it ends with a YYYY/MM/DD structure
        path_parts = root[len(base_path):].split(os.sep)
        print(path_parts)
        # Rejoin with the correct separator to normalize across OSes
        normalized_path = "/".join(path_parts)
        temp_path = root[len(base_path):]        
        if is_valid_date_structure(temp_path):
            print(normalized_path)
            print(f"Valid directory structure found: {root}")
            file_path = root
            key_list = list_files(file_path)

            total = len(key_list)
            current = 1
            print(f"{total} Videos left to process")
            print(key_list)
            # New version
            for x in range(len(key_list)):
                try:
                    # Initialize an empty list to store each row as a dictionary
                    data = []

                    # Open the file and process each line
                    with open(os.path.join(base_path, f"{key_list[x]}_YOLOv8n.csv"), 'r') as file:
                        header = file.readline().strip().split(',')
                        for line in file:
                            parsed_line = parse_row(line.strip())
                            data.append(dict(zip(header, parsed_line)))

                    # Create a DataFrame from the list of dictionaries
                    df = pd.DataFrame(data)
                    df['classification'] = df['classification'].str.replace('%', '').astype(float) / 100
                    # Calculate the area of each bounding box
                    print(df['xyxy'][0][0])
                    # df['area'] = df['xyxy'].apply(lambda xyxy: (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))

                    # Sum the areas of the bounding boxes for each vehicle_id
                    # area_sum = df.groupby('vehicle_id')['area'].sum().reset_index()
                    width = 2560
                    height = 1440

                    # Create a blank heatmap
                    heatmap = np.zeros((height, width))

                    # Iterate through the dataframe and add areas to the heatmap
                    for _, row in df.iterrows():
                        if str(row['xyxy']) != "nan":
                            x_min, y_min, x_max, y_max = row['xyxy']
                            heatmap[y_min:y_max, x_min:x_max] += 1

                    # Normalize the heatmap
                    heatmap = heatmap / np.max(heatmap)

                    # Plot the heatmap
                    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
                    plt.colorbar()
                    plt.title("Heatmap of Vehicle Detections")
                    plt.savefig(os.path.join(base_path, f"{key_list[x]}_heatmap.png"))
                    plt.clf()
                    # plt.show()
                finally:
                    current +=1
                    print(current, total, current/total)


base_directory = "/media/deathstar/8TBHDD/fileserver/dashcam/"  # Adjust this path to your base directory
list_directories(base_directory)
