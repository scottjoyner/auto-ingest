import os, re
from datetime import datetime
from moviepy.editor import *

def is_valid_date_structure(dir_name):
    try:
        # If the directory name can be converted to a date, it's valid
        datetime.strptime(dir_name, "%Y/%m/%d")
        return True
    except ValueError:
        # If a ValueError is raised, the directory name is not a date
        return False

def list_files(directory):
    file_keys = set([])
    for filename in os.listdir(directory):
        if re.search("_R.MP4", filename):
            file_keys.add(filename.rsplit('.', 1)[0])
        if re.search("_F.MP4", filename):
            file_keys.add(filename.rsplit('.', 1)[0])
    # file_keys_copy = file_keys.copy()
    # for filename in file_keys_copy:
    #     if os.path.exists(f"{directory}/{filename}_YOLOv8n.csv"):
    #         file_keys.remove(filename)
    #     if os.path.exists(f"{directory}/{filename}_FR.MP4"):
    #         file_keys.remove(filename)
    return sorted(list(file_keys))



def list_directories(base_path):
    total = 0
    for root, dirs, files in os.walk(base_path):
        # print(root,dirs,files)
        # Split the path to analyze if it ends with a YYYY/MM/DD structure
        path_parts = root[len(base_path):].split(os.sep)
        print(path_parts)
        # Rejoin with the correct separator to normalize across OSes
        normalized_path = "/".join(path_parts)
        temp_path = root[len(base_path):]        
        if is_valid_date_structure(temp_path):
            print(normalized_path)
            # print(f"Valid directory structure found: {root}")
            file_path = root

            key_list = list_files(file_path)
            print(f"{len(key_list)/2} Minutes")
            total += len(key_list)
        print(total)
    print(f"Total Footage Count: {total/120} Hours")
        



base_directory = "/media/deathstar/8TBHDD/fileserver/video/"  # Adjust this path to your base directory
list_directories(base_directory)
