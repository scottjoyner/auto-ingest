import os, re
from datetime import datetime
import pandas as pd
from moviepy.editor import *
import easyocr
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory


def list_files(directory):
    file_keys = set([])
    for filename in os.listdir(directory):
        if re.search("_F.MP4", filename):
            file_keys.add(filename.rsplit('_', 1)[0])
    for filename in os.listdir(directory):
        if re.search("_metadata.csv", filename):
            file_keys.add(filename.rsplit('_', 1)[0])
            file_keys.remove(filename.rsplit('_', 1)[0])
    file_keys_copy = file_keys.copy()
    count = 0
    for filename in file_keys_copy:
        if not os.path.exists(f"{directory}/{filename}_R.MP4"):
            file_keys.remove(filename)
        if os.path.exists(f"{directory}/{filename}_metadata.csv"):
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
file_key = "2024_0425_204318"
file_path = "D:/2024/04/25"

def list_directories(base_path):
    # for root, dirs, files in os.walk(base_path):
    #     # Split the path to analyze if it ends with a YYYY/MM/DD structure
    #     path_parts = root[len(base_path):].split(os.sep)
    #     # Rejoin with the correct separator to normalize across OSes
    #     normalized_path = "/".join(path_parts)
        
    #     if is_valid_date_structure(normalized_path):
    #         print(normalized_path)
    #         print(f"Valid directory structure found: {root}")
    #         file_path = root
    key_list = list_files(base_path)
    count = 0
    for filename in key_list:
        count += 1
        print(f"Working on {count}/{len(key_list)}")
        print(filename)
        with VideoFileClip(os.path.join(base_path, f"{filename}_F.mp4")) as clip_f:
            duration = clip_f.duration
            metadata = []
            # print(duration)
            speed = clip_f.crop(x1=20,y1=1370,x2=102,y2=1420)
            date = clip_f.crop(x1=2042,y1=1370,x2=2314,y2=1420)
            day = clip_f.crop(x1=2042,y1=1370,x2=2100,y2=1420)
            month = clip_f.crop(x1=2125,y1=1370,x2=2180,y2=1420)
            year = clip_f.crop(x1=2200,y1=1370,x2=2314,y2=1420)

            hour = clip_f.crop(x1=2328,y1=1370,x2=2395,y2=1420)
            minute = clip_f.crop(x1=2410,y1=1370,x2=2470,y2=1420)
            second = clip_f.crop(x1=2490,y1=1370,x2=2550,y2=1420)
            # time = clip_f.crop(x1=2328,y1=1370,x2=2550,y2=1420)
            time = clip_f.crop(x1=2328,y1=1370,x2=2550,y2=1420)
            time = clip_f.crop(x1=2328,y1=1370,x2=2550,y2=1420)
            # time = clip_f.crop(x1=2328,y1=1370,x2=2550,y2=1420)
            
            long = clip_f.crop(x1=226,y1=1370,x2=472,y2=1420)
            lon_heading = clip_f.crop(x1=230,y1=1375,x2=260,y2=1410)
            lon_deg1 = clip_f.crop(x1=280,y1=1375,x2=340,y2=1410)
            lon_deg2 = clip_f.crop(x1=355,y1=1375,x2=472,y2=1410)

            lat = clip_f.crop(x1=488,y1=1370,x2=758,y2=1420)
            lat_heading = clip_f.crop(x1=490,y1=1375,x2=525,y2=1410)
            lat_deg1 = clip_f.crop(x1=540,y1=1375,x2=600,y2=1410)
            lat_deg2 = clip_f.crop(x1=615,y1=1375,x2=735,y2=1410)
            s = "Speed,Latitude,Longitude,Frame\n"
            for x in range(int(duration)):
                speed.save_frame('speed.png', x)
                speed_value = reader.readtext('speed.png', detail = 0)
                try:
                    speed_str = f"{speed_value[0]}"
                except:
                    speed_str = f"000"
                #print(speed)
                # lat.save_frame('lat.png', x)
                # laditude = reader.readtext('lat.png', detail = 0)
                # long.save_frame('lon.png', x)
                # lon_heading.save_frame('lon_heading.png', x)
                # lon_heading_out = reader.readtext('lon_heading.png', detail = 0)
                lon_deg1.save_frame('lon_deg1_heading.png', x)
                lon_deg1_out = reader.readtext('lon_deg1_heading.png', detail = 0)
                lon_deg2.save_frame('lon_deg2_heading.png', x)
                lon_deg2_out = reader.readtext('lon_deg2_heading.png', detail = 0)
                try:
                    if len(lon_deg1_out) == 1 and len(lon_deg2_out) == 1:
                        longitude = f"{lon_deg1_out[0]}.{lon_deg2_out[0]}"
                    else:
                        longitude = f""
                except:
                    longitude = f""
                #print(longitude)
                # lat_heading.save_frame('lat_heading.png', x)
                # lat_heading_out = reader.readtext('lat_heading.png', detail = 0)
                lat_deg1.save_frame('lat_deg1_heading.png', x)
                lat_deg1_out = reader.readtext('lat_deg1_heading.png', detail = 0)
                lat_deg2.save_frame('lat_deg2_heading.png', x)
                lat_deg2_out = reader.readtext('lat_deg2_heading.png', detail = 0)
                try:
                    if len(lat_deg1_out) == 1 and len(lat_deg2_out) == 1:
                        laditude = f"{lat_deg1_out[0]}.{lat_deg2_out[0]}"
                    else:
                        laditude = f""
                except:
                    laditude = f""
                #print(laditude)
                # date.save_frame('date.png', x)
                # date_value = reader.readtext('date.png', detail = 0)
                # time.save_frame('time.png', x)
                # time_value = reader.readtext('time.png', detail = 0)
                # Include the below to extract dates and times
                # day.save_frame('day.png', x)
                # day_out = reader.readtext('day.png', detail = 0)
                # month.save_frame('month.png', x)
                # month_out = reader.readtext('month.png', detail = 0)
                # year.save_frame('year.png', x)
                # year_out = reader.readtext('year.png', detail = 0)
                # try:
                #     if len(day_out) == 1 and len(month_out) == 1 and len(year_out) == 1:
                #         date = f"{day_out[0]}:{month_out[0]}:{year_out[0]}"
                #     else:
                #         date = f""
                # except:
                #     date = f""
                # #print(date)
                # hour.save_frame('hour.png')
                # hour_out = reader.readtext('hour.png', detail = 0)
                # minute.save_frame('minute.png')
                # minute_out = reader.readtext('minute.png', detail = 0)
                # second.save_frame('second.png')
                # second_out = reader.readtext('second.png', detail = 0)
                # try:
                #     if len(hour_out) == 1 and len(minute_out) == 1 and len(second_out) == 1:
                #         time = f"{hour_out[0]}:{minute_out[0]}:{second_out[0]}"
                #     else:
                #         time = f""
                # except:
                #     time = f""
                #print(time)
                line = f"{filename},{speed_str},{longitude},{laditude},{x}"
                line = re.sub("o","0", line)
                line = re.sub(" ","", line)
                metadata.append(line)

        metadata_file = f"{filename}_metadata.csv"
        with open(os.path.join(base_path,metadata_file), 'w') as f:
            f.write(f"Key,MPH,Lat,Long,Frame\n")
            for line in metadata:
                f.write(f"{line}\n")

base_directory = "/Volumes/Untitled/DCIM/Movie"  # Adjust this path to your base directory
list_directories(base_directory)
