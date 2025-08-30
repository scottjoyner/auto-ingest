import os, re
from datetime import datetime
from moviepy.editor import *
import easyocr
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory

def list_files(directory):
    file_keys = set([])
    for filename in os.listdir(directory):
        if re.search("_F.MP4", filename):
            file_keys.add(filename.rsplit('_', 1)[0])
    for filename in os.listdir(directory):
        if re.search("_metadata.txt", filename):
            file_keys.add(filename.rsplit('_', 1)[0])
            file_keys.remove(filename.rsplit('_', 1)[0])
    return file_keys

# Example usage:
file_path = "/Volumes/Untitled/DCIM/Movie" # replace with your own file path
# file_path = "/Users/scottjoyner/Documents/rides"
key_list = list_files(file_path)
count = 0
for filename in key_list:
    count += 1
    print(f"Working on {count}/{len(key_list)}")
    clip_f = VideoFileClip(os.path.join(file_path, f"{filename}_F.mp4"))
    duration = clip_f.duration
    metadata = []
    print(duration)
    speed = clip_f.crop(x1=18,y1=1370,x2=186,y2=1420)
    date = clip_f.crop(x1=2042,y1=1370,x2=2314,y2=1420)
    time = clip_f.crop(x1=2328,y1=1370,x2=2550,y2=1420)
    long = clip_f.crop(x1=226,y1=1370,x2=472,y2=1420)
    lat = clip_f.crop(x1=488,y1=1370,x2=758,y2=1420)
    for x in range(60):
        speed.save_frame('speed.png', x)
        speed_value = reader.readtext('speed.png', detail = 0)
        lat.save_frame('lat.png', x)
        laditude = reader.readtext('lat.png', detail = 0)
        long.save_frame('lon.png', x)
        longitude = reader.readtext('lon.png', detail = 0)
        date.save_frame('date.png', x)
        date_value = reader.readtext('date.png', detail = 0)
        time.save_frame('time.png', x)
        time_value = reader.readtext('time.png', detail = 0)
        if len(laditude) > 1:
            # Itterate over the laditude pices and concatenate them together
            full_laditude = ""
            for piece in laditude:
                full_laditude += piece
            full_laditude = re.sub(" ","", full_laditude)
            laditude = [full_laditude]
        if len(longitude) > 1:
            full_longitude = ""
            for piece in longitude:
                full_longitude += piece
            full_longitude = re.sub(" ","", full_longitude)
            longitude = [full_longitude]
        if len(speed_value) > 1:
            full_speed_value = ""
            for piece in speed_value:
                full_speed_value += piece
            full_speed_value = re.sub(" ","", full_speed_value)
            speed_value = [full_speed_value]
        if len(time_value) > 1:
            full_time_value = ""
            for piece in time_value:
                full_time_value += piece
            full_time_value = re.sub(" ","", full_time_value)
            time_value = [full_time_value]
        values = speed_value[0], laditude[0], longitude[0], date_value[0], time_value[0]
        string = ""
        for x in range(len(values)):
            string += re.sub(" ","", values[x])
        # print(speed_value, laditude, longitude, date_value, time_value)
        metadata.append(string)
    # If you are looking to include object detection for other things like road signs this would be the plaece to turn this on, I am leaving it off for performance reasons
    # for frame in clip_f.iter_frames(fps=1):
    #     clip_f.save_frame('frame.png')
    #     result = reader.readtext('frame.png')
    #     metadata.append(result)
    #     print(result)
    #     os.remove('frame.png')
    metadata_file = f"/Volumes/Untitled/DCIM/Movie/{filename}_metadata.txt"
    with open(metadata_file, 'w') as f:
        for line in metadata:
            f.write(f"{line}\n")
    del clip_f, speed, date, time, long, lat
