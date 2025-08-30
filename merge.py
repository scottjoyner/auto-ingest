import os, re
from datetime import datetime
from moviepy.editor import *

def list_files(directory):
    file_keys = set([])
    for filename in os.listdir(directory):
        if re.search("_F.MP4", filename):
            file_keys.add(filename.rsplit('_', 1)[0])

    file_keys_copy = file_keys.copy()
    for filename in file_keys_copy:
        if not os.path.exists(f"{directory}/{filename}_R.MP4"):
            file_keys.remove(filename)
        if os.path.exists(f"{directory}/{filename}_FR.MP4"):
            file_keys.remove(filename)
    
    for filename in file_keys:
        if not os.path.exists(f"{directory}/{filename}_R.MP4"):
            file_keys.remove(filename)
        if os.path.exists(f"{directory}/{filename}_FRA.MP4"):
            file_keys.remove(filename)
        if os.path.exists(f"{directory}/{filename}_FR.MP4"):
                file_keys.remove(filename)
    # # Check for corresponding _R.MP4 files and remove keys if not found
    # for key in list(file_keys):  # Convert to list to avoid modifying the set during iteration
    #     if f"{key}_R.MP4" not in os.listdir(directory):
    #         file_keys.remove(key)

    return sorted(list(file_keys))

# Example usage:
file_path = "/Volumes/Untitled/DCIM/Movie" # replace with your own file path

key_list = list_files(file_path)

video_pairs = []
for key in key_list:
    video_pairs.append([f"{key}_F.MP4", f"{key}_R.MP4", f"{key}_FR.MP4"])
total = len(key_list)
current = 0
print(f"{total} Videos left to process")
# New version
for x in range(len(video_pairs)):
    print(f"Working on {current}/{total}")
    try:
        clip_f = VideoFileClip(os.path.join(file_path, video_pairs[x][0]))
        clip_r = VideoFileClip(os.path.join(file_path, video_pairs[x][1])).without_audio()
        speed = clip_f.crop(y1=(clip_f.size[1]//10)*1, y2=clip_f.size[1])
        crop_width, crop_height = 165, 50
        speed = clip_f.crop(x1=20, y1=clip_f.h - (crop_height + 20), width=crop_width, height=crop_height)

        clip_r = clip_r.fx(vfx.mirror_x)
        clip_f = clip_f.crop(y1=(clip_f.size[1]//10)*3, y2=(clip_f.size[1]//10)*9.5)
        clip_r = clip_r.crop(y1=(clip_r.size[1]//10)*5, y2=(clip_r.size[1]//10)*9.5)
        background_clip = clips_array([[clip_f], [clip_r]])
        # Calculate the position to place the cropped clip in the center of the background clip
        pos_x = (background_clip.w - crop_width) / 2
        pos_y = ((background_clip.h - crop_height) / 2) + 100

        # Overlay the cropped clip onto the background clip at the calculated position
        final_clip = CompositeVideoClip([background_clip, speed.set_position((pos_x, pos_y))])
        # final_clip = CompositeVideoClip([background_clip])
        output_file = os.path.join(file_path, video_pairs[x][2])

        final_clip.write_videofile(output_file, codec="libx264", fps=clip_f.fps)
    finally:     
        current += 1


# # OLD VERSION 
# for x in range(len(video_pairs)):
#     clip_f = VideoFileClip(os.path.join(file_path, video_pairs[x][0]))
#     clip_r = VideoFileClip(os.path.join(file_path, video_pairs[x][1]))
#     speed = clip_f.crop(y1=(clip_f.size[1]//10)*1, y2=clip_f.size[1])
#     crop_width, crop_height = 165, 50
#     speed = clip_f.crop(x1=20, y1=clip_f.h - (crop_height + 20), width=crop_width, height=crop_height)

#     clip_r = clip_r.fx(vfx.mirror_x)
#     clip_f = clip_f.crop(y1=(clip_f.size[1]//10)*1, y2=(clip_f.size[1]//10)*7)
#     clip_r = clip_r.crop(y1=(clip_r.size[1]//10)*5.5, y2=(clip_r.size[1]//10)*9.5)

#     background_clip = clips_array([[clip_f], [clip_r]])

#     # Calculate the position to place the cropped clip in the center of the background clip
#     pos_x = (background_clip.w - crop_width) / 2
#     pos_y = ((background_clip.h - crop_height) / 2) + 100

#     # Overlay the cropped clip onto the background clip at the calculated position
#     final_clip = CompositeVideoClip([background_clip, speed.set_position((pos_x, pos_y))])

#     output_file = os.path.join(file_path, video_pairs[x][2])

#     final_clip.write_videofile(output_file, codec="libx264", fps=clip_f.fps)
