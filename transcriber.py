import ultralytics
ultralytics.checks()
from collections import defaultdict
import os, re
from datetime import datetime
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ImageClip
from ultralytics.utils.plotting import Annotator, colors
from PIL import Image
import uuid
import whisper
import numpy as np
import csv
from moviepy.config import change_settings
os.environ["IMAGEMAGICK_BINARY"] = "/usr/bin/convert"  # Adjust path if necessary
from PIL import Image, ImageDraw, ImageFont
def create_text_image(text, font_size, image_size, font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
    """
    Create an image with the specified text using Pillow.
    :param text: The text to display
    :param font_size: The size of the font
    :param image_size: The size of the image (width, height)
    :param font_path: Path to the font file (default is DejaVuSans-Bold)
    :return: Pillow Image object
    """
    # Create a blank image with a transparent background
    img = Image.new('RGBA', image_size, color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Use the specified font (make sure it's available)
    font = ImageFont.truetype(font_path, font_size)
    
    # Use textbbox() instead of textsize() to get the bounding box of the text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]  # width = right - left
    text_height = bbox[3] - bbox[1]  # height = bottom - top
    
    # Calculate the position to center the text horizontally and vertically at the bottom
    position = ((image_size[0] - text_width) // 2, image_size[1] - text_height - 10)
    
    # Draw the text on the image
    draw.text(position, text, font=font, fill="white")

    return img
# def create_text_image(text, font_size, image_size, font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
#     """
#     Create an image with the specified text using Pillow.
#     :param text: The text to display
#     :param font_size: The size of the font
#     :param image_size: The size of the image (width, height)
#     :param font_path: Path to the font file (default is Arial)
#     :return: Pillow Image object
#     """
#     # Create a blank image with white background
#     img = Image.new('RGBA', image_size, color=(0, 0, 0, 0))
#     draw = ImageDraw.Draw(img)

#     # Use a truetype font (you can specify a path to your font file)
#     font = ImageFont.truetype(font_path, font_size)
    
#     # Calculate the width and height of the text to center it
#     text_width, text_height = draw.textsize(text, font=font)
#     position = ((image_size[0] - text_width) // 2, image_size[1] - text_height - 10)
    
#     # Draw the text on the image
#     draw.text(position, text, font=font, fill="white")

#     return img

def save_transcription_segments_to_csv(whisper_result, file_path):
    """
    Writes each segment of the Whisper transcription to a CSV with columns:
    [segment_index, start_time, end_time, segment_text]
    """
    with open(file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["SegmentIndex", "StartTime", "EndTime", "Text"])

        for i, segment in enumerate(whisper_result["segments"]):
            writer.writerow([
                i,
                segment["start"],
                segment["end"],
                segment["text"]
            ])

    print(f"Segmented transcription saved to {file_path}")



def create_shorts_video(
    input_video_path: str,
    output_text_path: str,
    output_video_path: str,
    model_size="base",
    target_width=1080,
    target_height=1920
):
    """
    1) Transcribes audio using Whisper
    2) Converts video to vertical 9:16
    3) Adds subtitle overlay (simple or timed)
    4) Exports final video
    """
    # 1) Transcribe
    print("Transcribing audio with Whisper...")
    model = whisper.load_model(model_size)
    result = model.transcribe(input_video_path)
    transcription = result["text"]
    print(result, transcription)
    print(f"Saving Transcription audio to file...\nFile:{output_text_path}")
    save_transcription_segments_to_csv(result, output_text_path)

    # 2) Convert to vertical
    print("Converting video to 9:16 format...")
    vertical_video_path = "temp_vertical.mp4"
    with VideoFileClip(input_video_path) as clip:
        width, height = clip.size
        current_aspect = width / height
        target_aspect = target_width / target_height
        
        if current_aspect > target_aspect:
            # Crop horizontally
            new_width = int(height * target_aspect)
            x_center = width // 2
            x1 = x_center - new_width // 2
            x2 = x_center + new_width // 2
            cropped_clip = clip.crop(x1=x1, y1=0, x2=x2, y2=height)
        else:
            # Crop vertically
            new_height = int(width / target_aspect)
            y_center = height // 2
            y1 = y_center - new_height // 2
            y2 = y_center + new_height // 2
            cropped_clip = clip.crop(x1=0, y1=y1, x2=width, y2=y2)

        final_clip = cropped_clip.resize((target_width, target_height))
        final_clip.write_videofile(vertical_video_path,
                                   fps=clip.fps,
                                   codec="libx264",
                                   audio_codec="aac")
    # 3) Add subtitle overlay using Pillow
    print("Adding subtitles overlay...")
    final_output_path = output_video_path
    with VideoFileClip(vertical_video_path) as vclip:
        # Create the text image using Pillow
        img = create_text_image(transcription, font_size=60, image_size=(vclip.w, 100))
        
        # Convert Pillow image to a MoviePy ImageClip
        txt_clip = ImageClip(np.array(img))
        txt_clip = txt_clip.set_duration(vclip.duration)  # Match video duration
        txt_clip = txt_clip.set_position(("center", 0.8 * vclip.h))  # Position at the bottom

        # Composite the video and text clips
        result_clip = CompositeVideoClip([vclip, txt_clip])
        result_clip.write_videofile(final_output_path,
                                    fps=vclip.fps,
                                    codec="libx264",
                                    audio_codec="aac")
    ## 3) Add subtitle overlay
    # print("Adding subtitles overlay...")
    # final_output_path = output_video_path
    # with VideoFileClip(vertical_video_path) as vclip:
    #     txt_clip = (TextClip(transcription,
    #                          fontsize=60,
    #                          font="Arial",
    #                          color="white",
    #                          method='caption',
    #                          size=(vclip.w * 0.8, None),  # auto height
    #                          align='center')
    #                 .set_position(("center", 0.8*vclip.h))
    #                 .set_duration(vclip.duration))

    #     result_clip = CompositeVideoClip([vclip, txt_clip])
    #     result_clip.write_videofile(final_output_path,
    #                                 fps=vclip.fps,
    #                                 codec="libx264",
    #                                 audio_codec="aac")

    print("Finished! Output saved to:", final_output_path)

def list_files(directory):
    file_keys = set([])
    for filename in os.listdir(directory):
        if re.search("_R.MP4", filename):
            file_keys.add(filename.rsplit('_R.', 1)[0])
        # if re.search("_F.MP4", filename):
        #     file_keys.add(filename.rsplit('.', 1)[0])
    file_keys_copy = file_keys.copy()
    for filename in file_keys_copy:
        if os.path.exists(f"{directory}/{filename}_transcription.txt"):
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
                print(os.path.join(file_path, f"{key_list[x]}_transcription.txt"))
                create_shorts_video(
                    input_video_path=f"{file_path}/{key_list[x]}_R.MP4",
                    output_text_path=f"{file_path}/{key_list[x]}_transcription.csv",
                    output_video_path=f"{file_path}/{key_list[x]}_short.MP4",
                    model_size="base",
                    target_width=1080,
                    target_height=1920
                )
if __name__ == "__main__":
    base_directory = "/media/deathstar/8TBHDD/fileserver/dashcam/"  # Adjust this path to your base directory
    list_directories(base_directory)
    base_directory = "/media/deathstar/8TB_2025/fileserver/dashcam/"
    list_directories(base_directory)