import os, re, uuid, csv, json, time
from datetime import datetime
import whisper
import numpy as np
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

# Set ImageMagick binary path
os.environ["IMAGEMAGICK_BINARY"] = "/usr/bin/convert"  # Adjust path if necessary
# # Version 3 Source
# def create_text_image(text, font_size, image_size, font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
#     """
#     Create an image with the specified text using Pillow.
#     :param text: The text to display
#     :param font_size: The size of the font
#     :param image_size: The size of the image (width, height)
#     :param font_path: Path to the font file (default is DejaVuSans-Bold)
#     :return: Pillow Image object
#     """
#     img = Image.new('RGBA', image_size, color=(0, 0, 0, 0))  # Transparent background
#     draw = ImageDraw.Draw(img)
#     font = ImageFont.truetype(font_path, font_size)

#     # Use textbbox to get bounding box
#     bbox = draw.textbbox((0, 0), text, font=font)
#     text_width = bbox[2] - bbox[0]  # width = right - left
#     text_height = bbox[3] - bbox[1]  # height = bottom - top

#     # Position text in the center horizontally and vertically near the bottom
#     position = ((image_size[0] - text_width) // 2, image_size[1] - text_height - 10)
#     draw.text(position, text, font=font, fill="white")

#     return img
# # Version 4
# def create_text_image(text, font_size, image_size, highlight_word=None, font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
#     """
#     Create an image with the specified text using Pillow, with an optional highlight for the currently spoken word.
#     :param text: The full text to display
#     :param font_size: The size of the font
#     :param image_size: The size of the image (width, height)
#     :param highlight_word: The word to highlight
#     :param font_path: Path to the font file (default is DejaVuSans-Bold)
#     :return: Pillow Image object
#     """
#     img = Image.new('RGBA', image_size, color=(0, 0, 0, 0))  # Transparent background
#     draw = ImageDraw.Draw(img)
#     font = ImageFont.truetype(font_path, font_size)

#     # Create text box for the full sentence
#     bbox = draw.textbbox((0, 0), text, font=font)
#     text_width = bbox[2] - bbox[0]  # width = right - left
#     text_height = bbox[3] - bbox[1]  # height = bottom - top

#     # Position text in the center horizontally and vertically near the bottom
#     position = ((image_size[0] - text_width) // 2, image_size[1] - text_height - 10)

#     # Draw black border for text
#     border_offset = 5  # Adjust for thicker or thinner borders
#     draw.text((position[0] - border_offset, position[1] - border_offset), text, font=font, fill="black")
#     draw.text((position[0] + border_offset, position[1] - border_offset), text, font=font, fill="black")
#     draw.text((position[0] - border_offset, position[1] + border_offset), text, font=font, fill="black")
#     draw.text((position[0] + border_offset, position[1] + border_offset), text, font=font, fill="black")
    
#     # Draw the main text
#     draw.text(position, text, font=font, fill="white")

#     return img
# # Version 5
# def create_text_image(text, font_size, image_size, highlight_word=None, font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
#     """
#     Create an image with the specified text using Pillow, with an optional highlight for the currently spoken word.
#     :param text: The full text to display
#     :param font_size: The size of the font
#     :param image_size: The size of the image (width, height)
#     :param highlight_word: The word to highlight
#     :param font_path: Path to the font file (default is DejaVuSans-Bold)
#     :return: Pillow Image object
#     """
#     img = Image.new('RGBA', image_size, color=(0, 0, 0, 0))  # Transparent background
#     draw = ImageDraw.Draw(img)
#     font = ImageFont.truetype(font_path, font_size)

#     # Split the text into words
#     words = text.split()

#     # Calculate the total width of all words
#     x_position = 0
#     y_position = (image_size[1] - font_size) // 2  # Vertically center the text
#     line_height = font_size * 1.2  # Line spacing
#     print(words, highlight_word)
#     for word in words:
#         print(word)
#         # Apply background color if this word is the one to highlight
#         if word == highlight_word:
#             print("Same")
#             print(word, highlight_word)
#             word_bbox = draw.textbbox((x_position, y_position), word, font=font)
#             draw.rectangle([word_bbox[0] - 2, word_bbox[1] - 2, word_bbox[2] + 2, word_bbox[3] + 2], fill="yellow")
#             # Draw the actual word
#             draw.text((x_position, y_position), word, font=font, fill="orange")
#         else:
#             # Draw the actual word
#             draw.text((x_position, y_position), word, font=font, fill="white")
#         # # Draw black border around the word
#         # border_offset = 1  # Adjust border thickness
#         # draw.text((x_position - border_offset, y_position - border_offset), word, font=font, fill="black")
#         # draw.text((x_position + border_offset, y_position - border_offset), word, font=font, fill="black")
#         # draw.text((x_position - border_offset, y_position + border_offset), word, font=font, fill="black")
#         # draw.text((x_position + border_offset, y_position + border_offset), word, font=font, fill="black")

        

#         # Update the x_position for the next word (space between words)
#         word_bbox = draw.textbbox((x_position, y_position), word, font=font)
#         x_position += word_bbox[2] - word_bbox[0] + 10  # Add some space between words

#     return img

def create_text_image(text, font_size, image_size, highlight_word=None, font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", debug=False):
    """
    Create an image with the specified text using Pillow, with an optional highlight for the currently spoken word.
    :param text: The full text to display
    :param font_size: The size of the font
    :param image_size: The size of the image (width, height)
    :param highlight_word: The word to highlight
    :param font_path: Path to the font file (default is DejaVuSans-Bold)
    :param debug: Whether to save the image for debugging
    :return: Pillow Image object
    """
    img = Image.new('RGBA', image_size, color=(0, 0, 0, 0))  # Transparent background
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)

    # Find the position of the highlight word in the text
    highlight_start = 0
    if highlight_word:
        highlight_start = text.find(highlight_word)

    # Draw the sentence word by word, checking if it matches the word to highlight
    x_position = 0
    y_position = (image_size[1] - font_size) // 2  # Vertically center the text
    line_height = font_size * 1.2  # Line spacing
    words = text.split()

    for word in words:
        word_bbox = draw.textbbox((x_position, y_position), word, font=font)
        
        # Check if the current word matches the highlighted word
        if highlight_word and text[highlight_start:highlight_start + len(highlight_word)] == word:
            # Apply background color (highlight) to the word
            draw.rectangle([word_bbox[0] - 5, word_bbox[1] - 5, word_bbox[2] + 5, word_bbox[3] + 5], fill="yellow")

        # Draw black border around the word
        border_offset = 2  # Adjust border thickness
        draw.text((x_position - border_offset, y_position - border_offset), word, font=font, fill="black")
        draw.text((x_position + border_offset, y_position - border_offset), word, font=font, fill="black")
        draw.text((x_position - border_offset, y_position + border_offset), word, font=font, fill="black")
        draw.text((x_position + border_offset, y_position + border_offset), word, font=font, fill="black")

        # Draw the actual word
        draw.text((x_position, y_position), word, font=font, fill="white")

        # Update the x_position for the next word (space between words)
        x_position += word_bbox[2] - word_bbox[0] + 10  # Add some space between words

    if debug:
        img.save("debug_image.png")  # Save the image for debugging purposes
        print("Saved debug image: debug_image.png")

    return img

def draw_words_image(wordlist, index,font_size, image_size, highlight_word=None, font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", debug=True):
    """
    Create an image with the specified text using Pillow, with an optional highlight for the currently spoken word.
    :param text: The full text to display
    :param font_size: The size of the font
    :param image_size: The size of the image (width, height)
    :param highlight_word: The word to highlight
    :param font_path: Path to the font file (default is DejaVuSans-Bold)
    :param debug: Whether to save the image for debugging
    :return: Pillow Image object
    """
    img = Image.new('RGBA', image_size, color=(0, 0, 0, 0))  # Transparent background
    draw = ImageDraw.Draw(img)
    font_word = ImageFont.truetype(font_path, font_size * 1.2)
    font = ImageFont.truetype(font_path, font_size)
    x_len = 0
    # Draw the sentence word by word, checking if it matches the word to highlight
    x_position = 0
    y_position = (image_size[1] - font_size) // 2  # Vertically center the text
    line_height = font_size * 1.2  # Line spacing
    count = 0
    x_position = 100
    for wordinfo in wordlist:
        word = wordinfo["word"]
        word_bbox = draw.textbbox((x_position, y_position), word, font=font)
        if x_position > (980 - (word_bbox[2] - word_bbox[0] + 10)):
            y_position = y_position + 120
            x_position = 100
            word_bbox = draw.textbbox((x_position, y_position), word, font=font)
        
        if count == index:
            draw.rectangle([word_bbox[0] - 5, word_bbox[1] - 5, word_bbox[2] + 5, word_bbox[3] + 5], fill="blue")
            
            border_offset = 2  # Adjust border thickness
            draw.text((x_position - border_offset, y_position - border_offset), word, font=font_word, fill="white")
            draw.text((x_position + border_offset, y_position - border_offset), word, font=font_word, fill="white")
            draw.text((x_position - border_offset, y_position + border_offset), word, font=font_word, fill="white")
            draw.text((x_position + border_offset, y_position + border_offset), word, font=font_word, fill="white")
            draw.text((x_position, y_position), word, font=font_word, fill="black")
        elif count < index:
                    # Draw black border around the word
            border_offset = 2  # Adjust border thickness
            draw.text((x_position - border_offset, y_position - border_offset), word, font=font, fill="white")
            draw.text((x_position + border_offset, y_position - border_offset), word, font=font, fill="white")
            draw.text((x_position - border_offset, y_position + border_offset), word, font=font, fill="white")
            draw.text((x_position + border_offset, y_position + border_offset), word, font=font, fill="white")
            draw.text((x_position, y_position), word, font=font, fill="black")
        elif count > index:
                    # Draw black border around the word
            border_offset = 2  # Adjust border thickness
            draw.text((x_position - border_offset, y_position - border_offset), word, font=font, fill="black")
            draw.text((x_position + border_offset, y_position - border_offset), word, font=font, fill="black")
            draw.text((x_position - border_offset, y_position + border_offset), word, font=font, fill="black")
            draw.text((x_position + border_offset, y_position + border_offset), word, font=font, fill="black")
            draw.text((x_position, y_position), word, font=font, fill="white")

            # Update the x_position for the next word (space between words)
        x_position += word_bbox[2] - word_bbox[0] + 10  # Add some space between words

        count += 1

    if debug:
        img.save("debug_image.png")  # Save the image for debugging purposes
        print("Saved debug image: debug_image.png")

    return img

def save_raw_result(whisper_result, file_path):
    with open(file_path, "w") as file:
        # Convert the dictionary to a string using json.dumps()
        file.write(json.dumps(whisper_result, indent=4))  # Optional: 'indent' argument for formatted output

def save_transcription_segments_to_csv(whisper_result, file_path):
    print(whisper_result)
    """
    Writes each segment of the Whisper transcription to a CSV with columns:
    [segment_index, start_time, end_time, segment_text]
    """
    with open(file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["SegmentIndex", "StartTime", "EndTime", "Text"])

        for i, segment in enumerate(whisper_result["segments"]):
            writer.writerow([i, segment["start"], segment["end"], segment["text"]])

    print(f"Segmented transcription saved to {file_path}")
# # Version 3 SRC
# def create_shorts_video(input_audio_path, input_video_path, output_text_path, output_whisper_dump, output_video_path, model_size="base", target_width=1080, target_height=1920):
#     """
#     Transcribes audio using Whisper, converts video to vertical 9:16, adds subtitle overlay with real-time word highlighting, and exports the final video.
#     """
#     print("Transcribing audio with Whisper...")
#     model = whisper.load_model(model_size)
#     result = model.transcribe(input_audio_path, word_timestamps=True)
#     print(f"Saving Transcription Dump to file...\nFile:{output_whisper_dump}")
#     save_raw_result(result, output_whisper_dump)
#     print(f"Saving Transcription to file...\nFile:{output_text_path}")
#     save_transcription_segments_to_csv(result, output_text_path)
#     # Convert video to vertical 9:16
#     print("Converting video to 9:16 format...")
#     vertical_video_path = "temp_vertical.mp4"
#     with VideoFileClip(input_video_path) as clip:
#         width, height = clip.size
#         current_aspect = width / height
#         target_aspect = target_width / target_height

#         if current_aspect > target_aspect:
#             new_width = int(height * target_aspect)
#             x_center = width // 2
#             x1 = x_center - new_width // 2
#             x2 = x_center + new_width // 2
#             cropped_clip = clip.crop(x1=x1, y1=0, x2=x2, y2=height)
#         else:
#             new_height = int(width / target_aspect)
#             y_center = height // 2
#             y1 = y_center - new_height // 2
#             y2 = y_center + new_height // 2
#             cropped_clip = clip.crop(x1=0, y1=y1, x2=width, y2=y2)

#         final_clip = cropped_clip.resize((target_width, target_height))
#         final_clip.write_videofile(vertical_video_path, fps=clip.fps, codec="libx264", audio_codec="aac")

#     # Add dynamic subtitles with word highlighting
#     print("Adding subtitles overlay with word highlighting...")
#     final_output_path = output_video_path
#     with VideoFileClip(vertical_video_path) as vclip:
#         word_clips = []
#         for segment in result["segments"]:
#             print(f"Segment:{segment}")
#             for word_info in segment["words"]:
#                 print(f"Word Info: {word_info}")
#                 start_time = word_info["start"]
#                 end_time = word_info["end"]
#                 word = word_info["word"]

#                 # Create highlighted text for current word
#                 img = create_text_image(word, font_size=85, image_size=(vclip.w, 100))
#                 txt_clip = ImageClip(np.array(img))
#                 txt_clip = txt_clip.set_duration(end_time - start_time)
#                 txt_clip = txt_clip.set_position(("center", 0.5 * vclip.h))
#                 txt_clip = txt_clip.set_start(start_time)

#                 word_clips.append(txt_clip)

#         # Combine video and text clips
#         result_clip = CompositeVideoClip([vclip] + word_clips)
#         result_clip.write_videofile(final_output_path, fps=vclip.fps, codec="libx264", audio_codec="aac")

#     print(f"Finished! Output saved to {final_output_path}")

# Version 4 
def transcribe_dashcam(input_audio_path, output_text_path, output_whisper_dump, model):
    """
    Transcribes audio using Whisper, converts video to vertical 9:16, adds subtitle overlay with real-time word highlighting, and exports the final video.
    """
    print("Transcribing audio with Whisper...")
    try:
        result = model.transcribe(input_audio_path, word_timestamps=True)
        
        print(f"Saving Transcription Dump to file...\nFile:{output_whisper_dump}")
        save_raw_result(result, output_whisper_dump)
        print(f"Saving Transcription to file...\nFile:{output_text_path}")
        save_transcription_segments_to_csv(result, output_text_path)
    except Exception as error:
        print(error)
    


# Helper function to list valid video directories
def list_files(directory, transcriptions, whisper_model):
    file_keys = set([])
    for filename in os.listdir(directory):
        if re.search("_R.MP4", filename):
            file_keys.add(filename.rsplit('_R.', 1)[0])
    file_keys_copy = file_keys.copy()
    for filename in file_keys_copy:
        if os.path.exists(f"{transcriptions}/{filename}_{whisper_model}_transcription.txt"):
            file_keys.remove(filename)
    return sorted(list(file_keys))

def is_valid_date_structure(dir_name):
    try:
        datetime.strptime(dir_name, "%Y/%m/%d")
        return True
    except ValueError:
        return False

def list_directories(base_path, whisper_model):
    for root, dirs, files in os.walk(base_path):
        path_parts = root[len(base_path):].split(os.sep)
        normalized_path = "/".join(path_parts)
        temp_path = root[len(base_path):]        
        if is_valid_date_structure(temp_path):
            print(f"Valid directory structure found: {root}")
            file_path = root
            transcriptions = "/mnt/8TB_2025/fileserver/dashcam/transcriptions"
            key_list = list_files(file_path, transcriptions, whisper_model)

            total = len(key_list)
            current = 1
            print(f"{total} Videos left to process")
            for x in range(len(key_list)):
                try:
                    print(key_list[x])
                    model = whisper.load_model(whisper_model)
                    transcribe_dashcam(
                        input_audio_path=f"{file_path}/{key_list[x]}_R.MP4",
                        output_text_path=f"{transcriptions}/{key_list[x]}_transcription.csv",
                        output_whisper_dump=f"{transcriptions}/{key_list[x]}_{whisper_model}_transcription.txt",
                        model=model
                    )
                    print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                finally:
                    print("Finally:", print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
if __name__ == "__main__":
    # base_directory = "/mnt/8TBHDD/fileserver/dashcam/"  # Adjust this path to your base directory
    # list_directories(base_directory, whisper_model="medium")
    base_directory = "/mnt/8TB_2025/fileserver/dashcam/"
    list_directories(base_directory, whisper_model="medium")
