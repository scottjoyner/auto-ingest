from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip, concatenate_videoclips
import whisper_timestamped as whisper
import json, sys, time
import moviepy.video.fx.all as vfx

def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio_path = video_path.replace('.mkv', '.wav')  # Adjusted for MKV
    audio.write_audiofile(audio_path)
    return audio_path

def transcribe_audio(audio_path):
    audio = whisper.load_audio(audio_path)
    model = whisper.load_model("medium", device="cpu")
    result = whisper.transcribe(model, audio, language="en")  # Assuming English language
    print(result)
    return result


def create_highlighted_caption_clips(transcript, video):
    caption_clips = []
    for segment in transcript['segments']:
        current_seg = ""
        count = 0
        for word_info in segment['words']:
            if count >= 20:
                current_seg += "\n" + word_info['text']
                count = 0
            else:
                current_seg += " " + word_info['text']
            count += len(word_info['text'])
            word = word_info['text']
            start_time = word_info['start']
            end_time = word_info['end']
            text_clip = TextClip(current_seg, fontsize=65, color='white', bg_color='blue', font=" Courier-New")
            text_clip = text_clip.set_position(('center', 'center')).set_duration(end_time - start_time).set_start(start_time)
            caption_clips.append(text_clip)
    return caption_clips


# def create_highlighted_caption_clips(transcript, video):
#     caption_clips = []
#     for segment in transcript['segments']:
#         full_text = segment['text']
#         segment_clip = None
#         previous_end_time = segment['start']

#         for word_info in segment['words']:
#             # Split the segment text into before, current, and after the word
#             current_word_start = full_text.find(word_info['text'])
#             before_word = full_text[:current_word_start]
#             current_word = full_text[current_word_start:current_word_start+len(word_info['text'])]
#             after_word = full_text[current_word_start+len(word_info['text']):]

#             # Create text clips for each part
#             before_text_clip = TextClip(before_word, fontsize=32, color='white', bg_color='black').set_duration(word_info['start'] - previous_end_time).set_start(previous_end_time)
#             current_text_clip = TextClip(current_word, fontsize=50, color='yellow', bg_color='black').set_duration(word_info['end'] - word_info['start']).set_start(word_info['start'])
#             after_text_clip = TextClip(after_word, fontsize=24, color='white', bg_color='black').set_duration(segment['end'] - word_info['end']).set_start(word_info['end'])
#             # Concatenate clips for this word

#             #word_clips = current_text_clip
#             # word_clips = concatenate_videoclips([before_text_clip, current_text_clip])
#             word_clips = concatenate_videoclips([before_text_clip, current_text_clip, after_text_clip])
#             segment_clip = word_clips if segment_clip is None else concatenate_videoclips([segment_clip, word_clips])

#             previous_end_time = word_info['end']

#         segment_clip = segment_clip.set_position(('center', 'center'))
#         caption_clips.append(segment_clip)

#     return caption_clips

def overlay_captions(video_path, caption_clips):
    video = VideoFileClip(video_path)
    final_video = CompositeVideoClip([video, *caption_clips])
    output_path = video_path.replace('.mkv', '_captioned.mkv')  # Adjusted for MKV
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')

def main(video_path):
    # audio_path = extract_audio(video_path)
    transcript = transcribe_audio(video_path)
    print("transcript")
    video = VideoFileClip(video_path)
    print("video")
    video = VideoFileClip(video_path)
    print("Caption")
    caption_clips = create_highlighted_caption_clips(transcript, video)
    print("overlay")
    overlay_captions(video_path, caption_clips)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
        main(video_file)
    else:
        print("Please specify a video file path.")

