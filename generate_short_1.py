import whisper
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

def create_shorts_video(
    input_video_path: str,
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

    # 3) Add subtitle overlay
    print("Adding subtitles overlay...")
    final_output_path = output_video_path
    with VideoFileClip(vertical_video_path) as vclip:
        txt_clip = (TextClip(transcription,
                             fontsize=60,
                             font="Arial",
                             color="white",
                             method='caption',
                             size=(vclip.w * 0.8, None),  # auto height
                             align='center')
                    .set_position(("center", 0.8*vclip.h))
                    .set_duration(vclip.duration))

        result_clip = CompositeVideoClip([vclip, txt_clip])
        result_clip.write_videofile(final_output_path,
                                    fps=vclip.fps,
                                    codec="libx264",
                                    audio_codec="aac")

    print("Finished! Output saved to:", final_output_path)


if __name__ == "__main__":
    create_shorts_video(
        input_video_path="input.mp4",
        output_video_path="shorts_output.mp4",
        model_size="base",
        target_width=1080,
        target_height=1920
    )
