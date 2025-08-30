
import os, re, uuid, csv, json, time
from datetime import datetime
import numpy as np
from moviepy.editor import VideoFileClip
from transformers import AutoTokenizer, AutoModel
from gliner import GLiNER
from pyannote.audio import Pipeline

imbeddings_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
imbeddings_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# entity_recognition_classifier = GLiNER.from_pretrained("urchade/gliner_large-v2") 

def get_speaker_notes(audio_path, output_file, speakers=0):
    if speakers == 0:
        speaker_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="<HF_TOKEN>")
        diarization = speaker_pipeline(audio_path, min_speakers=1, max_speakers=5)
        print(diarization)
        with open(output_file, "w") as rttm:
            diarization.write_rttm(rttm)
        del speaker_pipeline
        # with open("audio.rttm", "w") as rttm:
        #     diarization.write_rttm(rttm)
    return diarization


# Function to generate embeddings
def get_embeddings_for_text(text):
    inputs = imbeddings_tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = imbeddings_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy().tolist()

def list_files(directory):
    file_keys = set([])
    for filename in os.listdir(directory):
        if re.search("_R.MP4", filename):
            file_keys.add(filename.rsplit('_R.', 1)[0])
    file_keys_copy = file_keys.copy()
    for filename in file_keys_copy:
        if os.path.exists(f"/media/deathstar/8TB_2025/fileserver/dashcam/audio/{filename}_speakers.rttm"):
            file_keys.remove(filename)
    return sorted(list(file_keys))

def is_valid_date_structure(dir_name):
    try:
        datetime.strptime(dir_name, "%Y/%m/%d")
        return True
    except ValueError:
        return False

def list_directories(base_path):
    for root, dirs, files in os.walk(base_path):
        path_parts = root[len(base_path):].split(os.sep)
        normalized_path = "/".join(path_parts)
        temp_path = root[len(base_path):]        
        if is_valid_date_structure(temp_path):
            print(f"Valid directory structure found: {root}")
            file_path = root
            transcriptions = "/media/deathstar/8TB_2025/fileserver/dashcam/transcriptions"
            audio_dir = "/media/deathstar/8TB_2025/fileserver/dashcam/audio"
            
            key_list = list_files(file_path)

            total = len(key_list)
            current = 1
            print(f"{total} Videos left to process")
            for x in range(len(key_list)):
                if os.path.exists(f"{transcriptions}/{key_list[x]}_medium_transcription.txt"):
                    with open(f"{transcriptions}/{key_list[x]}_medium_transcription.txt", 'r') as file:
                        if not os.path.exists(f"/media/deathstar/8TB_2025/fileserver/dashcam/audio/{key_list[x]}.mp3"):
                            try:
                                print(key_list[x])
                                video = VideoFileClip(f"{file_path}/{key_list[x]}_R.MP4")
                                # Extract the audio from the video
                                audio = video.audio
                                # Save the extracted audio as an MP3 file
                                audio.write_audiofile(f"{audio_dir}/{key_list[x]}.mp3")
                            except AttributeError:
                                continue

list_directories("/media/deathstar/8TBHDD/fileserver/dashcam/")
list_directories("/media/deathstar/8TB_2025/fileserver/dashcam/")