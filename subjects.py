
import os, re, uuid, csv, json, time
from datetime import datetime
import numpy as np
from moviepy.editor import VideoFileClip
from transformers import AutoTokenizer, AutoModel
from gliner import GLiNER
from pyannote.audio import Pipeline


entity_recognition_classifier = GLiNER.from_pretrained("urchade/gliner_large-v2") 

def named_entity_recognizer(query, labels=["Person", "Place", "Event", "Date", "Subject"]):
    entities = entity_recognition_classifier.predict_entities(query, labels)
    return entities


def list_files(directory):
    file_keys = set([])
    for filename in os.listdir(directory):
        if re.search("_medium_transcription.txt", filename):
            file_keys.add(filename.rsplit('_medium_transcription.txt', 1)[0])
    file_keys_copy = file_keys.copy()
    for filename in file_keys_copy:
        if os.path.exists(f"/media/deathstar/8TB_2025/fileserver/dashcam/transcriptions/{filename}_medium_transcription_entities.csv"):
            file_keys.remove(filename)
    return sorted(list(file_keys))

def is_valid_date_structure(dir_name):
    try:
        datetime.strptime(dir_name, "%Y/%m/%d")
        return True
    except ValueError:
        return False

def list_directories(base_path):

    transcriptions = "/media/deathstar/8TB_2025/fileserver/dashcam/transcriptions"
    
    key_list = list_files(transcriptions)

    total = len(key_list)
    current = 1
    print(f"{total} Videos left to process")
    for x in range(len(key_list)):
        
        if os.path.exists(f"{transcriptions}/{key_list[x]}_medium_transcription.txt"):
            with open(f"{transcriptions}/{key_list[x]}_medium_transcription.txt", 'r') as file:
                data = json.load(file)
                # with open(f"{transcriptions}/{key_list[x]}_medium_transcription_entities.txt", 'w') as file:
                
                all_text_response = named_entity_recognizer(data['text'])
                if len(all_text_response) > 0:
                    with open(f"{transcriptions}/{key_list[x]}_medium_transcription_entities.csv", 'w', newline='') as file:
                        print(key_list[x])
                        # Get the header by using the keys from the first dictionary, adding 'id' as the first column
                        header = ['id'] + list(all_text_response[0].keys())
                        
                        writer = csv.DictWriter(file, fieldnames=header)
                        
                        # Write the header to the CSV
                        writer.writeheader()
                        
                        # Add the ID and write each row to the CSV
                        for index, row in enumerate(all_text_response, start=1):
                            row['id'] = key_list[x]  # Add an ID column
                            writer.writerow(row)



# list_directories("/media/deathstar/8TBHDD/fileserver/dashcam/")
list_directories("/mnt/8TB_2025/fileserver/dashcam/")