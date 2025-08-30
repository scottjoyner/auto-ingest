
import os, re, uuid, csv, json, time
from datetime import datetime
import numpy as np

from transformers import AutoTokenizer, AutoModel
from gliner import GLiNER

imbeddings_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
imbeddings_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
entity_recognition_classifier = GLiNER.from_pretrained("urchade/gliner_large-v2")


def named_entity_recognizer(query, labels=["Person", "Place", "Event", "Date", "Subject"]):
    entities = entity_recognition_classifier.predict_entities(query, labels)
    return entities

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
            transcriptions = "/mnt/8TB_2025/fileserver/dashcam/transcriptions"
            key_list = list_files(file_path)

            total = len(key_list)
            current = 1
            print(f"{total} Videos left to process")
            for x in range(len(key_list)):
                if os.path.exists(f"{transcriptions}/{key_list[x]}_medium_transcription.txt"):
                    with open(f"{transcriptions}/{key_list[x]}_medium_transcription.txt", 'r') as file:

                        # Load the content as a JSON object
                        data = json.load(file)
                        print(key_list[x], end="|")
                        embeddings = get_embeddings_for_text(data['text'])
                        print(embeddings, end="|")
                        print(data['text'], end="|")
                        print("")
                    with open(f"{transcriptions}/{key_list[x]}_medium_transcription_imbeddings.txt", 'w') as file:
                        print(key_list[x])
                        print(data['text'])
                        for segment in data["segments"]:
                            print(segment['text'], segment['tokens'])

list_directories("/mnt/8TBHDD/fileserver/dashcam/")
list_directories("/mnt/8TB_2025/fileserver/dashcam/")
