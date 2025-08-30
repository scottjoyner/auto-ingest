import os
import numpy as np
from pyannote.audio import Audio, Model
from pyannote.core import Annotation, Segment
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from pydub import AudioSegment
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
# Path settings
audio_dir = "/media/deathstar/8TB_2025/fileserver/dashcam/audio"
rttm_suffix = "_speakers.rttm"

# Load embedding model
model = Model.from_pretrained("pyannote/embedding", use_auth_token="<HF_TOKEN>")
audio_processor = Audio(sample_rate=16000)

# Store segment info
embeddings = []
segment_metadata = []
LIMIT = 100
count = 0
for file in os.listdir(audio_dir):
    if file.endswith(".mp3"):
        file_key = file.replace(".mp3", "")
        rttm_path = os.path.join(audio_dir, file_key + rttm_suffix)
        if not os.path.exists(rttm_path):
            continue
        
        audio_file = os.path.join(audio_dir, file)
        annotation = Annotation()
        
        with open(rttm_path, "r") as rttm:
            for line in rttm:
                parts = line.strip().split()
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                segment = Segment(start, start + duration)
                annotation[segment] = speaker

        for segment, track, label in annotation.itertracks(yield_label=True):
            print(f"🔊 Segment: {segment.start:.2f}s - {segment.end:.2f}s from {audio_file}")
            try:
                waveform, sample_rate = audio_processor.crop(audio_file, segment)
                if waveform.shape[0] == 2:
                    waveform = waveform.mean(dim=0, keepdim=True)  # Convert stereo to mono
                if segment.duration < 0.5:
                    print(f"⚠️ Skipping short segment ({segment.duration:.2f}s)")
                    continue

                emb = model(waveform).data.numpy().mean(axis=0)
                embeddings.append(emb)
                segment_metadata.append({
                    "file": audio_file,
                    "segment": segment,
                    "local_speaker": label
                })
            except Exception as e:
                print(f"⚠️ Skipping segment due to error: {e}")

        count += 1
        if LIMIT < count:
            break
print(f"Total embeddings collected: {len(embeddings)}")
if not embeddings:
    print("❌ No embeddings were collected. Exiting early.")
    exit()

# Cluster speakers globally
X = np.array(embeddings)
# Use cosine distance (1 - cosine similarity)
distance_matrix = cosine_distances(X)

# You can tune eps (similarity threshold) and min_samples
clustering = DBSCAN(metric="precomputed", eps=0.3, min_samples=2)
labels = clustering.fit_predict(distance_matrix)

# n_clusters = 5  # You can tune this or use a method like DBSCAN
# clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
# labels = clustering.fit_predict(X)

# Mapping: global speaker ID to all segments
global_speaker_segments = defaultdict(list)

for idx, global_label in enumerate(labels):
    metadata = segment_metadata[idx]
    global_speaker_segments[global_label].append(metadata)

# Optional: create audio files per global speaker
output_dir = os.path.join(audio_dir, "global_speakers")
os.makedirs(output_dir, exist_ok=True)

for global_label, segments in global_speaker_segments.items():
    combined = AudioSegment.silent(duration=500)  # Start with brief silence
    for seg in segments:
        audio = AudioSegment.from_mp3(seg["file"])
        start_ms = int(seg["segment"].start * 1000)
        end_ms = int(seg["segment"].end * 1000)
        combined += audio[start_ms:end_ms] + AudioSegment.silent(duration=200)

    combined.export(os.path.join(output_dir, f"speaker_{global_label}.mp3"), format="mp3")

print("✅ Global speaker clustering complete. Output saved.")
