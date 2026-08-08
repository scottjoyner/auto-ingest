FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tzdata cron openssh-client rsync \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY content_os /app/content_os
COPY requirements.txt /app/requirements.txt

# Install content-os package first so its declared dependency floor is known.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -e .

# The repository requirements file is a broad workstation snapshot, not a
# container lockfile. Build the production CPU image without mutually exclusive
# GPU runtimes and the legacy TensorFlow/inaSpeechSegmenter path. The latter is
# used by 01_precompute_music_segments.py, not by the supported container
# services, and otherwise pulls tensorflow[and-cuda] plus multi-GB NVIDIA wheels.
#
# Also ignore the stale typing_extensions pin and reinstall a version satisfying
# Pydantic/Pydantic-core after the snapshot is applied.
RUN sed -E \
      -e '/^openai-whisper/d' \
      -e '/^nvidia-/d' \
      -e '/^onnxruntime-gpu/d' \
      -e '/^tensorflow([<=>]|$)/d' \
      -e '/^tf_keras([<=>]|$)/d' \
      -e '/^inaSpeechSegmenter([<=>]|$)/d' \
      -e '/^triton([<=>]|$)/d' \
      -e '/^typing_extensions([<=>]|$)/d' \
      /app/requirements.txt > /tmp/reqs.txt \
    && pip install --no-cache-dir -r /tmp/reqs.txt \
       --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir 'typing_extensions>=4.14.1' \
    && pip install --no-cache-dir --no-build-isolation openai-whisper \
    && pip check

# Copy all project files only after dependency resolution so source edits do not
# invalidate the expensive dependency layer.
COPY . /app

CMD ["python", "-m", "content_os", "--help"]
