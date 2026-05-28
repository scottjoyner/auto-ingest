FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tzdata cron \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY content_os /app/content_os
COPY requirements.txt /app/requirements.txt

# Install content-os package
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -e .

# Install requirements in stages:
# 1) Non-nvidia, non-whisper packages (CPU PyTorch index)
# 2) nvidia packages separately (they conflict with CPU index)
# 3) openai-whisper with no-build-isolation (needs pkg_resources)
RUN sed '/^openai-whisper\|^nvidia-/d' /app/requirements.txt > /tmp/reqs.txt \
    && pip install --no-cache-dir -r /tmp/reqs.txt \
       --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir nvidia-cublas-cu12==12.8.4.1 \
       nvidia-cuda-cupti-cu12==12.8.90 \
       nvidia-cuda-nvcc-cu12==12.9.86 \
       nvidia-cuda-nvrtc-cu12==12.8.93 \
       nvidia-cuda-runtime-cu12==12.8.90 \
       nvidia-cudnn-cu12==9.10.2.21 \
       nvidia-cufft-cu12==11.3.3.83 \
       nvidia-cufile-cu12==1.13.1.3 \
       nvidia-curand-cu12==10.3.9.90 \
       nvidia-cusolver-cu12==11.7.3.90 \
       nvidia-cusparse-cu12==12.5.8.93 \
       nvidia-cusparselt-cu12==0.7.1 \
       nvidia-nccl-cu12==2.27.3 \
       nvidia-nvjitlink-cu12==12.8.93 \
       nvidia-nvtx-cu12==12.8.90 \
    && pip install --no-cache-dir --no-build-isolation openai-whisper

# Copy all project files
COPY . /app

CMD ["python", "-m", "content_os", "--help"]
