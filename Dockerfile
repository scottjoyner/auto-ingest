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

# Install content-os package
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -e .

# Install requirements in stages:
# 1) Non-nvidia, non-whisper packages (CPU PyTorch index)
# 2) openai-whisper with no-build-isolation (needs pkg_resources)
RUN sed '/^openai-whisper\|^nvidia-/d' /app/requirements.txt > /tmp/reqs.txt \
    && pip install --no-cache-dir -r /tmp/reqs.txt \
       --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --no-build-isolation openai-whisper

# Copy all project files
COPY . /app

CMD ["python", "-m", "content_os", "--help"]
