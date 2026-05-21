FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tzdata cron \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY content_os /app/content_os
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e . \
    && if [ -s /app/requirements.txt ]; then pip install --no-cache-dir -r /app/requirements.txt; fi

COPY . /app

ENTRYPOINT ["python"]
CMD ["-m", "content_os", "--help"]
