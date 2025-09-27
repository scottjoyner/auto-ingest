# syntax=docker/dockerfile:1
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn

COPY . .

ENV FLASK_APP=compare_transcripts.py \
    PORT=8000

EXPOSE 8000

CMD ["gunicorn", "-b", "0.0.0.0:8000", "quality_api.wsgi:app"]
