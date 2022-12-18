FROM python:3.10-slim-bullseye

WORKDIR /app

RUN pip install flask tensorflow numpy gunicorn pillow

COPY . .

CMD ["gunicorn", "app:app", "-w", "2", "-b", "0.0.0.0:8000"]