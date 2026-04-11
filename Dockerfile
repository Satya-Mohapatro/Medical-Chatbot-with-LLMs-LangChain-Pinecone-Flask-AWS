FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces requires port 7860
EXPOSE 7860

CMD ["python3", "app.py"]