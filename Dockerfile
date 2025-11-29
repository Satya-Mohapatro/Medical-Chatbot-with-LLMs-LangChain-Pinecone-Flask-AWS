FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y git && apt-get clean

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]
