FROM python:3.11

WORKDIR /app

RUN python3 -m venv /opt/venv

COPY . .
RUN . /opt/venv/bin/activate && pip install -r requirements.txt

RUN ollama pull SpeakLeash/bielik-11b-v2.2-instruct-imatrix:Q8_0

EXPOSE 8000

CMD . /opt/venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8000