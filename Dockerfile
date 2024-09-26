FROM python:3.11

WORKDIR /app

COPY . .

EXPOSE 8000

CMD ["/bin/sh", "-c", "source /venv/bin/activate && pip install -r requirements.txt && uvicorn main:app --host 0.0.0.0 --port 8000"]