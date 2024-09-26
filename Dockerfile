FROM python:3.11

WORKDIR /app

COPY . .

EXPOSE 8000

CMD ["/venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt", "&&", "/venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]