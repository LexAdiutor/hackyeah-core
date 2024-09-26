from fastapi import FastAPI, HTTPException
import time
from threading import Lock

app = FastAPI()
lock = Lock()

@app.get("/")
def read_root():
  # Jeśli blokada jest zajęta, zwracamy odpowiedni błąd
  if not lock.acquire(blocking=False):
      raise HTTPException(status_code=503, detail="API is busy. Please try again later.")
  
  # Po udanym przejęciu blokady
  try:
    time.sleep(10)  # symulacja długotrwałej operacji
    return {"message": "Hello, World!"}
    
  finally:
    lock.release()  # Zwalniamy blokadę na koniec operacji
