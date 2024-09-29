from fastapi import FastAPI, HTTPException
import time
from threading import Lock

### LLM
from langchain_ollama import ChatOllama

from langchain_core.messages import HumanMessage, SystemMessage

print("loading model")
local_llm = 'SpeakLeash/bielik-11b-v2.2-instruct-imatrix:Q8_0'
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")
print("model loaded")

app = FastAPI()
lock = Lock()

system_prompt = """
Na podstawie poniższej listy kroków, twoim zadaniem będzie ustalenie, czy pytanie dotyczy podatku PCC-3.
Zwróć jsona postaci {"is_pcc": "pcc"} jeśli pytanie dotyczy PCC-3 lub {"is_pcc": "no"} jeśli pytanie nie dotyczy PCC-3.

Kroki do określenia, czy wiadomość użytkownika dotyczy podatku PCC: 1. Sprawdzenie charakteru czynności: Zidentyfikuj, czy wiadomość opisuje jakąkolwiek czynność cywilnoprawną. Ustal, czy czynność ma charakter prawnym i finansowy (np. sprzedaż, wymiana, darowizna, pożyczka, zabezpieczenia hipoteczne). 2. Analiza przeniesienia własności lub praw majątkowych: Zwróć uwagę, czy wiadomość opisuje przeniesienie własności, takich jak rzeczy ruchome, nieruchomości lub inne prawa majątkowe. Czy transakcja dotyczy odpłatności lub innych korzyści finansowych? 3. Rozpoznanie umowy sprzedaży lub zamiany: Ustal, czy wiadomość odnosi się do sprzedaży (nabycie czegoś za pieniądze). Sprawdź, czy opisuje zamianę rzeczy lub praw (np. „wymiana mieszkania na samochód”). 4. Rozważenie możliwości pożyczki: Sprawdź, czy w wiadomości mowa o pożyczeniu pieniędzy lub innych wartości z obowiązkiem ich zwrotu. Poszukaj sformułowań sugerujących, że użytkownik „otrzymał środki do zwrotu” lub „przekazał środki znajomemu do oddania”. 5. Ocena umowy darowizny: Zwróć uwagę, czy użytkownik opisuje darowiznę, zwłaszcza z przejęciem zobowiązań (np. „otrzymałem dom z kredytem”). Sprawdź, czy darowizna nie jest bezpłatna, ale wiąże się z dodatkowymi obciążeniami. 6. Rozważenie odpłatnego użytkowania: Zidentyfikuj, czy wiadomość opisuje korzystanie z rzeczy lub nieruchomości w zamian za opłatę (np. „wynajem mieszkania” lub „użytkowanie samochodu za wynagrodzenie”). 7. Analiza zabezpieczeń hipotecznych: Zwróć uwagę, czy w wiadomości pojawia się ustanowienie hipoteki na nieruchomości (np. „ustanowiłem hipotekę na dom, aby zabezpieczyć kredyt”). 8. Kontekst sytuacyjny: Zrozum pełen kontekst wiadomości. Sprawdź, czy opisane działania mogą prowadzić do powstania zobowiązań finansowych lub majątkowych, które podlegają podatkowi PCC. 9. Wnioski: Jeśli wiadomość spełnia powyższe kryteria, można stwierdzić, że opisuje czynność cywilnoprawną objętą podatkiem PCC.
"""

from pydantic import BaseModel

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

import json

class MsgRequest(BaseModel):
    msg_id: str
    message: str

@app.post("/sendMichalMsg")
def send_msg(msg: MsgRequest):
  if not lock.acquire(blocking=False):
    raise HTTPException(status_code=503, detail="API is busy. Please try again later.")
  
  try:
    brbr = llm_json_mode.invoke([SystemMessage(content=system_prompt)] + [HumanMessage(content="PYTANIE UŻYTKOWNIKA: " + msg.message + "\nJSON:")])
    print(brbr.content)
    return json.loads(brbr.content)
  
  finally:
    lock.release()  # Zwalniamy blokadę na koniec operacji