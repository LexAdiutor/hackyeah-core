from tkinter import END
from typing import Dict, List, TypedDict
from fastapi import FastAPI, HTTPException
import time
from threading import Lock

### LLM
from langchain_ollama import ChatOllama

class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    tax_rate : float # User question
    fields: List[Dict[str, str]]
    tax_value: float
    is_known: bool
    query: str
    is_mortgage: bool
    is_only_one: bool
    type: str
    short_description: str

from langgraph.graph import StateGraph

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("shorten_description", shorten_description)
workflow.add_node("get_value", get_value)

workflow.add_node("sprzedaz", sprzedaz)
workflow.add_node("sprzedaz_many", sprzedaz_many)
workflow.add_node("sprzedaz_only_one", sprzedaz_only_one)

workflow.add_node("zamiana", zamiana)
workflow.add_node("darowizna_dlug", darowizna_dlug)
workflow.add_node("uzytkowanie", uzytkowanie)

workflow.add_node("pozyczka_fake", pozyczka_fake)
workflow.add_node("pozyczka", pozyczka)
workflow.add_node("hipoteka", hipoteka)
workflow.add_node("hipoteka_nieznana", hipoteka_nieznana)
workflow.add_node("hipoteka_znana", hipoteka_znana)

workflow.set_entry_point("shorten_description")
workflow.add_edge("shorten_description", "get_value")
workflow.add_edge("shorten_description", "get_value")

workflow.add_conditional_edges(
    "get_value",
    get_type,
    {
        "SPR": "sprzedaz",
        "ZAM": "zamiana",
        "POZ": "pozyczka_fake",
        "DAR": "darowizna_dlug",
        "UZY": "uzytkowanie",
    },
)

workflow.add_conditional_edges(
    "sprzedaz",
    sprzedaz_decyzja,
    {
        "one": "sprzedaz_only_one",
        "many": "sprzedaz_many",
    },
)


workflow.add_conditional_edges(
    "pozyczka_fake",
    pozyczka_fake_decyzja,
    {
        "hipoteka": "hipoteka",
        "pozyczka": "pozyczka",
    },
)

workflow.add_conditional_edges(
    "hipoteka",
    hipoteka_decyzja,
    {
        "hipoteka_znana": "hipoteka_znana",
        "hipoteka_nieznana": "hipoteka_nieznana",
    },
)

workflow.add_edge("zamiana", END)
workflow.add_edge("sprzedaz_only_one", END)
workflow.add_edge("sprzedaz_many", END)
workflow.add_edge("pozyczka", END)
workflow.add_edge("hipoteka_znana", END)
workflow.add_edge("hipoteka_nieznana", END)
workflow.add_edge("darowizna_dlug", END)
workflow.add_edge("uzytkowanie", END)

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

from utils import darowizna_dlug, get_type, get_value, hipoteka, hipoteka_decyzja, hipoteka_nieznana, hipoteka_znana, pozyczka, pozyczka_fake, pozyczka_fake_decyzja, shorten_description, sprzedaz, sprzedaz_decyzja, sprzedaz_many, sprzedaz_only_one, uzytkowanie, zamiana

MEMORY = {}

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
from utils import graph, info

class MsgRequest(BaseModel):
    msg_id: str
    message: str
    isFirstFormMessage: bool

@app.post("/sendMichalMsg")
def send_msg(msg: MsgRequest):
  if not lock.acquire(blocking=False):
    return {"info": info}
  
  try:
    if msg.isFirstFormMessage:
      MEMORY[msg.msg_id] = []
      brbr = llm_json_mode.invoke([SystemMessage(content=system_prompt)] + [HumanMessage(content="PYTANIE UŻYTKOWNIKA: " + msg.message + "\nJSON:")])
      return json.loads(brbr.content)
    
    res = json.loads(json.dumps(graph.invoke(msg.message)))
    return res
  
  finally:
    lock.release()  # Zwalniamy blokadę na koniec operacji