from typing import Dict, List, TypedDict
from fastapi import FastAPI, HTTPException
import time
from threading import Lock
from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama
from langgraph.graph import END
from langchain_core.messages import HumanMessage, SystemMessage

### LLM
from langchain_ollama import ChatOllama
from pydantic import BaseModel

local_llm = 'SpeakLeash/bielik-11b-v2.2-instruct-imatrix:Q8_0'
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format='json')

import operator
from typing_extensions import TypedDict
from typing import List, Annotated

class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    question : str # User question
    generation : str # LLM generation
    web_search : str # Binary decision to run web search
    max_retries : int # Max number of retries for answer generation 
    answers : int # Number of answers generated
    loop_step: Annotated[int, operator.add] 
    documents : List[str] # List of retrieved documents
    explanation: str
    
from langchain.schema import Document
from langgraph.graph import END
import json
from langchain_core.messages import HumanMessage, SystemMessage
import json
from langchain_core.messages import HumanMessage, SystemMessage

info = None

### Nodes
def shorten_description(state):
  global info
  print("Skracam opis...")
  info = "Skracam opis..."
  # System 
  desc_system_prompt = """Jesteś ekspertem w przekształcaniu złożonych sytuacji podatników na zwięzły opis.

  W zwięzłym opisie nie używaj potocznych słów. 

  Jeśli podatnik opisuje KUPNO KILKU RZECZY albo SPRZEDAŻ KILKU RZECZY, czyli porozumienie między dwoma osobami, w którym jedna kupuje kilka rzeczy za pieniądze od drugiej, wtedy skonstruuj zwięzyły opis dla każdej z rzeczy i oddziel je przecinkiem.

  Zwróć json z kluczem "short_description" zawierający zwięzły opis.
  """
  
  def get_short_desc_user_prompt(query):
    user_prompt_template = """
    PYTANIE: Wczoraj kupiłem na giełdzie samochodowej Fiata 126p rok prod. 1975, kolor zielony. Przejechane ma
    1000000 km, idzie jak przecinak, nic nie stuka, nic nie puka, dosłownie igła. Zapłaciłem za niego 1000
    zł ale jego wartość jest wyższa o 2000 zł i co mam z tym zrobić?
    PRZEKSZTAŁCENIE: {"short_description": "kupiłem samochód"}

    PYTANIE: wczoraj pożyczyłam od swojej przyjaciółki Katarzyny 20 000 zł na remont mieszkania. 
    PRZEKSZTAŁCENIE: {"short_description": "pożyczyłam pieniądze"}

    PYTANIE: 5 dni temu podpisałem umowę z kolegą według której kupuje od niego materiały budowlane za 10000 zł oraz akcje za 5000zł
    PRZEKSZTAŁCENIE: {"short_description": "kupiłem auto, kupiłem materiały budowlane"}

    PYTANIE: \"""" + query + "\"\nPRZEKSZTAŁCENIE: "
    
    return user_prompt_template

  # Test router
  test_web_search = llm_json_mode.invoke([SystemMessage(content=desc_system_prompt)] + [HumanMessage(
    content=get_short_desc_user_prompt("2 dni temu kupiłem mieszkanie, wielkie, przejrzyste, 100m2. Jest przepiękne. Jaki podatek muszę zapłacić? A zapomniałem - do tego kupiłem farbę za 100 zł. Jaki podatek powinienem zapłacić?"),
  )])
  
  v = json.loads(test_web_search.content)["short_description"]
  
  state["fields"].append({"name": "P_23", "value": v})
  
  return {"short_description": v}

def check_values(state):
  desc_system_prompt = """Jesteś ekspertem od decydowania, czy pytanie użytkownika zawiera wszystkie kluczowe informacje. Pytanie zawiera wszystkie kluczowe informacje, jeżeli opisuje przedmiot umowy i ceny/wartości dla każdego z elementów.
  
  Jeśli umowa dotyczy sprzedaży, a użytkownik poda tylko wartość zakupu, dopytaj o wartość rynkową.

  Twoim zadaniem jest zwrócić obiekt json z kluczami "explanation" oraz "is_compleet". Explenation ma zawierać bardzo uprzejme uzasadnienie decyzji pisane tak jak do użytkownika, z prośbą o dodanie informacji lub informacją, że wszystko jest dobrze. Is compleet to true jeśli prompt jest kompletny lub false, jeśli nie jest.

  Przykłady:

  PYTANIE: Kupiłem samochód, jaki podatek powinienem zapłacić?
  ODPOWIEDŹ: {"explanation": "Szanowny Panie/Szanowna Pani, proszę o doprecyzowanie, jaka była cena samochodu?", "is_compleet": false}

  PYTANIE: Wczoraj kupiłem na giełdzie samochodowej Fiata 126p rok prod. 1975, kolor zielony. Przejechane ma 1000000 km, idzie jak przecinak, nic nie stuka, nic nie puka, dosłownie igła. Zapłaciłem za niego 1000 zł ale jego wartość jest wyższa o 2000 zł i co mam z tym zrobić ?
  ODPOWIEDŹ: {"explanation": "Dziękuję, informacje są kompletne", "is_compleet": true}
  """
  
  # Test router
  test_web_search = llm_json_mode.invoke([SystemMessage(content=desc_system_prompt)] + [HumanMessage(
    content=state["query"],
  )])


  return {"is_compleet": json.loads(test_web_search.content)["is_compleet"], "explanation": json.loads(test_web_search.content)["explanation"] + " Proszę o ponowne wprowadzenie całego polecenia, wraz z uzupełnionymi danymi."}


def check_values_decide(state):
  if state["is_compleet"]:
    return "CON"
  return "END"


def get_value(state):
  global info
  print("Analizuję kwotę czynności...")
  info = "Analizuję kwotę czynności..."
  ### Rate from prompt
  # System 
  rate_system_prompt = """Jesteś ekspertem w rozumieniu czynności prawnej podatnika i wyodrębnianiu wartości tej czynności. 

  Jako odpowiedź zwróć obiekt json

  Jeżeli sytuacja dotyczy kupna lub sprzedaży KILKU RZECZY lub SPRZEDAZY KILKU RZECZY, pierwszym kluczem jest klucz "chain_of_thought" - opis obliczeń krok po kroku, które prowadzą do zsumowania wartości wszystkich rzeczy, a drugim kluczem jest "rate" - pojedyncza liczba, wynik sumowania.

  Jeżeli sytuacja dotyczy kupna lub sprzedaży DOKŁADNIE JEDNEJ rzeczy, pierwszym i jedynym kluczem jest "rate" - pojedyncza liczba, stawka znajdująca się w dostarczonej rozmowie podatnikiem.

  Jeżeli sytuacja wymaga przeprowadzenia obliczeń do uzyskania konkretnej wartości, pierwszym kluczem jest klucz "chain_of_thought" - opis obliczeń krok po kroku, które prowadzą do obliczenia wartości, a drugim kluczem jest "rate" - pojedyncza liczba, wynik obliczeń.

  Nie odpowiadaj na pytanie użytkownika, w szczególności nie obliczaj podatku, tylko wyodrębnij wartość czynności.

  UWAGA: 
  Kiedy masz doczynienia z przekazaniem kredytu wzamian za rzecz lub, wartość czynności = wartość rzeczy - wartość kredytu, lub 0 zł jeśli wartość kredytu jest większa.
  Kiedy masz doczynienia z zamianą rzeczy za rzecz wartość czynności = abs(wartość rzeczy 1 - wartość rzeczy 2)
  """

  def get_rate_user_prompt(query):
    user_prompt_template = """
    PYTANIE: Wczoraj kupiłem na giełdzie samochodowej Fiata 126p rok prod. 1975, kolor zielony. Przejechane ma
    1000000 km, idzie jak przecinak, nic nie stuka, nic nie puka, dosłownie igła. Zapłaciłem za niego 1000
    zł ale jego wartość jest wyższa o 2000 zł i co mam z tym zrobić?
    PRZEKSZTAŁCENIE: {"chain_of_thought": "Aby obliczyć końcową wartość samochodu, musimy wziąć pod uwagę wartość, którą za niego zapłaciłeś oraz dodatkową wartość, która podnosi jego cenę. Zaczynamy krok po kroku: 1. Cena zakupu samochodu: To kwota, którą faktycznie zapłaciłeś za samochód, czyli 1000 zł. 2. Dodatkowa wartość: Wiemy, że wartość samochodu jest wyższa o 2000 zł niż cena, którą zapłaciłeś. Oznacza to, że wartość samochodu powinna wzrosnąć o tę kwotę. 3. Końcowa wartość samochodu: Aby obliczyć końcową wartość, musimy dodać cenę zakupu do tej dodatkowej wartości: końcowa wartość = cena zakupu + dodatkowa wartość Końcowa wartość = 1000 zł + 2000 zł Końcowa wartość samochodu to 1000 zł + 2000 zł = 3000 zł.", "rate": 3000}

    PYTANIE: wczoraj pożyczyłam od swojej przyjaciółki Katarzyny 20 000 zł na remont mieszkania. 
    PRZEKSZTAŁCENIE: {"rate": 20000}

    PYTANIE: 5 dni temu podpisałem umowę z kolegą według której kupuje od niego materiały budowlane za 10000 zł oraz akcje za 5000zł
    PRZEKSZTAŁCENIE: {"chain_of_thought": "Aby obliczyć końcową wartość transakcji, musimy dodać wartość wszystkich zakupionych przedmiotów. 1. Wartość materiałów budowlanych wynosi 10000 zł. 2. Wartość akcji wynosi 5000 zł. 3. Wartość mleka: masz 50 kartonów, a każdy kosztuje 10 zł, więc 50 * 10 zł = 500 zł. 4. Teraz dodajemy wszystkie kwoty: 10000 zł + 5000 zł + 500 zł = 15500 zł. Końcowa wartość transakcji wynosi 15500 zł.", rate: 15500}

    PYTANIE: Wczoraj moja kuzynka przekazała mi swój samochód, który ma wartość 40 000 PLN, a w zamian przejęłam jej kredyt na ten samochód w wysokości 20 000 PLN.
    PRZEKSZTAŁCENIE: {"chain_of_thought": "Aby obliczyć wartość czynności prawnej, musimy rozważyć zarówno wartość przekazanego samochodu, jak i kwotę przejętego kredytu. Krok po kroku: 1. Wartość samochodu: Samochód ma wartość 40 000 PLN. 2. Kwota kredytu: Kuzynka przekazuje kredyt na samochód o wartości 20 000 PLN. 3. Obliczenie wartości czynności prawnej: Ponieważ następuje wymiana samochodu na kredyt, aby obliczyć całkowitą wartość transakcji, musimy ODJĄĆ wartość samochodu do kwoty kredytu. Wartość czynności prawnej = wartość samochodu - kwota kredytu Wartość czynności prawnej = 40 000 PLN - 20 000 PLN = 20 000 PLN.", rate: 20000}

    PYTANIE: \"""" + query + "\"\nPZEKSZTAŁCENIE: "
    
    return user_prompt_template

  # Test router
  test_web_search = llm_json_mode.invoke([SystemMessage(content=rate_system_prompt)] + [HumanMessage(
    content=get_rate_user_prompt(state["query"]),
  )])

  return {"tax_value": json.loads(test_web_search.content)['rate']}

def get_type(state):
  global info
  print("Analizuję typ umowy...")
  info = "Analizuję typ umowy..."
  ### Type
  # System 
  type_system_prompt = """Jesteś ekspertem w przypisywaniu rodzaju czynności prawnej do jednej z definicji

  Jako wejście przyjmij pytanie lub sytuację podatnika. Jako odpowiedź zwróć obiekt json {'code': kod_definicji} gdzie kod_definicji zastąp faktycznym kodem definicji.

  DEFINICJE:
  Nazwa definicji: Umowa sprzedaży
  Kod definicji: SPR
  Definicja: Umowa sprzedaży to umowa między dwiema stronami, w której sprzedawca zgadza się sprzedać rzecz lub prawo majątkowe, a kupujący zobowiązuje się zapłacić za nie ustaloną cenę.

  Nazwa definicji: Umowa zamiany
  Kod definicji: ZAM
  Definicja: Umowa zamiany to umowa, w której obie strony zgadzają się wymienić swoje rzeczy lub prawa majątkowe. Każda strona przekazuje jedną rzecz lub prawo, a w zamian otrzymuje od drugiej strony inną rzecz lub prawo

  Nazwa definicji: Umowa pożyczki lub depozytu nieprawidłowego, w tym zwolniona na podstawie art. 9 pkt 10 lit. b ustawy4)
  Kod definicji: POZ
  Definicja: Umowa pożyczki lub depozytu nieprawidłowego to umowa, w której jedna strona (pożyczkodawca) przekazuje drugiej stronie (pożyczkobiorcy) określoną sumę pieniędzy lub rzeczy tego samego rodzaju, a pożyczkobiorca zobowiązuje się oddać tę samą kwotę pieniędzy lub rzeczy w takiej samej ilości i jakości. W przypadku depozytu nieprawidłowego, przechowawca ma prawo korzystać z przekazanych mu pieniędzy lub rzeczy.

  Nazwa definicji: Umowa darowizny w części dotyczącej przejęcia przez obdarowanego długów i ciężarów lub zobowiązań darczyńcy
  Kod definicji: DAR
  Definicja: Umowa darowizny w części dotyczącej przejęcia przez obdarowanego długów i ciężarów darczyńcy to czynność prawna, w której darczyńca, przekazując obdarowanemu określony przedmiot lub wartość, jednocześnie zobowiązuje go do przejęcia jego długów, ciężarów oraz innych zobowiązań, co oznacza, że obdarowany staje się odpowiedzialny za te zobowiązania, umożliwiając tym samym uregulowanie sytuacji finansowej darczyńcy. 

  Nazwa definicji: Ustanowienie odpłatnego użytkowania, w tym użytkowania nieprawidłowego
  Kod definicji: UZY
  Definicja: Użytkowanie to umowa, w której jedna strona (użytkownik) zyskuje prawo do korzystania z cudzej rzeczy lub nieruchomości za wynagrodzeniem, jednocześnie zobowiązując się do przestrzegania ustalonych warunków. W przypadku użytkowania nieprawidłowego, gdy obejmuje ono pieniądze lub inne przedmioty oznaczone tylko co do gatunku, użytkownik staje się ich właścicielem w momencie ich wydania. Po zakończeniu użytkowania jest zobowiązany do zwrotu przedmiotów według przepisów dotyczących zwrotu pożyczki. Niezastosowanie się do warunków umowy może skutkować koniecznością zapłaty odszkodowania lub zwrotu przedmiotu w stanie niezgodnym z umową.
  
  PRZYKŁADY:
  PYTANIE: Postanowiłem wczoraj pożyczyć 5,000 zł od Piotra.
  PRZEKSZTAŁCENIE: {"code": "POZ"}

  PYTANIE: Wczoraj mój przyjaciel przekazał mi swoje mieszkanie warte 100000 PLN, a w zamian przejęłam jego kredyt hipoteczny, który jeszcze spłaca w wysokości 50000
  PRZEKSZTAŁCENIE: {"code": "DAR"}
  """

  def get_type_user_prompt(query):
    user_prompt_template = """
    PYTANIE: \"""" + query + "\"\nPZEKSZTAŁCENIE: "
    
    return user_prompt_template

  test_web_search = llm_json_mode.invoke([SystemMessage(content=type_system_prompt)] + [HumanMessage(
    content=get_type_user_prompt(state["query"]),
  )])

  return json.loads(test_web_search.content)['code']


def sprzedaz(state):
  global info
  print("Decyduję o stawce sprzedaży...")
  info = "Decyduję o stawce sprzedaży..."
  # System 
  decide_system_prompt = """Jesteś ekspertem od decydowania, czy w danym przypadku umowy sprzedaży lub kupna, sprzedawana jest dokładnie jedna rzecz, czy więcej rzeczy.

  Jako odpowiedź zwróć obiekt json z kluczem "is_only_one"

  Jeżeli sprzedaż lub kupno dotyczy WIĘCEJ NIŻ jednej rzeczy, zwróć {"is_only_one": false}

  Jeżeli sprzedaż lub kupno dotyczy DOKŁADNIE jednej rzeczy, zwróć {"is_only_one": true}"""

  def decide_if_only_one():
    shorten = state["short_description"]
    
    return llm_json_mode.invoke([SystemMessage(content=decide_system_prompt)] + [HumanMessage(
      content=shorten,
    )])

  return {"is_only_one": json.loads(decide_if_only_one().content)["is_only_one"]}

def sprzedaz_decyzja(state):
  if state["is_only_one"]:
    return "one"
  
  return "many"

def sprzedaz_many(state):
  state["fields"].append({"name": "P_26", "value": state["tax_value"]})
  state["fields"].append({"name": "P_27", "value": state["tax_value"] * 0.02})

  return {"tax_rate": 0.02}

def sprzedaz_only_one(state):
  global info
  print("Określam stawkę dla jednego przedmiotu...")
  info = "Określam stawkę dla jednego przedmiotu..."
  
  only_one_system_prompts = """
  Jesteś ekspertem od określania, jaki % podaktu powinien być dla konkretnej zamiany.

  Twoje postępowanie powinno przebiegać następująco:

  1) Dla każdej z rzeczy ustal, do której kategorii należy:
  0.01 - Podatek PCC w wysokości 0.01 od praw majątkowych (np. praw do domeny internetowej). Prawa majątkowe to prawa, które przysługują osobie w odniesieniu do rzeczy, przedmiotów materialnych lub dóbr niematerialnych, takich jak prawa autorskie czy patenty. Obejmują one m.in. prawo do korzystania z tych dóbr oraz prawo do ich sprzedaży lub wynajmu.
  0.02 - Podatek PCC w wysokości 0.02 nieruchomości, takich jak mieszkania, domy czy działki, oraz ruchomości, na przykład samochodów, motocykli i innych wartościowych przedmiotów, gdy ich wartość rynkowa przekracza 1 000 zł.
  2) Jeśli obie rzeczy są na stawce 0.01, opodatkowanie zamiany to 0.01. Jeśli obie rzeczy są na stawce 0.02, opodatkowanie zamiany to 0.02. Jeśli rzeczy mają różną stawkę, opodatkowanie wynosi 0.02

  Zwróć jsona {'chain_of_thought': ..., 'tax_value': ...}
  chain_of_thought to proces decyzyjny, który doprowadził cię do takiej decyzji, a tax_value to ustalona wartość podatku.
  """

  def get_tax_value_prompt(query):
    user_prompt_template = """
    PYTANIE: \"""" + query + "\"\nJSON: "
    
    return user_prompt_template

  test_web_search = llm_json_mode.invoke([SystemMessage(content=only_one_system_prompts)] + [HumanMessage(
    content=get_tax_value_prompt(state["query"]),
  )])

  v = json.loads(test_web_search.content)["tax_value"]
  
  if v == 0.01:
    state["fields"].append({"name": "P_24", "value": state["tax_value"]})
    state["fields"].append({"name": "P_25", "value": state["tax_value"] * 0.01})

  if v == 0.02:
    state["fields"].append({"name": "P_26", "value": state["tax_value"]})
    state["fields"].append({"name": "P_27", "value": state["tax_value"] * 0.02})

  return {"tax_rate": json.loads(test_web_search.content)["tax_value"]}

def zamiana(state):
  global info
  print("Określam stawkę dla zamiany...")
  info = "Określam stawkę dla jednego przedmiotu..."
  
  change_system_prompt = """Jesteś ekspertem od określania, jaki podatek powinien być dla konkretnej zamiany.

  Twoje postępowanie powinno przebiegać następująco:

  1) Dla każdej z rzeczy ustal, do której kategorii należy:
  0.01 - Podatek PCC w wysokości 0.01 od praw majątkowych (np. praw do domeny internetowej). Prawa majątkowe to prawa, które przysługują osobie w odniesieniu do rzeczy, przedmiotów materialnych lub dóbr niematerialnych, takich jak prawa autorskie czy patenty. Obejmują one m.in. prawo do korzystania z tych dóbr oraz prawo do ich sprzedaży lub wynajmu.
  0.02 - Podatek PCC w wysokości 0.02 nieruchomości, takich jak mieszkania, domy czy działki, oraz ruchomości, na przykład samochodów, motocykli i innych wartościowych przedmiotów, gdy ich wartość rynkowa przekracza 1 000 zł.
  2) Jeśli obie rzeczy są na stawce 0.01, opodatkowanie zamiany to 0.01. Jeśli obie rzeczy są na stawce 0.02, opodatkowanie zamiany to 0.02. Jeśli rzeczy mają różną stawkę, opodatkowanie wynosi 0.02

  Zwróć jsona {'chain_of_thought': ..., 'tax_value': ...}
  Chain of thought to proces decyzyjny, który doprowadził cię do takiej decyzji, a tax_value to ustalona wartość podatku."""

  test_web_search = llm_json_mode.invoke([SystemMessage(content=change_system_prompt)] + [HumanMessage(
    content="Ania, przekazała mi auto o wartości 5 000 zł, a w zamian przekazałem jej auto o wartości 7 000 zł.",
  )])

  v = json.loads(test_web_search.content)["tax_value"]
  state["fields"].append({"name": "P_28", "value": state["tax_value"]})
  state["fields"].append({"name": "P_29", "value": v})
  state["fields"].append({"name": "P_30", "value": state["tax_value"] * v})
  
  return {"tax_rate": v}

def darowizna_dlug(state):
  state["fields"].append({"name": "P_34", "value": state["tax_value"]})
  state["fields"].append({"name": "P_35", "value": 0.02})
  state["fields"].append({"name": "P_36", "value": state["tax_value"] * 0.02})
  
  return {"tax_rate": 0.02}

def uzytkowanie(state):
  state["fields"].append({"name": "P_37", "value": state["tax_value"]})
  state["fields"].append({"name": "P_38", "value": 0.02})
  state["fields"].append({"name": "P_39", "value": state["tax_value"] * 0.02})
  return {"tax_rate": 0.02}

def pozyczka_fake(state):
  global info
  print("Poprawiam decyzję dla pożyczki...")
  info = "Poprawiam decyzję dla pożyczki..."
  
  mortage_system_prompt = """Jesteś ekspertem od decydowania, czy dana sytuacja jest umową hipoteki, czy umową pożyczki

  Umowa hipoteki - jeśli użyto nieruchomości jako zabezpieczenia (obciążono nieruchomość)
  Umowa pożyczki - jeśli nie użyto nieruchomości jako zabezpieczenia pożyczki.

  Zwróć pojedynczy json {"is_mortgage": true} jeśli umowa jest umową hipoteki lub {"is_mortgage": false} jeśli umowa nie jest umową hipoteki.

  Few shot examples:
  PYTANIE: Tydzień temu zdecydowałem się wziąć kredyt hipoteczny na 100,000 zł, z zabezpieczeniem na mój dom.
  PRZEKSZTAŁCENIE: {"is_mortgage": true}

  PYTANIE: Postanowiłem wczoraj pożyczyć 5,000 zł od Piotra.
  PRZEKSZTAŁCENIE: {"is_mortgage": false}
  """

  def get_is_mortage_prompt(query):
    user_prompt_template = """
    PYTANIE: \"""" + query + "\"\nPZEKSZTAŁCENIE: "
    
    return user_prompt_template

  test_web_search = llm_json_mode.invoke([SystemMessage(content=mortage_system_prompt)] + [HumanMessage(
    content=get_is_mortage_prompt(state["query"]),
  )])

  return {"is_mortgage": json.loads(test_web_search.content)["is_mortgage"]}
  
def pozyczka_fake_decyzja(state):
  if state["is_mortgage"]:
    return "hipoteka"
  return "pozyczka"
  
def pozyczka(state):
  state["fields"].append({"name": "P_31", "value": state["tax_value"]})
  state["fields"].append({"name": "P_32", "value": 0.005})
  state["fields"].append({"name": "P_33", "value": 0.005 * state["tax_value"]})
  return {"tax_rate": 0.005}

def hipoteka(state):
  global info
  print("Ustalam stawkę hipoteki...")
  info = "Ustalam stawkę hipoteki..."
  
  mortage_known_system_prompt = """Jesteś ekspertem od ustalania, czy podstawa, wartość hipoteki jest znana.

  Nie odpowiadaj na pytanie, wyłącznie zwróć pojedynczy json {"is_known": true} jeśli podstawa hipoteki jest znana lub {"is_known": false} jeśli nie jest znana
  """

  test_web_search = llm_json_mode.invoke([SystemMessage(content=mortage_known_system_prompt)] + [HumanMessage(
    content=state["query"],
  )])
  
  return {"is_known": json.loads(test_web_search.content)["is_known"]}
  
def hipoteka_decyzja(state):
  if state["is_known"]:
    return "hipoteka_znana"
  return "hipoteka_nieznana"
  
def hipoteka_nieznana(state):
  state["fields"].append({"name": "P_42", "value": 19})
  state["tax_value"] = 1
  return {"tax_rate": 19}
  
def hipoteka_znana(state):
  state["fields"].append({"name": "P_40", "value": state["tax_value"]})
  state["fields"].append({"name": "P_41", "value": state["tax_value"] * 0.01})
  return {"tax_rate": 0.001}

import operator
from typing_extensions import TypedDict
from typing import TypedDict, List, Dict
from typing import List, Annotated

class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    tax_rate : float # User question
    fields: List[Dict[str, str]]
    tax_value: float
    is_known: bool
    explanation: str
    query: str
    is_mortgage: bool
    is_only_one: bool
    type: str
    short_description: str

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("shorten_description", shorten_description)
workflow.add_node("get_value", get_value)

workflow.add_node("sprzedaz", sprzedaz)
workflow.add_node("sprzedaz_many", sprzedaz_many)
workflow.add_node("sprzedaz_only_one", sprzedaz_only_one)

workflow.add_node("zamiana", zamiana)
# workflow.add_node("check_values", check_values)
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

graph = workflow.compile()

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

class MsgRequest(BaseModel):
    msg_id: str
    message: str
    isFirstFormMessage: bool

@app.post("/getState")
def send_msg():
  return {"info": info}

@app.post("/sendMichalMsg")
def send_msg(msg: MsgRequest):
  if not lock.acquire(blocking=False):
    return {"info": info}
  
  try:
    if msg.isFirstFormMessage:
      MEMORY[msg.msg_id] = []
      brbr = llm_json_mode.invoke([SystemMessage(content=system_prompt)] + [HumanMessage(content="PYTANIE UŻYTKOWNIKA: " + msg.message + "\nJSON:")])
      return json.loads(brbr.content)
    
    res = json.loads(json.dumps(graph.invoke({"query": msg.message, "fields": []})))
    res["finished"] = True
    
    res["final_message"] = "Przeanalizowałem Pana/Pani pytanie. Następujące pola zostały wypełnione:\n" + "\n".join(list(map(lambda x: str(x[0]) + " - " + str(x[1]), res["fields"])))
    return res
  
  finally:
    lock.release()  # Zwalniamy blokadę na koniec operacji