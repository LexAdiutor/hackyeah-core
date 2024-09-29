import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from uuid import uuid4
from langchain_community.vectorstores import FAISS

urls = [
    "https://www.podatki.gov.pl/pcc-sd/",
    "https://www.podatki.gov.pl/pcc-sd/rozliczenie-podatku-pcc-od-innych-czynnosci/",
    "https://www.podatki.gov.pl/pcc-sd/rozliczenie-podatku-pcc-od-kupna-samochodu/",
    "https://www.podatki.gov.pl/pcc-sd/rozliczenie-podatku-pcc-od-pozyczki/",
    "https://poradnikprzedsiebiorcy.pl/-co-to-jest-podatek-vat-zasady-jego-dzialania",
    "https://www.pit.pl/pit/",
    "https://www.biznes.gov.pl/pl/portal/00251",
    "https://www.biznes.gov.pl/pl/portal/ou893",
    "https://poradnikprzedsiebiorcy.pl/-kiedy-wystepuje-koniecznosc-zaplaty-pcc",
    "https://allekurier.pl/blog/co-to-jest-clo-i-kiedy-sie-je-placi",
    "https://www.ifirma.pl/blog/podatek-od-czynnosci-cywilnoprawnych-kompendium-wiedzy-w-zakresie-podatku-pcc/"
]

# Load documents
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split documents
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=200, chunk_overlap=40
)

doc_splits = text_splitter.split_documents(docs_list)
doc_splits_ids = [str(uuid4()) for _ in range(len(doc_splits))]

print("Loading embeddings model")

from langchain_nomic.embeddings import NomicEmbeddings
embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")

print("Populating database")

# Initialize FAISS vectorstore with document splits and embeddings
vectorstore = FAISS.from_documents(
    documents=doc_splits,
    embedding=embeddings,
)

vectorstore.save_local("my_faiss_store")
