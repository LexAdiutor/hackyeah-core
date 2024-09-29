from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_nomic.embeddings import NomicEmbeddings
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


def rerank(query):
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    embeddings2 = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")

    vectorstore2 = FAISS.load_local("my_faiss_store", embeddings=embeddings2, allow_dangerous_deserialization=True)

    results = vectorstore2.similarity_search(query, k=50)

    chunks = [result.page_content for result in results]

    print("Started reranking...")

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def rerank_chunks(query, chunks):
        inputs = [query + " [SEP] " + chunk for chunk in chunks]
        tokenized_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**tokenized_inputs)
            scores = outputs.logits.squeeze().cpu().tolist() 

        reranked_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), reverse=True)]
        return reranked_chunks

    reranked_chunks = rerank_chunks(query, chunks)

    for i, chunk in enumerate(reranked_chunks[:10]):
        print(f"Rank {i+1}: {chunk}")

    return reranked_chunks[:6]

rerank("Czym się PIT od podatku CIT?")