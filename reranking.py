from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.ranking import Reranker
from langchain_nomic.embeddings import NomicEmbeddings

#Embedding model
embeddings2 = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")

vectorstore2 = FAISS.load_local("my_faiss_store", embeddings=embeddings2, allow_dangerous_deserialization=True)
retriever2 = vectorstore2.as_retriever(k=3)

retrieval_chain = RetrievalQA.from_chain_type(
    llm=None,  # You can specify your LLM here if needed
    chain_type="stuff",
    retriever=vectorstore2.as_retriever(search_kwargs={"k": 50})  # Retrieve top 50
)

# Define your query
query = "Podatek od zakupu żyrafy"

# Retrieve results
results = retrieval_chain(query)

# Results will be in a structured format
chunks = results['result']

# Initialize Reranker with a specific model
reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Example model for reranking
reranker = Reranker.from_pretrained(reranker_model)

# Perform reranking
reranked_results = reranker.rerank(query, chunks)

for i, (chunk, score) in enumerate(reranked_results[:10]):  # Print top 10
    print(f"Rank {i+1}: {chunk} (Score: {score})")