import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from duckduckgo_search import DDGS

load_dotenv()
API_KEY = os.getenv("API")

def search_duckduckgo(query):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=1):
                results.append(r['body'])
        return "\n".join(results)
    except Exception as e:
        print(f"Error during DuckDuckGo search: {e}")   
        return ""
def rag_chain(url):
    # Loads data
    # Creates Embeddings
    # vectorstore
    # retriever
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        if not docs:
            return None
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(texts, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        print(f"Error in rag_chain: {e}")
        return None
