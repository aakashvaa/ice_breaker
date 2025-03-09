import os
from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.base import Embeddings
import ollama

from consts import INDEX_NAME

# Custom class to use Ollama for embeddings
class OllamaEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [ollama.embeddings(model="nomic-embed-text", prompt=text)['embedding'] for text in texts]

    def embed_query(self, text):
        return ollama.embeddings(model="nomic-embed-text", prompt=text)['embedding']

embeddings = OllamaEmbeddings()

def ingest_docs():
    loader = ReadTheDocsLoader("docs/api.python.langchain.com/en/latest")
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    
    for doc in documents:
        new_url = doc.metadata["source"].replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("**** Loading to vectorstore done ***")

if __name__ == "__main__":
    ingest_docs()