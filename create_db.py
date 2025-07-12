import os
import shutil
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader # carrega os documentos de um diretório
from langchain_chroma import Chroma  # Importa o ChromaDB para armazenamento de vetores
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings


os.environ["OPENAI_API_KEY"] = os.getenv("MY_OPENAI_API_KEY", '')  # Set the OpenAI API key from environment variable
      
CHROMA_PATH = "./chroma" # Path to the ChromaDB database

def load_documents():
    DATA_PATH = "./data"
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {DATA_PATH}")
    return documents

def split_into_chunks(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True  # Ensure the start index is added for each chunk
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]) -> Chroma:
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(),
        persist_directory=CHROMA_PATH
    )

    print(f"saved {len(chunks)} chunks to {CHROMA_PATH}")
    return db


def generate_data_store():
    documents = load_documents()
    chunks = split_into_chunks(documents)
    save_to_chroma(chunks)


if __name__ == "__main__":
    generate_data_store()