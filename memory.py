from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import os
from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader


load_dotenv(find_dotenv())

# Step 1: Load PDFs from Data Folder
DATA_PATH = "data/"


def load_pdf_documents(data_path):

    try:
        loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyMuPDFLoader)
        documents = loader.load()
        if not documents:
            raise ValueError("No PDF files found in the specified directory.")
        print(f"Loaded {len(documents)} pages from PDF files.")
        return documents
    except Exception as e:
        print(f"Error loading PDFs: {e}")
        return []


documents = load_pdf_documents(DATA_PATH)

if not documents:
    print("No documents loaded. Exiting.")
    exit()

# Step 2: Split Documents into Chunks
def create_text_chunks(documents, chunk_size=500, chunk_overlap=50):
    """Splits documents into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    text_chunks = text_splitter.split_documents(documents)
    print(f"Generated {len(text_chunks)} text chunks.")
    return text_chunks


text_chunks = create_text_chunks(documents)

# Step 3: Load Embedding Model
def load_embedding_model():

    try:
        model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("Loaded Hugging Face embedding model successfully.")
        return model
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        exit()


embedding_model = load_embedding_model()

# Step 4: Store Embeddings in FAISS
DB_FAISS_PATH = os.path.abspath("vectorstore/db_faiss")

def store_embeddings(text_chunks, embedding_model, db_path):

    try:
        os.makedirs(db_path, exist_ok=True)
        db = FAISS.from_documents(text_chunks, embedding_model)
        db.save_local(db_path)
        print(f"Embeddings successfully stored in FAISS at {db_path}")
    except Exception as e:
        print(f"Error storing embeddings in FAISS: {e}")
        exit()


store_embeddings(text_chunks, embedding_model, DB_FAISS_PATH)



