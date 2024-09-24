
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyMuPDFLoader


embeddings_model_name =  "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)


file_path = r'D:\Prakash\Github\Custom-PDF-ChatBot-Using-Llama-LLM-RAG-\STEM_ImaginX_Sequencing_Gene_Linked_to_Obesity.pdf'
def ingest(file_path):
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()

    chunk_size = 1000
    chunk_overlap = 200
    persist_directory = 'physics-chemical'
    embeddings_model_name = 'all-MiniLM-L6-v2'

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    texts = text_splitter.split_documents(pages)
    file_name = 'physics'
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, collection_name=file_name)
    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")
ingest(file_path)



