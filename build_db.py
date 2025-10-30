import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Wczytanie PDF
persist_dir = "./chroma_db"
pdf_path = "./data/sylabus.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# Tworzenie wektorowej bazy danych (embeddingów)
if os.path.exists(persist_dir):
    print("Baza wektorowa już istnieje.")
else:
    # Połączenie stron w jeden tekst i wstępne czyszczenie
    text = "\n".join([page.page_content for page in pages])
    text = text.replace("\n", " ").replace("  ", " ")
    # Podział na fragmenty
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
    print(f"Załadowano {len(docs)} fragmentów tekstu z pliku {pdf_path}")

    # Awaryjne przycięcie strony - nie powinienem tego robić, ale...
    max_length = 3000
    for i, doc in enumerate(docs):
        if len(doc.page_content) > max_length:
            print(f"Fragment {i} ma {len(doc.page_content)} znaków - przycinam awaryjnie do {max_length}.")
            doc.page_content = doc.page_content[:max_length]

    print("Tworzenie nowej bazy wektorowej (embeddingi)...")
    embeddings = OllamaEmbeddings(model="rjmalagon/gte-qwen2-1.5b-instruct-embed-f16:latest")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
    print("Utworzono lokalną bazę wektorową.")