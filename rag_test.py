import os
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
import time

persist_dir = "./chroma_db"

# Wczytywanie wektorowej bazy danych (embeddingów)
try:
    if os.path.exists(persist_dir):
        print("Wczytywanie istniejącej bazy wektorowej...")
        embeddings = OllamaEmbeddings(model="rjmalagon/gte-qwen2-1.5b-instruct-embed-f16:latest")
        db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        print("Wczytano lokalną bazę wektorową.")
    else:
        print("Baza wektorowa nie istnieje!")
        raise FileNotFoundError(f"Baza wektorowa {persist_dir} nie istnieje!")
except Exception as e:
    raise Exception(f"Błąd podczas wczytywania bazy wektorowej: {str(e)}")

# Testowe zapytanie do chatbota
query = "Na jakim profilu kształcenia jest przedmiot Elementy metodyki badań naukowych?"
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)
context_docs = retriever.invoke(query)
context_text = ""

for idx, d in enumerate(context_docs):
    context_text += '\n' + f'<document {idx + 1}>' + d.page_content + f'</document {idx + 1}>'

prompt = f"""
Na podstawie poniższych dokumentów sylabusa odpowiedz na pytanie:
{query}
Dokumenty:{context_text}
"""
print(prompt)

messages = [
    {"role": "system", "content": "Jesteś asystentem studenta, pomagasz w wyszukiwaniu informacji na temat studiów. Rozmawiasz w języku polskim. Odpowiadaj zwięźle i na temat oraz nie poruszaj dodatkowych kwestii i szczegółów o które NIE zapytał student. W odpowiedzi nie wspominaj o załączonych dokumentach lub na podstawie którego dokumentu odpowiadasz. Jeśli nie znasz odpowiedzi na pytanie to nie zmyślaj i odpowiedz, że nie znalazłeś informacji na ten temat."},
    {"role": "user", "content": f"Context: {prompt}?"}
]
llm = ChatOllama(model="SpeakLeash/bielik-11b-v2.3-instruct-imatrix:IQ4_XS")

print("\nPrzetwarzanie zapytania...")
start = time.time()
response = llm.invoke(messages)
end = time.time()
print("\n\nOdpowiedź modelu:")
print(response.content)
print(f"\nCzas przetwarzania: {end - start} sek.")