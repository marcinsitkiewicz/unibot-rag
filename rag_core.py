import os
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

class UniBotRAG:
    def __init__(self):
        persist_dir = "./chroma_db"

        # Przygotowanie modeli
        self.embeddings = OllamaEmbeddings(model="rjmalagon/gte-qwen2-1.5b-instruct-embed-f16:latest")
        self.llm = ChatOllama(model="SpeakLeash/bielik-11b-v2.3-instruct-imatrix:IQ4_XS")

        # Sprawdzenie czy na pewno istnieje już baza wektorowa
        if not os.path.exists(persist_dir):
            raise FileNotFoundError(f"Baza wektorowa {persist_dir} nie istnieje!")

        # Przygotowanie bazy i retrievera
        self.db = Chroma(persist_directory=persist_dir, embedding_function=self.embeddings)
        self.retriever = self.db.as_retriever(search_kwargs={"k": 3})

    # Zwraca odpowiedź na pytanie na podstawie sylabusa
    def ask(self, query: str) -> str:

        # Wyciąganie najbardziej odpowiadających dokumentów (fragmentów tekstu) z bazy i połączenie ich
        context_docs = self.retriever.invoke(query)
        context_text = ""
        for idx, d in enumerate(context_docs):
            context_text += f"\n<fragment {idx+1}>\n{d.page_content}\n</fragment {idx+1}>"

        # Budowa prompta
        prompt = f"""
        Na podstawie poniższych fragmentów sylabusa odpowiedz na pytanie użytkownika:
        {query}

        Fragmenty kontekstu:
        {context_text}

        Jeśli nie ma informacji w powyższych dokumentach, odpowiedz: "Nie znalazłem informacji na ten temat."
        """

        # Umieszczenie prompta w strukturze 'messages' w style openai (klucze 'role' i 'content')
        messages = [
            {"role": "system", "content": "Jesteś asystentem studenta, pomagasz w wyszukiwaniu informacji na temat studiów. Rozmawiasz w języku polskim. Odpowiadaj zwięźle, konkretnie i na temat oraz nie poruszaj dodatkowych kwestii i szczegółów o które NIE zapytał student - bez wstępu, wstępnych fraz, komentarzy ani tłumaczenia kontekstu. W odpowiedzi nie wspominaj o załączonych dokumentach lub na podstawie którego dokumentu odpowiadasz. Jeśli nie znasz odpowiedzi na pytanie to nie zmyślaj i odpowiedz, że nie znalazłeś informacji na ten temat."},
            {"role": "user", "content": prompt}
        ]

        # Wykonanie zapytania
        response = self.llm.invoke(messages)
        return response.content

    # Zwraca odpowiedź na pytanie tylko na podstawie wiedzy modelu, ale z promptem systemowym
    def ask_without_rag(self, query: str) -> str:
        # Budowa prompta
        prompt = query

        # Umieszczenie prompta w strukturze 'messages' w style openai (klucze 'role' i 'content')
        messages = [
            {"role": "system", "content": "Jesteś asystentem studenta, pomagasz w wyszukiwaniu informacji na temat studiów. Rozmawiasz w języku polskim. Odpowiadaj zwięźle, konkretnie i na temat oraz nie poruszaj dodatkowych kwestii i szczegółów o które NIE zapytał student - bez wstępu, wstępnych fraz, komentarzy ani tłumaczenia kontekstu. W odpowiedzi nie wspominaj o załączonych dokumentach lub na podstawie którego dokumentu odpowiadasz. Jeśli nie znasz odpowiedzi na pytanie to nie zmyślaj i odpowiedz, że nie znalazłeś informacji na ten temat."},
            {"role": "user", "content": prompt}
        ]

        # Wykonanie zapytania
        response = self.llm.invoke(messages)
        return response.content

    # Zwraca odpowiedź na pytanie tylko na podstawie wiedzy modelu i bez prompta systemowego
    def ask_without_rag_and_system_prompt(self, query: str) -> str:
        # Budowa prompta
        prompt = query

        # Umieszczenie prompta w strukturze 'messages' w style openai (klucze 'role' i 'content')
        messages = [
            {"role": "user", "content": prompt}
        ]

        # Wykonanie zapytania
        response = self.llm.invoke(messages)
        return response.content