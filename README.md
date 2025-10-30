# UniBot – Asystent Studenta SAN

UniBot to lokalny chatbot z wykorzystaniem **RAG (Retrieval-Augmented Generation)** i modeli LLM działających przez **Ollama**, zaprojektowany do wspomagania studentów w przeglądaniu sylabusów i programów studiów.

---

## 📝 Opis projektu

Celem projektu jest umożliwienie użytkownikowi zadawania pytań dotyczących programu studiów, a chatbot w czasie rzeczywistym wyszukuje odpowiednie fragmenty sylabusa w lokalnej bazie embeddingów i generuje precyzyjne odpowiedzi.

Dzięki RAG system może:
- wyszukiwać odpowiednie informacje w dokumentach,
- udzielać odpowiedzi w języku polskim,
- unikać „halucynacji” typowych dla modeli LLM.

---

## ⚡ Funkcje

- Interaktywny czat w przeglądarce z historią konwersacji,
- Wyszukiwanie kontekstowe w dokumentach PDF z sylabusem,
- Lokalna baza embeddingów z wykorzystaniem Chroma + Ollama embeddings,
- Obsługa wielu modeli: osobny model do embeddingów i do generowania odpowiedzi,
- Wersja webowa oparta na **Gradio**.

---

## 🧠 Modele embeddingowe i LLM

W projekcie UniBot wykorzystywane są następujące modele:

1. **Model do embeddingów (tworzenie bazy wektorowej):**  
   `rjmalagon/gte-qwen2-1.5b-instruct-embed-f16:latest`  
   - Służy do zamiany fragmentów PDF sylabusa na wektory w bazie Chroma.
   - Zapewnia wysoką jakość wyszukiwania kontekstowego.
   - [Link do modelu GTE-Qwen2](https://ollama.com/rjmalagon/gte-qwen2-1.5b-instruct-embed-f16)

2. **Model do generowania odpowiedzi (LLM):**  
   `SpeakLeash/bielik-11b-v2.3-instruct-imatrix:IQ4_XS`  
   - Przetwarza zapytania użytkownika wraz z pobranym kontekstem z bazy embeddingów.
   - Generuje precyzyjne odpowiedzi w języku polskim, minimalizując halucynacje.
   - [Link do modelu Bielik](https://ollama.com/SpeakLeash/bielik-11b-v2.3-instruct-imatrix)

---

## 🛠️ Instalacja i uruchomienie

1. Sklonuj repozytorium:  
```bash
git clone <adres_repozytorium>
cd <nazwa_folderu>
```

2. Utwórz wirtualne środowisko:  
```bash
python -m venv venv
```

3. Aktywuj środowisko:  
- Windows:  
```bash
venv\Scripts\activate
```
- Linux / macOS:  
```bash
source venv/bin/activate
```

4. Zainstaluj wymagane pakiety:  
```bash
pip install -r requirements.txt
```

5. Uruchom serwer Ollama (w osobnym terminalu):  
```bash
ollama serve
```

6. Utwórz lub wczytaj bazę embeddingów: 
```bash
python build_db.py
```

7. Uruchom aplikację webową:  
```bash
python web_app.py
```

8. Otwórz przeglądarkę pod adresem:  
```
http://localhost:7860
```

---

## 📂 Struktura projektu

```
UniBot/
│
├─ rag_core.py           # Logika RAG (budowanie embeddingów i zapytania)
├─ rag_test.py           # Testowy skrypt RAG (bez wykorzystania aplikacji webowej)
├─ build_db.py           # Skrypt do tworzenia bazy embeddingów
├─ web_app.py            # Interfejs webowy w Gradio
├─ data/                 # Folder z plikami sylabusa
│ ├─ sylabus.pdf         # Dokument PDF z programem studiów
│ └─ sylabus_short.pdf   # Skrócony sylabus do testów
├─ requirements.txt      # Lista pakietów Python
├─ README.md             # Ten plik
└─ .gitignore            # Pliki/foldery ignorowane przez Git
```

**Uwaga:** Nie dodawaj folderu `venv/` ani bazy Chroma (`chroma/`) do repozytorium.

---

## 📚 Wymagania

- Python 3.11+
- Ollama (lokalny serwer LLM)
- Pakiety Python z `requirements.txt`
- System operacyjny: Windows / Linux / macOS

---

## ⚙️ Uwagi

- Baza embeddingów musi być stworzona przed pierwszym użyciem (`build_db.py`).
- Chatbot działa lokalnie, dane nie są przesyłane do chmury.
- W przypadku dużych PDF-ów proces tworzenia embeddingów może chwilę potrwać.

---

## 📖 Bibliografia

- [LangChain Documentation](https://docs.langchain.com/oss/python/langchain/overview)
- [Chroma DB](https://docs.trychroma.com/docs/overview/introduction)
- [Ollama](https://docs.ollama.com/)
- [Gradio](https://www.gradio.app/docs/python-client/introduction)
