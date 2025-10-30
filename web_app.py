import gradio as gr
from rag_core import UniBotRAG

# Inicjalizacja RAG przy starcie
bot = UniBotRAG()

def chat_fn(message, history):
    return bot.ask(message)

# UI w Gradio
with gr.Blocks(title="UniBot - Asystent Studenta SAN", theme=gr.themes.Origin(), fill_height=True) as demo:
    gr.Markdown("## UniBot - Asystent Studenta SAN\nZadaj pytanie dotyczące programu studiów (Informatyka II stopnia 2024/25):")
    gr.ChatInterface(fn=chat_fn, type="messages", fill_height=True)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)