from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
import gradio as gr
import os

# Caminhos
MODEL_PATH = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
INDEX_DIR = "index"

# Inicializar LLM
llm = LlamaCPP(
    model_path=MODEL_PATH,
    temperature=0.6,
    max_new_tokens=512,
    context_window=4096,
    model_kwargs={"n_gpu_layers": 33, "n_threads": 4},
    verbose=False,
)

# Embeddings
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
# Verificar índice
if not os.path.exists(INDEX_DIR):
    raise FileNotFoundError(f"\nA diretoria de índice '{INDEX_DIR}' não existe.\nCorre primeiro o script 'index_builder.py'.\n")

# Carregar índice
storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine(llm=llm, embed_model=embed_model, response_mode="compact")

# Função de resposta
def responder(pergunta, history=None):
    if not pergunta.strip():
        return "Por favor, escreve uma pergunta válida."
    try:
        resposta = query_engine.query(pergunta)
        return str(resposta)
    except Exception as e:
        return f"Erro ao responder: {e}"

# Interface Gradio
gr.ChatInterface(
    fn=responder,
    title="Chatbot IPVC",
    description="Coloca aqui as tuas perguntas sobre o IPVC com base nos regulamentos oficiais.",
    theme="default"
).launch()
