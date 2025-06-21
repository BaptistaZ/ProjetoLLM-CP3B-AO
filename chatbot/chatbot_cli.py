from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

INDEX_PATH = "index"
storage_context = StorageContext.from_defaults(persist_dir=INDEX_PATH)

index = load_index_from_storage(storage_context)

llm = LlamaCPP(
    model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.1,
    max_new_tokens=512,
    context_window=3900,
    model_kwargs={"n_gpu_layers": 1},
    verbose=False,
)

query_engine = index.as_query_engine(llm=llm)

print("CLI chatbot ready! Write your question below (or 'exit'):\n")

while True:
    pergunta = input("Question: ")
    if pergunta.lower() in ["sair", "exit", "quit"]:
        break
    resposta = query_engine.query(pergunta)
    print(f"\nResponse:\n{resposta}\n")
