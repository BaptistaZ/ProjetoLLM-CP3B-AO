from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from loader_local import load_all_pdfs_recursively
from pathlib import Path

llm = LlamaCPP(
    model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.6
)

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

Settings.llm = llm
Settings.embed_model = embed_model

def build_index():
    print(" Step 1: Loading documents from PDFs...")
    raw_docs = load_all_pdfs_recursively("data")
    print(f" Step 2: Uploaded documents: {len(raw_docs)}")

    documents = [
        Document(text=doc["text"], metadata=doc["metadata"])
        for doc in raw_docs
    ]
    print(" Step 3: Converted raw_docs to Document objects")

    print(" Step 4: Parsing documents into chunks...")
    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(documents)

    print(" Step 5: Constructing the vector index...")
    index = VectorStoreIndex(nodes)

    persist_path = (Path(__file__).parent.parent / "index").resolve()
    persist_path.mkdir(parents=True, exist_ok=True)

    print(f" Step 6: Persisting index to disk at {persist_path}...")
    index.storage_context.persist(persist_dir=str(persist_path))

    print(" Step 7: Index saved successfully!")

    return index

if __name__ == "__main__":
    build_index()
