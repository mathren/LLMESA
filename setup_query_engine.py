import os
import glob
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# read MESA_DIR from environment
try:
    MESA_DIR = os.environ['MESA_DIR']
    print("Using MESA installation at", MESA_DIR)
except KeyError:
    print("The environment variable MESA_DIR does not appear to be set!")
    print("Did you do:")
    print("export MESA_DIR=/path/to/your/MESA/installation/ ?")
    exit()

# setup nomic embedding model
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
# ollama
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

# store index so we don't have to recreate it every time
PERSIST_DIR = "./storage"

if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader(
        MESA_DIR,
        filename_as_id=True,
        recursive=True,
        required_exts=[".f90", ".f", ".defaults", ".list", ".inc", ".dek"],
    ).load_data(show_progress=True)
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print("creating persistent storage!")
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context, show_progress=True)
    print("index read from persistent storage!")

# setup query_engine
query_engine = index.as_query_engine()

print("Example usage:")
print("   response = query_engine.query('your question in human understandable language goes here')")
print("   print(response)")
