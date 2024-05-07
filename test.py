""" test llama-index + MESA """

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

# set number of cores
try:
    OMP_NUM_THREADS = int(os.environ['OMP_NUM_THREADS'])
except KeyError:
    print("The environment variable OMP_NUM_THREADS does not appear to be set!")
    print("Falling back on one core")
    OMP_NUM_THREADS = 1

# llama3 seems incapable of coping with the whole MESA_DIR, try copying relevant stuff in DATA_DIR
# copy documentations and code in the data folder
DATA_DIR = "./data"

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
    defaults = glob.glob(MESA_DIR+'/*/defaults/*')
    for f in defaults:
        os.system("cp "+f+" "+DATA_DIR)
    code = glob.glob(MESA_DIR+"/*/*/*.f")+glob.glob(MESA_DIR+"/*/*/*.f90")
    for f in code:
        os.system("cp "+f+" "+DATA_DIR)
    print("done creating", DATA_DIR)
else:
    print(DATA_DIR, "exists already, going to use it!")

# setup nomic embedding model
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
# ollama
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

# store index so we don't have to recreate it every time
PERSIST_DIR = "./storage"

if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader(DATA_DIR).load_data(show_progress=True, num_workers=OMP_NUM_THREADS)
    # documents = SimpleDirectoryReader(MESA_DIR).load_data(show_progress=True, num_workers=OMP_NUM_THREADS)
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print("creating persistent storage!")
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    print("index read from persistent storage!")

# setup query_engine
query_engine = index.as_query_engine()

print("Example usage")
print()
print("response = query_engine.query('your question in human understandable language goes here')")
print("print(response)")

# response = query_engine.query("How can I control the time resolution in a simulation run with MESA? I want to change the time steps in the simulations not just have more output as a function of time")
# print(response)
