""" test llama-index + MESA """

import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# setup MESA DIR
MESA_DIR = "/home/math/Documents/Research/codes/mesa/mesa-r24.03.1/"

# check if storage already exists
PERSIST_DIR = "./storage"

if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader(MESA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
print("index done")
# setup query engine
query_engine = index.as_query_engine()
print("query_engine setup")
response = query_engine.query("What is varcontrol_target?")
print(response)
