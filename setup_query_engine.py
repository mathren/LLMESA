import os
from llama_index.core import (VectorStoreIndex,
                              SimpleDirectoryReader,
                              Settings,
                              StorageContext,
                              load_index_from_storage,
                              PromptTemplate
                              )
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


def query_LLMESA(query_engine, query: str, template = '') -> 'str':
    if not template:
        template = PromptTemplate(
            """We have provided context information below. \n
            \n ---------------------\n
            MESA is a stellar evolution code written in Fortran 90 \n
            It can be used to simulate many stellar evolution problem \n
            in one-dimension (that is in spherical symmetry), although it can \n
            also handle rotation in `shellular approximation`. \n
            It can also evolve two stars orbiting each other in a binary system \n
            It is configured with input files called `inlists`, which can be multiple and nested, \n
            and it can be extended with customized functions in `run_star_extras.f90` and/or \n
            `run_binary_extras.f90`. You have been given all the code and input files, and \n
            you are acting as a helper to understand and setup this code. \n
            \n---------------------\n
            Given this information, please answer the question: {query_str}\n
            """)
    else:
        template = PromptTemplate(template)
    response = query_engine.query(template.format(query_str=query))
    return response

if __name__ == "__main__":
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
            # required_exts=[".f90", ".f", ".defaults", ".list", ".inc", ".dek"],
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

    # test
    # response = query_LLMESA(query_engine, "what is MESA?")
    # print(response)

    print("Example usage:")
    print("   response = query_LLMESA(query_engine, 'your question in natural language goes here')")
    print("   print(response)")
