import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


def query_LLMESA(query_engine, query: str, template='') -> 'str':
    if not template:
        template = PromptTemplate("""
        You are a MESA (Modules for
        Experiments in Stellar Astrophysics) expert assistant.

        MESA is a stellar evolution code written in Fortran 90 for
        simulating stellar evolution problems in spherical symmetry
        (1D) with support for rotation and binary systems. It uses
        configuration files called 'inlist*' and can be extended with
        custom functions.

        Instructions:
        - Keep answers short and to the point, do not anticipate further questions or provide information not strictly and directly relevant
        - Use ONLY the provided context information to answer questions, do not rely on anything not in the context information
        - If information is not in the context, clearly state that and refuse to answer
        - For configuration parameters, include the file type and location at the end of the answer
        - For code references, mention the source file and line number at the end of the answer"""+
                                  f"{query}"
                                  )
    else:
        template = PromptTemplate(template)
    response = query_engine.query(template.format(query_str=query))
    return response


if __name__ == "__main__":
    try:
        # read MESA_DIR from environment
        MESA_DIR = os.environ['MESA_DIR']
        print("Using MESA installation at", MESA_DIR)
    except KeyError:
        print("The environment variable MESA_DIR does not appear to be set!")
        print("Did you do:")
        print("export MESA_DIR=/path/to/your/MESA/installation/ ?")
        exit()

    # setup nomic embedding model
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")  # llama3:70b
    # ollama
    Settings.llm = Ollama(model="starcoder2", request_timeout=600.0) # llama3:70b
    # chuking
    Settings.chunk_size = 1200
    Settings.chunk_overlap = 200
    Settings.temperature = 0.0
    Settings.top_p = 0.9

    # store index so we don't have to recreate it every time
    PERSIST_DIR = "./storage-nomic-embed-text"

    if not os.path.exists(PERSIST_DIR):
        print("creating persistent storage!")
        documents = SimpleDirectoryReader(
            MESA_DIR,
            filename_as_id=True,
            recursive=True,
            required_exts=[
                ".defaults", ".list",  ".rst",
                ".f90", ".f", ".inc", ".dek"
            ],
            #required_exts=[".defaults",  ".list", ".rst"],
        ).load_data(show_progress=True, num_workers=10)
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        print("index read from persistent storage!")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context, show_progress=True)

    # setup query_engine
    query_engine = index.as_query_engine(llm=Settings.llm)

    # test
    response = query_LLMESA(query_engine, query="what is MESA?", template='')
    print("what is MESA?")
    print(response)
    response = query_LLMESA(query_engine, query="what is `time_delta_coeff`?")
    print("what is `time_delta_coeff`?")
    print(response)

    print("Example usage:")
    print("   response = query_LLMESA(query_engine, 'your question goes here')")
    print("   print(response)")

    # # Interactive mode
    # print("\nEntering interactive mode (type 'quit' to exit):")
    # while True:
    #     user_query = input("\n").strip()
    #     if user_query.lower() in ['quit', 'exit', 'q']:
    #         break
    #     if user_query:
    #         response = query_LLMESA(query_engine, user_query)
    #         print(f"\nResponse: {response}")
    # print("done")
    # exit()
