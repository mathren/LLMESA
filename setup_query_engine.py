import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
)
from llama_index.core.node_parser import (
    SentenceSplitter,
    CodeSplitter,
    MarkdownNodeParser,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import time
from typing import List
from pathlib import Path

def create_parsers():
    """Create and return all the different parsers"""
    fortran_parser = CodeSplitter(
        language="fortran",
        chunk_lines=50,  # Lines per chunk
        chunk_lines_overlap=5,  # Overlap between chunks
        max_chars=2000,
    )

    rst_parser = MarkdownNodeParser()  # RST is similar to Markdown

    # Custom parser for .defaults files
    defaults_parser = SentenceSplitter(
        chunk_size=800,  # Mid-sized chunks
        chunk_overlap=100,
        separator="! ::",  # Custom separator
    )

    # Single line parser for .list files
    list_parser = SentenceSplitter(
        chunk_size=200,  # Small chunks to keep lines separate
        chunk_overlap=0,  # No overlap needed
        separator="\n",
    )

    # Default parser for other files
    default_parser = SentenceSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    )

    return {
        'fortran': fortran_parser,
        'rst': rst_parser,
        'defaults': defaults_parser,
        'list': list_parser,
        'default': default_parser
    }


def get_parser_for_extension(file_ext, parsers):
    """Return the appropriate parser for a given file extension"""
    if file_ext in ['.f', '.f90', '.inc', '.dek']:
        return parsers['fortran'], 'Fortran-aware'
    elif file_ext == '.list':
        return parsers['list'], 'Line-by-line'
    elif file_ext == '.defaults':
        return parsers['defaults'], 'Defaults-aware'
    elif file_ext == '.rst':
        return parsers['rst'], 'RST-aware'
    else:
        return parsers['default'], 'Default'


def process_documents_with_custom_chunking(documents: List[Document]) -> List:
    """Process documents with type-specific chunking strategies"""
    parsers = create_parsers()
    processed_nodes = []

    for doc in documents:
        # Get file extension from metadata or document ID
        file_path = doc.metadata.get('file_path', '') or doc.doc_id or ''
        file_ext = Path(file_path).suffix.lower()

        print(f"Processing {file_path} with extension {file_ext}")

        # Get appropriate parser and description
        parser, description = get_parser_for_extension(file_ext, parsers)

        # Apply chunking strategy
        nodes = parser.get_nodes_from_documents([doc])
        print(f"  {description} chunking: {len(nodes)} chunks")

        processed_nodes.extend(nodes)

    return processed_nodes


def create_index_with_custom_chunking(mesa_dir: str, persist_dir: str,
                                      num_workers=10):
    """Create index with custom chunking for different file types"""
    # Load documents
    documents = SimpleDirectoryReader(
        mesa_dir,
        filename_as_id=True,
        recursive=True,
        required_exts=[
            ".defaults", ".list", ".rst",
            ".f90", ".f", ".inc", ".dek"
        ],
    ).load_data(show_progress=True, num_workers=num_workers)

    # Process with custom chunking
    nodes = process_documents_with_custom_chunking(documents)

    # Create index from processed nodes
    print(f"Creating index from {len(nodes)} total nodes...")
    index = VectorStoreIndex(nodes, show_progress=True)
    index.storage_context.persist(persist_dir=persist_dir)

    return index


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
        - Keep answers short and to the point
        - do not anticipate further questions
        - do not provide information not strictly and directly relevant to the question
        - Use ONLY the provided context information to answer questions
        - Ignore anything not in the context information
        - If there are multiple possible answers, provide all of them clearly distinguishing them
        - If information is not in the context, clearly state that and refuse to answer
        - For configuration parameters, include the file type and location at the end of the answer
        - For code references, mention the source file and line number at the end of the answer"""+
                                  f"{query}"
                                  )
    else:
        template = PromptTemplate(template)
    t_start = time.time()
    response = query_engine.query(template.format(query_str=query))
    t_end = time.time()
    response_time = t_end-t_start
    return response, response_time


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


    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    Settings.llm = Ollama(model="starcoder2", request_timeout=600.0)
    # Settings.chunk_size = 1200
    # Settings.chunk_overlap = 300
    Settings.temperature = 0.0
    Settings.top_p = 0.01
    # Settings.context_window = 4096

    # store index so we don't have to recreate it every time
    PERSIST_DIR = "./storage-nomic-embed-text-custom-chunking"

    if not os.path.exists(PERSIST_DIR):
        print("Creating persistent storage with custom chunking!")
        index = create_index_with_custom_chunking(MESA_DIR, PERSIST_DIR)
    else:
        print("Index read from persistent storage!")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context, show_progress=True)
    # if not os.path.exists(PERSIST_DIR):
    #     print("creating persistent storage!")
    #     documents = SimpleDirectoryReader(
    #         MESA_DIR,
    #         filename_as_id=True,
    #         recursive=True,
    #         required_exts=[
    #             ".defaults", ".list",  ".rst",
    #             ".f90", ".f", ".inc", ".dek"
    #         ],
    #     ).load_data(show_progress=True, num_workers=10)
    #     index = VectorStoreIndex.from_documents(documents, show_progress=True)
    #     index.storage_context.persist(persist_dir=PERSIST_DIR)
    # else:
    #     print("index read from persistent storage!")
    #     storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    #     index = load_index_from_storage(storage_context, show_progress=True)

    # setup query_engine
    query_engine = index.as_query_engine(llm=Settings.llm)

    # test
    query = "what is MESA?"
    response, response_time = query_LLMESA(query_engine, query=query)
    print(query, response_time)
    print(f"{response}")
    query = "what is `time_delta_coeff`?"
    response, response_time = query_LLMESA(query_engine, query=query)
    print(query, response_time)
    print(f"{response}")

    print("Example usage:")
    print("   response = query_LLMESA(query_engine, 'your question goes here')")
    print("   print(response)")

    # # Interactive mode
    print()
    print("Entering interactive mode (type 'quit' to exit):")
    while True:
        user_query = input("What's your question about MESA?\n").strip()
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
        if user_query:
            response, time = query_LLMESA(query_engine, user_query)
            print(f"{response}")
    print("done")
    exit()
