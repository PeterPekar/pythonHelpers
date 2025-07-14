import argparse
import json
import os
import sys
from typing import Optional, List, Dict, Any, Generator, cast

try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.models import PointStruct, ScoredPoint
except ImportError:
    print("Qdrant client not installed. Please install with 'pip install qdrant-client'")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("SentenceTransformers not installed. Please install with 'pip install sentence-transformers torch'")
    sys.exit(1)

try:
    import ollama
except ImportError:
    print("Ollama library not installed. Please install with 'pip install ollama'")
    sys.exit(1)

# --- Default Configuration ---
DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_TOP_K = 3
DEFAULT_QDRANT_TIMEOUT = 30

# --- Core Functions ---

def load_embedding_model(model_name: str) -> SentenceTransformer:
    print(f"INFO: Attempting to load embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        print(f"INFO: Embedding model '{model_name}' loaded.")
        return model
    except Exception as e:
        print(f"ERROR: Failed to load embedding model '{model_name}': {e}")
        raise

def init_qdrant_client(url: Optional[str] = None, api_key: Optional[str] = None,
                       path: Optional[str] = None, prefer_grpc: bool = False,
                       timeout_seconds: int = DEFAULT_QDRANT_TIMEOUT) -> QdrantClient:
    client_args: Dict[str, Any] = {}
    if path:
        print(f"INFO: Initializing Qdrant client with local path: {path}")
        client_args = {"path": path}
    elif url:
        print(f"INFO: Initializing Qdrant client with URL: {url} (GRPC: {prefer_grpc})")
        client_args = {"url": url, "prefer_grpc": prefer_grpc, "timeout": timeout_seconds}
        if api_key: client_args["api_key"] = api_key
    else:
        raise ValueError("Qdrant URL or path is required.")
    try:
        client = QdrantClient(**client_args)
        print("INFO: Qdrant client initialized and connected.")
        return client
    except Exception as e:
        print(f"ERROR: Failed to initialize Qdrant client: {e}")
        raise

def embed_query(model: SentenceTransformer, query: str) -> List[float]:
    print(f"INFO: Embedding query: \"{query[:100]}{'...' if len(query)>100 else ''}\"")
    try:
        return model.encode(query).tolist()
    except Exception as e:
        print(f"ERROR: Failed to embed query: {e}")
        raise

def search_qdrant(client: QdrantClient, collection_name: str, query_vector: List[float], top_k: int,
                  vector_name: Optional[str] = None) -> List[ScoredPoint]:
    print(f"INFO: Searching Qdrant collection '{collection_name}' for top {top_k} results.")

    search_params = models.SearchParams(hnsw_ef=128, exact=False)
    query_vector_arg: List[float] | tuple[str, List[float]] = (vector_name, query_vector) if vector_name else query_vector

    try:
        results = client.search(
            collection_name=collection_name, query_vector=query_vector_arg, # type: ignore
            limit=top_k,
            with_payload=True, search_params=search_params
        )
        print(f"INFO: Found {len(results)} results from Qdrant.")

        return results
    except Exception as e:
        print(f"ERROR: Failed to search Qdrant: {e}")
        raise

def retrieve_windows_from_chunks(client: QdrantClient, windows_collection_name: str, chunk_results: List[ScoredPoint]) -> List[Dict[str, Any]]:
    """
    Retrieves the parent windows for a list of chunk search results.
    """
    window_ids = [chunk.payload['parent_window_id'] for chunk in chunk_results if chunk.payload and 'parent_window_id' in chunk.payload]
    if not window_ids:
        return []

    # Use a set to get unique window IDs
    unique_window_ids = list(set(window_ids))
    
    retrieved_windows = client.retrieve(
        collection_name=windows_collection_name,
        ids=unique_window_ids,
        with_payload=True
    )
    
    # Convert retrieved points to a list of dictionaries
    return [window.payload for window in retrieved_windows]

def format_context_from_windows(windows: List[Dict[str, Any]]) -> str:
    if not windows:
        return "No relevant context found in the knowledge base."
    
    context_parts = []
    for window in windows:
        heading = window.get("heading", "N/A")
        content = window.get("content", "").strip()
        entry = f"Section: {heading}\nContent:\n{content}\n---"
        context_parts.append(entry)
        
    return "\n".join(context_parts)

def generate_with_ollama(ollama_host_url: str, ollama_model_name: str,
                         prompt: str, stream_response: bool = True) -> Generator[str, None, None] | str:
    """Generates response from Ollama using the ollama library."""
    print(f"INFO: Sending prompt to Ollama model '{ollama_model_name}' at host '{ollama_host_url}' (Streaming: {stream_response})...")

    client_params = {}
    if ollama_host_url != DEFAULT_OLLAMA_HOST:
        client_params['host'] = ollama_host_url

    try:
        client = ollama.Client(**client_params) # type: ignore

        if stream_response:
            def stream_generator() -> Generator[str, None, None]:
                response_stream = client.generate(
                    model=ollama_model_name,
                    prompt=prompt,
                    stream=True
                )
                for chunk_dict in response_stream:
                    if 'response' in chunk_dict:
                        yield chunk_dict['response']
                    if chunk_dict.get('done') and chunk_dict.get('error'):
                        error_msg = f"Ollama API stream returned an error: {chunk_dict['error']}"
                        print(f"\nERROR: {error_msg}")
                        break
                    elif chunk_dict.get('done'):
                        break
            return stream_generator()
        else: # Not streaming
            response_dict = client.generate(
                model=ollama_model_name,
                prompt=prompt,
                stream=False
            )
            if 'error' in response_dict:
                error_msg = f"Ollama API returned an error: {response_dict['error']}"
                print(f"ERROR: {error_msg}")
                return error_msg
            return response_dict.get('response', "Error: No 'response' field in Ollama output.")

    except ollama.ResponseError as e:
        error_msg = f"ERROR: Ollama API ResponseError: {e.error} (Status: {e.status_code})"
        print(error_msg)
    except Exception as e:
        error_msg = f"ERROR: Failed to communicate with Ollama or process its response: {e}"
        print(error_msg)

    if stream_response:
        def error_gen(): yield error_msg # type: ignore
        return error_gen()
    return error_msg # type: ignore


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve relevant chunks from Qdrant, fetch their parent windows, and generate a response using Ollama.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("user_query", help="The user's query string.")

    q_group = parser.add_argument_group('Qdrant Configuration')
    q_group.add_argument("--chunks_collection", default="chunks", help="Qdrant collection name for chunks.")
    q_group.add_argument("--windows_collection", default="windows", help="Qdrant collection name for windows.")
    q_conn_type = q_group.add_mutually_exclusive_group()
    q_conn_type.add_argument("--qdrant_url", default=os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL), help="Qdrant URL.")
    q_conn_type.add_argument("--qdrant_path", default=os.getenv("QDRANT_PATH"), help="Path to local Qdrant DB.")
    q_group.add_argument("--qdrant_api_key", default=os.getenv("QDRANT_API_KEY"), help="Qdrant Cloud API key.")
    q_group.add_argument("--qdrant_prefer_grpc", action="store_true", help="Prefer gRPC for Qdrant.")
    q_group.add_argument("--qdrant_timeout", type=int, default=DEFAULT_QDRANT_TIMEOUT, help="Qdrant client timeout.")
    q_group.add_argument("--vector_name", default=None, help="Named vector in Qdrant.")

    emb_group = parser.add_argument_group('Embedding Model Configuration')
    emb_group.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL, help="Sentence-transformer model.")

    ol_group = parser.add_argument_group('Ollama Configuration')
    ol_group.add_argument("--ollama_host", default=os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST), help="Ollama host URL.")
    ol_group.add_argument("--ollama_model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model name.")
    ol_group.add_argument("--no_stream", action="store_true", help="Disable Ollama streaming.")

    rag_group = parser.add_argument_group('Retrieval and RAG Configuration')
    rag_group.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Number of chunks from Qdrant.")

    args = parser.parse_args()

    print("--- Configuration ---")
    print(f"User Query: {args.user_query[:100]}...")
    print(f"Ollama Model: {args.ollama_model}, Host: {args.ollama_host}, Streaming: {not args.no_stream}")
    print(f"Chunks Collection: {args.chunks_collection}, Windows Collection: {args.windows_collection}, Top K: {args.top_k}")
    print("---------------------\n")

    model_instance = None
    q_client = None
    try:
        print("INFO: Initializing components...")
        model_instance = load_embedding_model(args.embedding_model)
        q_client = init_qdrant_client(
            url=args.qdrant_url if not args.qdrant_path else None,
            api_key=args.qdrant_api_key, path=args.qdrant_path,
            prefer_grpc=args.qdrant_prefer_grpc, timeout_seconds=args.qdrant_timeout
        )

        print("\nINFO: --- Retrieval Phase ---")
        query_vector = embed_query(model_instance, args.user_query)
        
        # Search for chunks
        chunk_search_results = search_qdrant(q_client, args.chunks_collection, query_vector, args.top_k, vector_name=args.vector_name)
        
        # Retrieve parent windows
        retrieved_windows = retrieve_windows_from_chunks(q_client, args.windows_collection, chunk_search_results)

        print("\nINFO: --- Augmentation & Generation Phase ---")
        context_str = format_context_from_windows(retrieved_windows)
        final_prompt = (
            f"You are a helpful assistant. Answer the user's query based on the provided context. "
            f"If the context does not adequately cover the query, state that the information is not found in the provided documents. "
            f"Be concise.\n\nCONTEXT:\n{context_str}\n\nUSER QUERY: {args.user_query}\n\nANSWER:")

        print("\n--- Ollama's Response ---")
        response_output = generate_with_ollama(args.ollama_host, args.ollama_model, final_prompt, stream_response=(not args.no_stream))

        if not args.no_stream:
            if isinstance(response_output, Generator):
                for part in response_output:
                    print(part, end="", flush=True)
                print()
        else:
            full_response = cast(str, response_output)
            print(full_response)

    except Exception as e:
        print(f"\nERROR: An critical error occurred in the RAG pipeline: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if q_client:
           try: q_client.close(); print("\nINFO: Qdrant client closed.")
           except Exception as e_close: print(f"ERROR: Could not close Qdrant client: {e_close}")

if __name__ == "__main__":
    main()
