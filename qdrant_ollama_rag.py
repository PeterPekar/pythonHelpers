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
    import ollama # Changed from requests
except ImportError:
    print("Ollama library not installed. Please install with 'pip install ollama'")
    sys.exit(1)

# --- Default Configuration ---
DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_OLLAMA_HOST = "http://localhost:11434" # ollama library uses OLLAMA_HOST env var by default
DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_TOP_K = 3
DEFAULT_QDRANT_TIMEOUT = 30

# --- Core Functions ---

def load_embedding_model(model_name: str) -> SentenceTransformer:
    print(f"INFO: Attempting to load embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        if hasattr(model, 'get_sentence_embedding_dimension'):
            dim = model.get_sentence_embedding_dimension()
            if dim is not None: model._actual_emb_dim = dim # type: ignore
        if not hasattr(model, '_actual_emb_dim'):
            dummy_emb = model.encode("test")
            model._actual_emb_dim = dummy_emb.shape[0] # type: ignore
        print(f"INFO: Embedding model '{model_name}' loaded. Dimension: {model._actual_emb_dim}") # type: ignore
        return model
    except Exception as e:
        print(f"ERROR: Failed to load embedding model '{model_name}': {e}")
        raise

def get_embedding_dimension(model: SentenceTransformer) -> int:
    if hasattr(model, '_actual_emb_dim'): return cast(int, model._actual_emb_dim) # type: ignore
    try:
        if hasattr(model, 'get_sentence_embedding_dimension'):
            dim = model.get_sentence_embedding_dimension()
            if dim is not None: return dim
        return model.encode("test").shape[0]
    except Exception as e:
        print(f"ERROR: Error getting embedding dimension: {e}")
        raise ValueError("Could not determine embedding dimension.")

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
                  filter_conditions: Optional[models.Filter] = None, vector_name: Optional[str] = None
                  ) -> List[ScoredPoint]:
    print(f"INFO: Searching Qdrant collection '{collection_name}' for top {top_k} results.")
    if filter_conditions: print(f"INFO: Applying filter conditions.")

    search_params = models.SearchParams(hnsw_ef=128, exact=False)
    query_vector_arg: List[float] | tuple[str, List[float]] = (vector_name, query_vector) if vector_name else query_vector

    try:
        results = client.search(
            collection_name=collection_name, query_vector=query_vector_arg, # type: ignore
            query_filter=filter_conditions, limit=top_k,
            with_payload=True, search_params=search_params
        )
        print(f"INFO: Found {len(results)} results from Qdrant.")
        return results
    except Exception as e:
        print(f"ERROR: Failed to search Qdrant: {e}")
        raise

def format_context_from_results(results: List[ScoredPoint], max_context_chars: int = 7000) -> str:
    if not results: return "No relevant context found in the knowledge base."
    context_parts = []; current_chars = 0
    for i, hit in enumerate(results):
        payload = cast(Dict[str, Any], hit.payload if hit.payload else {})
        content = payload.get("text_content", "").strip()
        metadata = cast(Dict[str, Any], payload.get("metadata", {}))
        source_file = metadata.get("source_filename", "Unknown Source")
        headings = cast(List[str], metadata.get("heading_hierarchy", []))

        is_code = metadata.get("is_code_block", False)
        is_table = metadata.get("is_table_row", False)
        is_list_item = metadata.get("is_list", False)

        entry_header = f"Context from Document: '{source_file}'"
        if headings: entry_header += f"\nSection: \"{' > '.join(headings)}\""

        content_type = "Text"
        if is_code:
            content_type = "Code Block"
        elif is_table:
            content_type = "Table Row"
        elif is_list_item:
            content_type = "List"

        entry_header += f"\nType: {content_type}"

        formatted_content = content
        if is_code:
            formatted_content = f"```\n{content}\n```"
        elif is_table:
            # Tables are already in Markdown format
            pass

        entry_len = len(entry_header) + len(formatted_content) + 50
        if max_context_chars > 0 and (current_chars + entry_len > max_context_chars) and context_parts:
            print(f"WARN: Max context length ({max_context_chars}) reached. Stopping at chunk {i+1}/{len(results)}.")
            break

        context_parts.append(f"{entry_header}\nContent:\n{formatted_content}\n---")
        current_chars += entry_len

    return "\n".join(context_parts) if context_parts else "No context formatted."

def generate_with_ollama(ollama_host_url: str, ollama_model_name: str,
                         prompt: str, stream_response: bool = True) -> Generator[str, None, None] | str:
    """Generates response from Ollama using the ollama library."""
    print(f"INFO: Sending prompt to Ollama model '{ollama_model_name}' at host '{ollama_host_url}' (Streaming: {stream_response})...")

    client_params = {}
    # The ollama library uses OLLAMA_HOST env var by default if host is not specified.
    # Only pass host to client if it's different from the default or if user explicitly set it.
    if ollama_host_url != DEFAULT_OLLAMA_HOST: # Compare with default, not just if it's None
        client_params['host'] = ollama_host_url

    try:
        # If client_params is empty, it uses default host (e.g. http://localhost:11434)
        # or OLLAMA_HOST env var if set.
        client = ollama.Client(**client_params) # type: ignore

        if stream_response:
            def stream_generator() -> Generator[str, None, None]:
                response_stream = client.generate(
                    model=ollama_model_name,
                    prompt=prompt,
                    stream=True
                    # options={} # Add Ollama options here if needed
                )
                for chunk_dict in response_stream:
                    if 'response' in chunk_dict:
                        yield chunk_dict['response']
                    if chunk_dict.get('done') and chunk_dict.get('error'):
                        error_msg = f"Ollama API stream returned an error: {chunk_dict['error']}"
                        print(f"\nERROR: {error_msg}")
                        # To signal error through generator, can yield it or raise
                        # For simplicity, error is printed, generator stops.
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
                return error_msg # Return the error message string
            return response_dict.get('response', "Error: No 'response' field in Ollama output.")

    except ollama.ResponseError as e:
        error_msg = f"ERROR: Ollama API ResponseError: {e.error} (Status: {e.status_code})"
        print(error_msg)
    except Exception as e:
        error_msg = f"ERROR: Failed to communicate with Ollama or process its response: {e}"
        print(error_msg)

    # Fallback error return
    if stream_response:
        def error_gen(): yield error_msg # type: ignore
        return error_gen()
    return error_msg # type: ignore


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve relevant chunks from Qdrant and generate a response using Ollama.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("user_query", help="The user's query string.")

    q_group = parser.add_argument_group('Qdrant Configuration')
    q_group.add_argument("--qdrant_collection", required=True, help="Qdrant collection name.")
    q_conn_type = q_group.add_mutually_exclusive_group()
    q_conn_type.add_argument("--qdrant_url", default=os.getenv("QDRANT_URL"), help=f"Qdrant URL. Default if no path: {DEFAULT_QDRANT_URL}.")
    q_conn_type.add_argument("--qdrant_path", default=os.getenv("QDRANT_PATH"), help="Path to local Qdrant DB.")
    q_group.add_argument("--qdrant_api_key", default=os.getenv("QDRANT_API_KEY"), help="Qdrant Cloud API key.")
    q_group.add_argument("--qdrant_prefer_grpc", action="store_true", help="Prefer gRPC for Qdrant.")
    q_group.add_argument("--qdrant_timeout", type=int, default=DEFAULT_QDRANT_TIMEOUT, help="Qdrant client timeout.")
    q_group.add_argument("--vector_name", default=None, help="Named vector in Qdrant.")

    emb_group = parser.add_argument_group('Embedding Model Configuration')
    emb_group.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL, help="Sentence-transformer model.")

    ol_group = parser.add_argument_group('Ollama Configuration')
    # Changed --ollama_base_url to --ollama_host to match ollama library's client parameter
    ol_group.add_argument("--ollama_host", default=os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST), help="Ollama host URL (e.g. http://localhost:11434).")
    ol_group.add_argument("--ollama_model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model name.")
    ol_group.add_argument("--no_stream", action="store_true", help="Disable Ollama streaming.")
    # Removed --ollama_timeout as ollama library handles timeouts differently (e.g. OLLAMA_REQUEST_TIMEOUT env var)

    rag_group = parser.add_argument_group('Retrieval and RAG Configuration')
    rag_group.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Number of chunks from Qdrant.")
    rag_group.add_argument("--filter_filename", default=None, help="Filter Qdrant by metadata.source_filename.")
    rag_group.add_argument("--filter_is_code", action="store_true", help="Filter for code blocks.")
    rag_group.add_argument("--filter_is_table", action="store_true", help="Filter for table rows.")
    rag_group.add_argument("--filter_is_list", action="store_true", help="Filter for lists.")
    rag_group.add_argument("--max_context_chars", type=int, default=7000, help="Max context characters for Ollama (0 for no limit).")

    args = parser.parse_args()

    qdrant_url_to_use = args.qdrant_url
    if not args.qdrant_path and not args.qdrant_url:
        qdrant_url_to_use = DEFAULT_QDRANT_URL

    print("--- Configuration ---")
    print(f"User Query: {args.user_query[:100]}...")
    print(f"Ollama Model: {args.ollama_model}, Host: {args.ollama_host}, Streaming: {not args.no_stream}")
    print(f"Qdrant Collection: {args.qdrant_collection}, Top K: {args.top_k}")
    if args.filter_filename: print(f"Filename Filter: {args.filter_filename}")
    print("---------------------\n")

    model_instance = None
    q_client = None
    try:
        print("INFO: Initializing components...")
        model_instance = load_embedding_model(args.embedding_model)
        q_client = init_qdrant_client(
            url=qdrant_url_to_use if not args.qdrant_path else None,
            api_key=args.qdrant_api_key, path=args.qdrant_path,
            prefer_grpc=args.qdrant_prefer_grpc, timeout_seconds=args.qdrant_timeout
        )

        print("\nINFO: --- Retrieval Phase ---")
        query_vector = embed_query(model_instance, args.user_query)

        filters = []
        if args.filter_filename:
            filters.append(models.FieldCondition(key="metadata.source_filename", match=models.MatchValue(value=args.filter_filename)))
        if args.filter_is_code:
            filters.append(models.FieldCondition(key="metadata.is_code_block", match=models.MatchValue(value=True)))
        if args.filter_is_table:
            filters.append(models.FieldCondition(key="metadata.is_table_row", match=models.MatchValue(value=True)))
        if args.filter_is_list:
            filters.append(models.FieldCondition(key="metadata.is_list", match=models.MatchValue(value=True)))

        q_filter = models.Filter(must=filters) if filters else None

        search_results = search_qdrant(q_client, args.qdrant_collection, query_vector, args.top_k, filter_conditions=q_filter, vector_name=args.vector_name)
        print(f"INFO: Retrieved {search_results} chunks from Qdrant.")

        print("\nINFO: --- Augmentation & Generation Phase ---")
        context_str = format_context_from_results(search_results, args.max_context_chars)
        final_prompt = (
            f"You are a helpful assistant. Answer the user's query based on the provided context. "
            f"The context below contains chunks of text from a technical document. Each chunk has a type (e.g., 'Text', 'Code Block', 'Table Row', 'List'). "
            f"Pay attention to the type of each chunk to understand its content. For example, 'Code Block' contains source code, and 'Table Row' is a row from a larger table. "
            f"If the context does not adequately cover the query, state that the information is not found in the provided documents. "
            f"Be concise.\n\nCONTEXT:\n{context_str}\n\nUSER QUERY: {args.user_query}\n\nANSWER:")

        print("\n--- Ollama's Response ---")
        response_output = generate_with_ollama(args.ollama_host, args.ollama_model, final_prompt, stream_response=(not args.no_stream))

        if not args.no_stream:
            first_chunk_processed = False
            if isinstance(response_output, Generator):
                for part in response_output:
                    # If the first part itself is an error message string (from error_gen)
                    if not first_chunk_processed and isinstance(part, str) and part.startswith("ERROR:"):
                        print(part) # Print the error and stop processing this generator
                        break
                    print(part, end="", flush=True)
                    first_chunk_processed = True
                if first_chunk_processed: print() # Newline after successful stream only if something was printed
            elif isinstance(response_output, str) and response_output.startswith("ERROR:"): # Should be caught by generator check mostly
                print(response_output)
        else:
            full_response = cast(str, response_output)
            print(full_response) # Error or actual response string

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
