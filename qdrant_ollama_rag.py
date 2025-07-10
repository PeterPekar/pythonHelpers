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
    import requests
except ImportError:
    print("Requests library not installed. Please install with 'pip install requests'")
    sys.exit(1)

# --- Default Configuration ---
DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_TOP_K = 3
DEFAULT_QDRANT_TIMEOUT = 30

# --- Core Functions ---

def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Loads a sentence-transformer model."""
    print(f"INFO: Attempting to load embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        if hasattr(model, 'get_sentence_embedding_dimension'):
            dim = model.get_sentence_embedding_dimension()
            if dim is not None:
                 model._actual_emb_dim = dim # type: ignore
        if not hasattr(model, '_actual_emb_dim'):
            dummy_emb = model.encode("test")
            model._actual_emb_dim = dummy_emb.shape[0] # type: ignore

        print(f"INFO: Embedding model '{model_name}' loaded successfully. Dimension: {model._actual_emb_dim}") # type: ignore
        return model
    except Exception as e:
        print(f"ERROR: Failed to load embedding model '{model_name}': {e}")
        raise

def get_embedding_dimension(model: SentenceTransformer) -> int:
    """Gets the embedding dimension of the loaded sentence transformer model."""
    if hasattr(model, '_actual_emb_dim'):
        return cast(int, model._actual_emb_dim) # type: ignore
    print("WARN: _actual_emb_dim not found on model, trying get_sentence_embedding_dimension().")
    try:
        if hasattr(model, 'get_sentence_embedding_dimension'):
            dim = model.get_sentence_embedding_dimension()
            if dim is not None:
                return dim
        dummy_embedding = model.encode("test")
        return dummy_embedding.shape[0]
    except Exception as e:
        print(f"ERROR: Error getting embedding dimension: {e}")
        raise ValueError("Could not determine embedding dimension from the model.")


def init_qdrant_client(url: Optional[str] = None,
                       api_key: Optional[str] = None,
                       path: Optional[str] = None,
                       prefer_grpc: bool = False,
                       timeout_seconds: int = DEFAULT_QDRANT_TIMEOUT) -> QdrantClient:
    """Initializes and returns a QdrantClient."""
    client_args: Dict[str, Any] = {}
    if path:
        print(f"INFO: Initializing Qdrant client with local path: {path}")
        client_args = {"path": path}
    elif url:
        effective_url = url
        print(f"INFO: Initializing Qdrant client with URL: {effective_url} (GRPC: {prefer_grpc})")
        client_args = {"url": effective_url, "prefer_grpc": prefer_grpc, "timeout": timeout_seconds}
        if api_key:
            print("INFO: Using API key for Qdrant connection.")
            client_args["api_key"] = api_key
    else:
        print(f"ERROR: No Qdrant URL or path specified for client initialization.")
        raise ValueError("Qdrant URL or path is required for client initialization.")

    try:
        client = QdrantClient(**client_args)
        client.health_check()
        print("INFO: Qdrant client initialized and connected successfully.")
        return client
    except Exception as e:
        print(f"ERROR: Failed to initialize Qdrant client: {e}")
        raise

def embed_query(model: SentenceTransformer, query: str) -> List[float]:
    """Embeds the user query."""
    print(f"INFO: Embedding query: \"{query[:100]}{'...' if len(query)>100 else ''}\"")
    try:
        embedding = model.encode(query)
        return embedding.tolist()
    except Exception as e:
        print(f"ERROR: Failed to embed query: {e}")
        raise

def search_qdrant(client: QdrantClient, collection_name: str, query_vector: List[float], top_k: int,
                  filter_conditions: Optional[models.Filter] = None, vector_name: Optional[str] = None
                  ) -> List[ScoredPoint]:
    """Searches Qdrant for similar chunks."""
    print(f"INFO: Searching Qdrant collection '{collection_name}' for top {top_k} results.")
    if filter_conditions:
        print(f"INFO: Applying filter conditions.")

    search_params = models.SearchParams(hnsw_ef=128, exact=False)
    query_vector_arg: List[float] | tuple[str, List[float]]
    if vector_name:
        query_vector_arg = (vector_name, query_vector)
    else:
        query_vector_arg = query_vector

    try:
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector_arg, # type: ignore
            query_filter=filter_conditions,
            limit=top_k,
            with_payload=True,
            search_params=search_params
        )
        print(f"INFO: Found {len(search_results)} results from Qdrant.")
        return search_results
    except Exception as e:
        print(f"ERROR: Failed to search Qdrant: {e}")
        raise

def format_context_from_results(results: List[ScoredPoint], max_context_chars: int = 7000) -> str:
    """Formats Qdrant results into a context string for Ollama, respecting max_context_chars."""
    if not results:
        return "No relevant context found in the knowledge base."

    context_parts = []
    current_chars = 0
    print(f"INFO: Formatting context from {len(results)} retrieved chunks...")

    for i, hit in enumerate(results):
        payload = hit.payload if hit.payload else {}
        payload_dict = cast(Dict[str, Any], payload)

        content = payload_dict.get("text_content", "").strip()
        metadata = payload_dict.get("metadata", {})
        metadata_dict = cast(Dict[str, Any], metadata if isinstance(metadata, dict) else {})

        source_file = metadata_dict.get("source_filename", "Unknown Source")
        heading_hierarchy = metadata_dict.get("heading_hierarchy", [])

        entry_header = f"Context from Document: '{source_file}'"
        if heading_hierarchy and isinstance(heading_hierarchy, list):
            entry_header += f"\nSection: \"{' > '.join(heading_hierarchy)}\""

        estimated_entry_length = len(entry_header) + len(content) + 50

        if max_context_chars > 0 and (current_chars + estimated_entry_length > max_context_chars) and context_parts:
            print(f"WARN: Max context length ({max_context_chars} chars) reached. Stopping context assembly at chunk {i+1} of {len(results)}.")
            break

        context_entry = f"{entry_header}\nContent:\n{content}\n---"
        context_parts.append(context_entry)
        current_chars += len(context_entry)

    if not context_parts:
        return "No relevant context could be formatted within the character limit."

    return "\n".join(context_parts)

def generate_with_ollama(ollama_base_url: str, ollama_model: str,
                         prompt: str, stream: bool = True,
                         timeout_seconds: int = 180) -> Generator[str, None, None] | str:
    """
    Generates a response from Ollama using the provided prompt.
    Can stream the response or return the full response.
    Includes better error checking from Ollama's JSON response.
    """
    api_url = f"{ollama_base_url.rstrip('/')}/api/generate"
    payload_data = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": stream,
    }
    print(f"INFO: Sending prompt to Ollama model '{ollama_model}' at {api_url} (Streaming: {stream})...")

    # This variable will store error message derived from Ollama's JSON response, if any
    ollama_json_error_msg: Optional[str] = None

    try:
        response = requests.post(api_url, json=payload_data, stream=stream, timeout=timeout_seconds)
        response.raise_for_status() # Check for HTTP errors like 404, 500

        if stream:
            def stream_generator() -> Generator[str, None, None]:
                nonlocal ollama_json_error_msg
                for line in response.iter_lines():
                    if line:
                        try:
                            json_line = json.loads(line.decode('utf-8'))
                            if "response" in json_line:
                                yield json_line["response"]

                            if "error" in json_line:
                                ollama_json_error_msg = f"Ollama API stream returned an error: {json_line['error']}"
                                print(f"\nERROR: {ollama_json_error_msg}")

                            if json_line.get("done"):
                                if ollama_json_error_msg:
                                     pass # Error already printed.
                                break
                        except json.JSONDecodeError:
                            print(f"WARN: Received non-JSON line from Ollama stream: {line[:100]}...")
            return stream_generator()
        else: # Not streaming
            response_data = response.json()
            if "error" in response_data:
                ollama_json_error_msg = f"Ollama API returned an error: {response_data['error']}"
                print(f"ERROR: {ollama_json_error_msg}")
                return ollama_json_error_msg # Return the error message string
            return response_data.get("response", "Error: No 'response' field in Ollama output.")

    except requests.exceptions.Timeout:
        # This specific exception is caught first
        error_msg_http = f"ERROR: Ollama API request timed out after {timeout_seconds} seconds."
        print(error_msg_http)
        if stream: return (yield error_msg_http) # type: ignore
        return error_msg_http
    except requests.exceptions.RequestException as e:
        error_msg_http = f"ERROR: Ollama API request failed: {e}"
        print(error_msg_http)
        if stream: return (yield error_msg_http) # type: ignore
        return error_msg_http
    except Exception as e: # Catch any other unexpected errors during the request or initial processing
        error_msg_general = f"ERROR: An unexpected error occurred during Ollama communication: {e}"
        print(error_msg_general)
        if stream: return (yield error_msg_general) # type: ignore
        return error_msg_general

    # This part is reached if stream=True and the requests.post was successful but
    # an error might have been found *within* the stream by ollama_json_error_msg.
    # For non-streaming, errors from JSON are returned directly.
    if stream and ollama_json_error_msg:
        # If an error was found in the stream, the generator might have already yielded parts.
        # It's tricky to retrospectively signal this error *through* a generator that might have completed.
        # The error is printed above. The generator will simply stop or might have yielded the error.
        # To ensure the caller of the generator sees an error if one occurred:
        def error_after_stream_gen():
            yield ollama_json_error_msg # type: ignore
        return error_after_stream_gen()

    # If it was a non-streaming call and ollama_json_error_msg was set, it was already returned.
    # This is a fallback for stream=True if no ollama_json_error and no prior exception,
    # which means the stream_generator itself is the correct return.
    # This function's structure for returning errors in stream mode is complex.
    # The primary way errors are communicated in stream mode is by printing and then the generator might end.
    # A more robust way would be for the generator to yield a special error object or raise an exception.
    # For now, printing the error is the main feedback for stream errors from JSON.

    # Fallback for unhandled error conditions, though ideally covered above.
    unspecified_error = "Ollama generation failed due to an unspecified error after the request."
    if stream:
        def final_fallback_gen(): yield unspecified_error
        return final_fallback_gen()
    return unspecified_error


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve relevant chunks from Qdrant and generate a response using Ollama.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("user_query", help="The user's query string.")

    qdrant_group = parser.add_argument_group('Qdrant Configuration')
    qdrant_group.add_argument("--qdrant_collection", required=True, help="Name of the Qdrant collection to search.")
    q_conn_type = qdrant_group.add_mutually_exclusive_group()
    q_conn_type.add_argument("--qdrant_url", default=os.getenv("QDRANT_URL"),
                               help=f"URL of Qdrant instance. If not set, and qdrant_path is not set, defaults to {DEFAULT_QDRANT_URL}.")
    q_conn_type.add_argument("--qdrant_path", default=os.getenv("QDRANT_PATH"),
                               help="Path to local Qdrant database file.")
    qdrant_group.add_argument("--qdrant_api_key", default=os.getenv("QDRANT_API_KEY"), help="API key for Qdrant Cloud.")
    qdrant_group.add_argument("--qdrant_prefer_grpc", action="store_true", help="Prefer gRPC for Qdrant connection.")
    qdrant_group.add_argument("--qdrant_timeout", type=int, default=DEFAULT_QDRANT_TIMEOUT, help="Timeout for Qdrant client ops.")
    qdrant_group.add_argument("--vector_name", default=None, help="Name of the vector in Qdrant (if using named vectors).")

    embedding_group = parser.add_argument_group('Embedding Model Configuration')
    embedding_group.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL, help="Sentence-transformer model name.")

    ollama_group = parser.add_argument_group('Ollama Configuration')
    ollama_group.add_argument("--ollama_base_url", default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL), help="Base URL for Ollama API.")
    ollama_group.add_argument("--ollama_model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model name.")
    ollama_group.add_argument("--no_stream", action="store_true", help="Disable Ollama streaming.")
    ollama_group.add_argument("--ollama_timeout", type=int, default=180, help="Timeout in seconds for Ollama API requests (default: 180).")

    rag_group = parser.add_argument_group('Retrieval and RAG Configuration')
    rag_group.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve from Qdrant.")
    rag_group.add_argument("--filter_filename", default=None, help="Filter Qdrant results by metadata.source_filename.")
    rag_group.add_argument("--max_context_chars", type=int, default=7000,
                           help="Maximum characters for the context passed to Ollama (0 for no limit, default: 7000).")

    args = parser.parse_args()

    effective_qdrant_url = args.qdrant_url
    if not args.qdrant_path and not args.qdrant_url:
        effective_qdrant_url = DEFAULT_QDRANT_URL

    print("--- Configuration ---")
    # ... (config printing can be reinstated if desired) ...
    print(f"User Query: {args.user_query}")
    print("---------------------\n")

    model_instance = None
    q_client = None
    try:
        print("INFO: Initializing components...")
        model_instance = load_embedding_model(args.embedding_model)

        qdrant_init_url = effective_qdrant_url if not args.qdrant_path else None
        q_client = init_qdrant_client(
            url=qdrant_init_url, api_key=args.qdrant_api_key, path=args.qdrant_path,
            prefer_grpc=args.qdrant_prefer_grpc, timeout_seconds=args.qdrant_timeout
        )

        print("\nINFO: --- Retrieval Phase ---")
        query_vector = embed_query(model_instance, args.user_query)

        qdrant_filter_conditions = None
        if args.filter_filename:
            qdrant_filter_conditions = models.Filter(must=[
                models.FieldCondition(key="metadata.source_filename", match=models.MatchValue(value=args.filter_filename))
            ])
            print(f"INFO: Applying filename filter for: '{args.filter_filename}'")

        search_results = search_qdrant(
            q_client, args.qdrant_collection, query_vector, args.top_k,
            filter_conditions=qdrant_filter_conditions, vector_name=args.vector_name
        )

        print("\nINFO: --- Augmentation & Generation Phase ---")
        context_str = format_context_from_results(search_results, args.max_context_chars)

        final_prompt = (
            f"You are a helpful assistant. Please answer the user's query based on the provided context. "
            f"If the context does not contain the answer, state that the information is not found in the provided documents. "
            f"Be concise and directly answer the query.\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"USER QUERY: {args.user_query}\n\n"
            f"ANSWER:"
        )

        print("\n--- Ollama's Response ---")
        ollama_response_gen_or_str = generate_with_ollama(
            args.ollama_base_url, args.ollama_model, final_prompt,
            stream=(not args.no_stream), timeout_seconds=args.ollama_timeout
        )

        if not args.no_stream:
            is_first_response_part = True
            has_printed_error = False
            if isinstance(ollama_response_gen_or_str, Generator):
                for part in ollama_response_gen_or_str:
                    if is_first_response_part and part.startswith("ERROR:"):
                        print(part)
                        has_printed_error = True
                        break
                    print(part, end="", flush=True)
                    is_first_response_part = False
                if not has_printed_error: print() # Newline after successful stream
            elif isinstance(ollama_response_gen_or_str, str) and ollama_response_gen_or_str.startswith("ERROR:"):
                 print(ollama_response_gen_or_str) # Error occurred before generator was even created
        else:
            full_response = cast(str, ollama_response_gen_or_str)
            # No need to check for "ERROR:" prefix here if generate_with_ollama prints it and returns it
            print(full_response)

    except Exception as e:
        print(f"\nERROR: An critical error occurred in the RAG pipeline: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if q_client:
           try:
               q_client.close()
               print("\nINFO: Qdrant client closed.")
           except Exception as e_close:
               print(f"ERROR: Could not close Qdrant client: {e_close}")

if __name__ == "__main__":
    main()
