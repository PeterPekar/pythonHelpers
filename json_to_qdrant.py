import argparse
import json
import os
import uuid
from typing import Optional, List, Dict, Any

try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.models import PointStruct # For type hinting
except ImportError:
    print("Qdrant client not installed. Please install with 'pip install qdrant-client'")
    exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("SentenceTransformers not installed. Please install with 'pip install sentence-transformers torch'")
    exit(1)

# Default values
DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_DISTANCE_METRIC_STR = "Cosine"

# --- Phase 1 Functions ---
def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Loads a sentence-transformer model."""
    print(f"Loading sentence transformer model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        print(f"Model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading sentence transformer model '{model_name}': {e}")
        raise

def get_embedding_dimension(model: SentenceTransformer) -> int:
    """Gets the embedding dimension of the loaded sentence transformer model."""
    try:
        dim = model.get_sentence_embedding_dimension()
        if dim is None:
            dummy_embedding = model.encode("test")
            dim = dummy_embedding.shape[0]
        return dim
    except Exception as e:
        print(f"Error getting embedding dimension: {e}")
        raise ValueError("Could not determine embedding dimension from the model.")

def init_qdrant_client(url: Optional[str] = None,
                       api_key: Optional[str] = None,
                       path: Optional[str] = None,
                       prefer_grpc: bool = False,
                       timeout_seconds: int = 30) -> QdrantClient:
    """Initializes and returns a QdrantClient."""
    client_args: Dict[str, Any] = {} # Ensure client_args is always defined
    if path:
        print(f"Initializing Qdrant client with local path: {path}")
        client_args = {"path": path}
    elif url:
        print(f"Initializing Qdrant client with URL: {url} (GRPC: {prefer_grpc})")
        client_args = {"url": url, "prefer_grpc": prefer_grpc, "timeout": timeout_seconds}
        if api_key:
            print("Using API key for Qdrant connection.")
            client_args["api_key"] = api_key
    else:
        print(f"No Qdrant URL or path specified, defaulting to: {DEFAULT_QDRANT_URL}")
        client_args = {"url": DEFAULT_QDRANT_URL, "prefer_grpc": prefer_grpc, "timeout": timeout_seconds}

    try:
        client = QdrantClient(**client_args)
        
        print("Qdrant client initialized and connected successfully.")
        return client
    except Exception as e:
        print(f"Error initializing Qdrant client: {e}")
        print("Please ensure Qdrant instance is running and accessible with the provided parameters.")
        if path: print(f"  Attempted path: {path}")
        elif 'url' in client_args: print(f"  Attempted URL: {client_args.get('url')}") # check if url key exists
        if api_key: print("  API key was provided.")
        raise

def ensure_collection(client: QdrantClient, collection_name: str,
                      vector_size: int, distance_metric_str: str,
                      vector_name: Optional[str] = None):
    """Ensures a Qdrant collection exists with the specified parameters."""
    distance_metric_map = {
        "cosine": models.Distance.COSINE,
        "euclidean": models.Distance.EUCLID,
        "dot": models.Distance.DOT
    }
    qdrant_distance_metric = distance_metric_map.get(distance_metric_str.lower())
    if qdrant_distance_metric is None:
        raise ValueError(f"Unsupported distance metric: {distance_metric_str}. Choose from Cosine, Euclidean, Dot.")

    effective_vector_name = vector_name or "default"
    print(f"Ensuring collection '{collection_name}' (vector_name: {effective_vector_name}) exists with vector size {vector_size} and distance {qdrant_distance_metric.name}...")

    try:
        collection_info = client.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
        actual_vectors_config = collection_info.config.params.vectors

        existing_size: int
        existing_distance: models.Distance

        if vector_name is None:
            if not isinstance(actual_vectors_config, models.VectorParams): # type: ignore
                raise ValueError(f"Collection '{collection_name}' exists but is configured for named vectors, while a default (unnamed) vector setup is expected.")
            existing_size = actual_vectors_config.size # type: ignore
            existing_distance = actual_vectors_config.distance # type: ignore
        else:
            if not isinstance(actual_vectors_config, dict) or vector_name not in actual_vectors_config:
                available_vectors = list(actual_vectors_config.keys()) if isinstance(actual_vectors_config, dict) else "default only (or misconfigured)"
                raise ValueError(f"Collection '{collection_name}' exists but named vector '{vector_name}' not found. Available: {available_vectors}")
            existing_size = actual_vectors_config[vector_name].size
            existing_distance = actual_vectors_config[vector_name].distance

        if existing_size != vector_size or existing_distance != qdrant_distance_metric:
            error_msg = (f"Collection '{collection_name}' (vector: {effective_vector_name}) exists with incompatible configuration. "
                         f"Expected size: {vector_size}, distance: {qdrant_distance_metric.name}. "
                         f"Found size: {existing_size}, distance: {existing_distance.name}. " # Safe to use .name as it's an enum
                         "Please use a different collection name or ensure parameters match.")
            raise ValueError(error_msg)
        print(f"Existing collection configuration for vector '{effective_vector_name}' is compatible.")

    except Exception as e:
        is_not_found_error = ("Not Found" in str(e) or "NOT_FOUND" in str(e) or
                              (hasattr(e, 'status_code') and e.status_code == 404) or
                              " yoktur" in str(e).lower() or "aucun" in str(e).lower() # Turkish, French for "no such" / "none"
                             )
        if is_not_found_error:
            print(f"Collection '{collection_name}' not found. Creating new collection...")
            vectors_param: models.VectorParams | Dict[str, models.VectorParams]
            if vector_name is None:
                vectors_param = models.VectorParams(size=vector_size, distance=qdrant_distance_metric)
            else:
                vectors_param = {vector_name: models.VectorParams(size=vector_size, distance=qdrant_distance_metric)}
            try:
                client.create_collection(collection_name=collection_name, vectors_config=vectors_param) # type: ignore
                print(f"Collection '{collection_name}' created successfully.")
            except Exception as creation_error:
                print(f"Failed to create collection '{collection_name}': {creation_error}")
                raise
        else:
            print(f"Error checking/handling collection '{collection_name}': {e}")
            raise

# --- Phase 2 Functions ---
def load_chunks_from_json(json_file_path: str) -> List[Dict[str, Any]]:
    """Loads chunks from the specified JSON file."""
    print(f"Loading chunks from JSON file: {json_file_path}...")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        if not isinstance(chunks_data, list):
            raise ValueError("JSON file should contain a list of chunk objects.")
        for i, chunk in enumerate(chunks_data):
            if not isinstance(chunk, dict) or "content" not in chunk or "metadata" not in chunk:
                raise ValueError(f"Chunk at index {i} has invalid structure (missing 'content' or 'metadata').")
        print(f"Loaded {len(chunks_data)} chunks from {json_file_path}.")
        return chunks_data
    except FileNotFoundError:
        print(f"Error: JSON file not found at '{json_file_path}'.")
        raise
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{json_file_path}': {e}")
        raise
    except ValueError as e: # Catches custom ValueErrors from structure check
        print(f"Error in JSON structure: {e}")
        raise

def process_and_upsert_chunks(client: QdrantClient, collection_name: str,
                              embedding_model: SentenceTransformer,
                              chunks: List[Dict[str, Any]],
                              batch_size: int,
                              id_field: Optional[str] = None,
                              vector_name: Optional[str] = None):
    """Generates embeddings for chunks and upserts them to Qdrant in batches."""
    total_chunks = len(chunks)
    if total_chunks == 0:
        print("No chunks to process.")
        return

    print(f"Starting processing and upserting of {total_chunks} chunks to collection '{collection_name}'...")

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_number = i // batch_size + 1
        total_batches = (total_chunks + batch_size - 1) // batch_size
        print(f"  Processing batch {batch_number}/{total_batches} (chunks {i+1}-{min(i+batch_size, total_chunks)})...")

        texts_to_embed = [chunk["content"] for chunk in batch_chunks]
        try:
            embeddings = embedding_model.encode(texts_to_embed, show_progress_bar=False)
        except Exception as e:
            print(f"Error generating embeddings for batch {batch_number}: {e}")
            print(f"Skipping this batch due to embedding error.")
            continue

        batch_points: List[PointStruct] = []
        for chunk_idx, chunk_data in enumerate(batch_chunks):
            point_id_val: str
            # Ensure metadata is a dict before trying to access id_field
            metadata_dict = chunk_data.get("metadata", {}) if isinstance(chunk_data.get("metadata"), dict) else {}

            if id_field and id_field in metadata_dict:
                point_id_val = str(metadata_dict[id_field])
            else:
                point_id_val = str(uuid.uuid4())

            payload = {
                "text_content": chunk_data["content"],
                "metadata": metadata_dict # Use the (potentially empty) metadata_dict
            }

            vector_input: models.VectorStruct | Dict[str, List[float]]
            current_embedding = embeddings[chunk_idx].tolist()
            if vector_name:
                vector_input = {vector_name: current_embedding}
            else:
                vector_input = current_embedding

            batch_points.append(PointStruct(id=point_id_val, vector=vector_input, payload=payload)) # type: ignore

        if batch_points:
            try:
                client.upsert(collection_name=collection_name, points=batch_points, wait=True)
            except Exception as e:
                print(f"Error upserting batch {batch_number} to Qdrant: {e}")
                print(f"Skipping this batch due to upsert error.")

    print(f"Finished processing and upserting all chunks.")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Load chunks from JSON and upsert to Qdrant.")

    parser.add_argument("json_file", help="Path to the input JSON file containing chunks.")
    parser.add_argument("collection_name", help="Name of the Qdrant collection.")

    q_group = parser.add_argument_group('Qdrant Connection')
    q_conn_type = q_group.add_mutually_exclusive_group()
    q_conn_type.add_argument("--qdrant_url", default=os.getenv("QDRANT_URL"),
                               help=f"URL of Qdrant instance. If neither URL nor path is given, defaults to '{DEFAULT_QDRANT_URL}'.")
    q_conn_type.add_argument("--qdrant_path", default=os.getenv("QDRANT_PATH"),
                               help="Path to local Qdrant database file.")
    q_group.add_argument("--qdrant_api_key", default=os.getenv("QDRANT_API_KEY"),
                        help="API key for Qdrant Cloud.")
    q_group.add_argument("--prefer_grpc", action="store_true",
                        help="Prefer gRPC for Qdrant connection.")
    q_group.add_argument("--qdrant_timeout", type=int, default=30,
                        help="Timeout in seconds for Qdrant client operations.")

    e_group = parser.add_argument_group('Embedding and Collection Parameters')
    e_group.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL,
                        help=f"Sentence-transformer model name (default: '{DEFAULT_EMBEDDING_MODEL}').")
    e_group.add_argument("--vector_name", default=None,
                        help="Name of vector in Qdrant (default: unnamed/default vector).")
    e_group.add_argument("--distance_metric", default=DEFAULT_DISTANCE_METRIC_STR, choices=["Cosine", "Euclidean", "Dot"],
                        help=f"Distance metric for Qdrant collection (default: {DEFAULT_DISTANCE_METRIC_STR}).")

    b_group = parser.add_argument_group('Batching and ID Parameters')
    b_group.add_argument("--batch_size", type=int, default=64,
                         help="Batch size for upserting points to Qdrant (default: 64).")
    b_group.add_argument("--id_field", default=None,
                         help="Field name in JSON metadata to use as Qdrant point ID (e.g., 'doc_chunk_id').")

    args = parser.parse_args()

    qdrant_url_to_use = args.qdrant_url
    if not args.qdrant_path and not args.qdrant_url:
        qdrant_url_to_use = DEFAULT_QDRANT_URL
        print(f"Qdrant URL or path not specified, using default URL: {qdrant_url_to_use}")

    print("--- Initializing ---")
    embedding_model_instance: Optional[SentenceTransformer] = None
    qdrant_client_instance: Optional[QdrantClient] = None

    try:
        all_chunks_data = load_chunks_from_json(args.json_file)
        if not all_chunks_data:
            print("No chunks found in JSON file or file is empty. Exiting.")
            return

        embedding_model_instance = load_embedding_model(args.embedding_model)
        vector_dimension = get_embedding_dimension(embedding_model_instance)
        print(f"Determined vector dimension: {vector_dimension} for model '{args.embedding_model}'")

        qdrant_client_instance = init_qdrant_client(
            url=qdrant_url_to_use if not args.qdrant_path else None,
            api_key=args.qdrant_api_key,
            path=args.qdrant_path,
            prefer_grpc=args.prefer_grpc,
            timeout_seconds=args.qdrant_timeout
        )

        ensure_collection(
            qdrant_client_instance,
            args.collection_name,
            vector_dimension,
            args.distance_metric,
            args.vector_name
        )

        print("\n--- Starting Chunk Processing and Upserting to Qdrant ---")
        process_and_upsert_chunks(
            qdrant_client_instance,
            args.collection_name,
            embedding_model_instance,
            all_chunks_data,
            args.batch_size,
            args.id_field,
            args.vector_name
        )

        print("\n--- Script execution completed successfully. ---")

    except FileNotFoundError: # Specifically for the input JSON file
        print(f"Critical error: Input JSON file '{args.json_file}' not found. Please check the path.")
    except Exception as e:
        print(f"An unexpected error occurred during script execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if qdrant_client_instance:
            try:
                qdrant_client_instance.close()
                print("Qdrant client closed.")
            except Exception as e_close:
                print(f"Error closing Qdrant client: {e_close}")

if __name__ == "__main__":
    main()
