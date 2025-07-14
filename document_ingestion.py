import os
import re
import uuid
from typing import Optional, List, Dict, Any
from docx import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

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

def is_heading(paragraph):
    """
    Determines if a paragraph is a heading.
    A paragraph is considered a heading if it's bold and has a larger font size
    than the preceding paragraph.
    """
    if not paragraph.runs:
        return False

    # Check for bold style
    is_bold = all(run.bold for run in paragraph.runs if run.text.strip())
    if not is_bold:
        return False

    # Check for increased font size (compared to previous paragraph)
    # This is a heuristic and might need adjustment based on the document's structure.
    # For simplicity, we'll consider any font size > 12 as a potential heading font size.
    # A more robust solution would compare with the document's default font size.
    font_size = paragraph.runs[0].font.size
    if font_size is None:
        return False # Cannot determine font size

    # Assuming standard body text is 12pt (152400 English Metric Units)
    return font_size > 152400

def docx_to_markdown(docx_path):
    """
    Converts a .docx file to a Markdown string, preserving headings.
    """
    document = Document(docx_path)
    markdown_lines = []

    for para in document.paragraphs:
        if para.style.name.startswith('Heading'):
            level = int(para.style.name[-1])
            markdown_lines.append(f"{'#' * level} {para.text}")
        elif is_heading(para):
            # Simple heuristic: treat bold paragraphs as level 2 headings
            markdown_lines.append(f"## {para.text}")
        else:
            markdown_lines.append(para.text)

    return "\n".join(markdown_lines)

def ingest_document(docx_path):
    """
    Ingests a .docx file, processes it into structured Markdown,
    and splits it into windows and chunks.
    """
    markdown_content = docx_to_markdown(docx_path)

    # 1. Windowing with MarkdownHeaderTextSplitter
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_content)

    # 2. Chunking with RecursiveCharacterTextSplitter
    chunk_size = 100
    chunk_overlap = 20
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    windows_and_chunks = []
    for i, split in enumerate(md_header_splits):
        chunks = text_splitter.split_text(split.page_content)
        window = {
            "index": i,
            "heading": split.metadata.get("Header 2", "N/A"),
            "content": split.page_content,
            "chunks": []
        }
        for j, chunk_text in enumerate(chunks):
            chunk = {
                "parent_window_index": i,
                "index_in_window": j,
                "position": len(window["chunks"]),
                "content": chunk_text,
                "metadata": {
                    "section_heading": window["heading"],
                }
            }
            window["chunks"].append(chunk)
        windows_and_chunks.append(window)

    return windows_and_chunks

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

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Ingest a .docx file and process it into structured data.")
    parser.add_argument("file_path", help="The path to the .docx file to ingest.")
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

    if not os.path.exists(args.file_path):
        print(f"Error: File not found at {args.file_path}")
    elif not args.file_path.lower().endswith('.docx'):
        print(f"Error: File is not a .docx file: {args.file_path}")
    else:
        # Ingest the document
        windows_and_chunks = ingest_document(args.file_path)
        
        all_chunks = []
        for window in windows_and_chunks:
            all_chunks.extend(window['chunks'])
        
        if not all_chunks:
            print("No chunks were generated from the document. Exiting.")
        else:
            print("--- Initializing ---")
            embedding_model_instance: Optional[SentenceTransformer] = None
            qdrant_client_instance: Optional[QdrantClient] = None

            try:
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
                    all_chunks,
                    args.batch_size,
                    args.id_field,
                    args.vector_name
                )

                print("\n--- Script execution completed successfully. ---")

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