# Qdrant Ollama RAG Client (`qdrant_ollama_rag.py`)

This Python script provides a command-line interface to perform Retrieval Augmented Generation (RAG). It takes a user query, retrieves relevant text chunks from a Qdrant vector database, and then uses these chunks as context to generate an answer using a local Ollama language model.

## Features

*   **Retrieval:**
    *   Connects to a Qdrant instance (local, Docker, on-disk, or Qdrant Cloud).
    *   Embeds the user query using a specified `sentence-transformers` model (must match the model used for ingesting data into Qdrant).
    *   Performs a similarity search in a Qdrant collection to find the most relevant text chunks.
    *   Supports filtering search results by `source_filename` stored in chunk metadata.
*   **Augmentation:**
    *   Constructs a context string from the retrieved chunks, including metadata like source document and section headings.
    *   Manages context length to avoid exceeding LLM limits.
*   **Generation:**
    *   Interacts with a local Ollama instance via its API.
    *   Sends the user query along with the retrieved context to a specified Ollama model (e.g., `llama3`, `mistral`).
    *   Supports both streaming (default) and non-streaming responses from Ollama.
*   **Configurable:** Most parameters are configurable via command-line arguments, including Qdrant/Ollama URLs, model names, retrieval count (`top_k`), etc.

## Requirements

*   Python 3.8+
*   Required Python libraries:
    ```bash
    pip install qdrant-client sentence-transformers torch requests
    ```
    (`torch` is usually a dependency of `sentence-transformers`).
*   **Qdrant Instance:** A running Qdrant vector database populated with text chunks and their embeddings. See the `json_to_qdrant.py` script (or similar) for how to populate Qdrant.
*   **Ollama Instance:** A running Ollama instance with the desired language model(s) pulled.
    *   Install Ollama from [ollama.com](https://ollama.com/).
    *   Pull a model, e.g.: `ollama pull llama3` or `ollama pull mistral`.

## Usage

Run the script from the command line:

```bash
python qdrant_ollama_rag.py "<your_query>" --qdrant_collection <collection_name> [options]
```

**Positional Arguments:**

*   `user_query`: The query string you want to ask. Enclose in quotes if it contains spaces.

**Required Option:**

*   `--qdrant_collection TEXT`: Name of the Qdrant collection to search.

**Optional Arguments:**

**Qdrant Configuration:**
*   `--qdrant_url URL`: URL of the Qdrant instance.
    *   Default: `http://localhost:6333` (or `QDRANT_URL` env var).
*   `--qdrant_path PATH`: Path to a local on-disk Qdrant database directory (overrides `--qdrant_url`).
    *   Default: `None` (or `QDRANT_PATH` env var).
*   `--qdrant_api_key KEY`: API key for connecting to Qdrant Cloud.
    *   Default: `None` (or `QDRANT_API_KEY` env var).
*   `--qdrant_prefer_grpc`: If specified, attempts to use gRPC for Qdrant connection.
*   `--qdrant_timeout INT`: Timeout in seconds for Qdrant client operations.
    *   Default: `30`
*   `--vector_name TEXT`: Name for the vector in Qdrant if using a named vector setup in your collection.
    *   Default: `None` (uses Qdrant's default unnamed vector).

**Embedding Model Configuration:**
*   `--embedding_model TEXT`: Name of the `sentence-transformers` model used for embedding the query (must match the model used for data ingestion).
    *   Default: `'all-MiniLM-L6-v2'`

**Ollama Configuration:**
*   `--ollama_base_url URL`: Base URL for the Ollama API.
    *   Default: `http://localhost:11434` (or `OLLAMA_BASE_URL` env var).
*   `--ollama_model TEXT`: Name of the Ollama model to use for generation (e.g., `llama3`, `mistral`). Ensure this model is pulled in your Ollama instance.
    *   Default: `'llama3'`
*   `--no_stream`: If specified, disables streaming from Ollama and waits for the full response.
*   `--ollama_timeout INT`: Timeout in seconds for Ollama API requests.
    *   Default: `180`

**Retrieval and RAG Configuration:**
*   `--top_k INT`: Number of top similar chunks to retrieve from Qdrant.
    *   Default: `3`
*   `--filter_filename TEXT`: (Optional) Specific `source_filename` (from chunk metadata) to filter Qdrant search results by. Useful for querying a specific document.
    *   Default: `None`
*   `--max_context_chars INT`: Maximum characters for the context string passed to Ollama. Helps prevent exceeding the LLM's context window. Set to `0` for no limit.
    *   Default: `7000`

**Examples:**

1.  **Basic query to a local Qdrant and Ollama:**
    ```bash
    python qdrant_ollama_rag.py "What are the main features of product X?" --qdrant_collection "product_docs"
    ```

2.  **Query with a specific filename filter and more results:**
    ```bash
    python qdrant_ollama_rag.py "How to configure advanced settings for X?" --qdrant_collection "manuals" --filter_filename "product_x_manual.docx" --top_k 5
    ```

3.  **Using a different embedding model and Ollama model, non-streaming:**
    ```bash
    python qdrant_ollama_rag.py "Summarize the installation process." --qdrant_collection "install_guides" --embedding_model "all-mpnet-base-v2" --ollama_model "mistral" --no_stream
    ```

4.  **Connecting to Qdrant Cloud:**
    ```bash
    python qdrant_ollama_rag.py "Tell me about security." --qdrant_collection "cloud_docs" --qdrant_url "https://your-cluster-url.qdrant.cloud:6333" --qdrant_api_key "YOUR_API_KEY"
    ```

## Workflow

1.  **Parse Arguments:** Reads and processes command-line arguments.
2.  **Initialize Services:**
    *   Loads the specified `sentence-transformers` model for query embedding.
    *   Initializes the `QdrantClient` to connect to the Qdrant database.
3.  **Embed Query:** The user's input query is converted into a vector embedding.
4.  **Search Qdrant (Retrieval):**
    *   The query vector is used to search the specified Qdrant collection for the `top_k` most similar chunks.
    *   If `--filter_filename` is provided, the search is restricted to chunks originating from that document.
5.  **Format Context (Augmentation):**
    *   The text content and relevant metadata (source filename, heading hierarchy) from the retrieved chunks are formatted into a single context string.
    *   Context length is managed by `--max_context_chars`.
6.  **Generate Prompt:** A final prompt is constructed, typically including a system message, the retrieved context, and the original user query.
7.  **Query Ollama (Generation):**
    *   The prompt is sent to the specified Ollama model via its API.
    *   The response is either streamed to the console token by token (default) or printed once fully received (`--no_stream`).
8.  **Cleanup:** The Qdrant client connection is closed.

## Troubleshooting

*   **`ModuleNotFoundError`:** Ensure all required libraries (`qdrant-client`, `sentence-transformers`, `torch`, `requests`) are installed in the Python environment you are using to run the script.
*   **Connection Errors (Qdrant or Ollama):**
    *   Verify that your Qdrant instance and Ollama service are running and accessible at the specified URLs/paths.
    *   Check network connectivity, firewalls, and API keys (for Qdrant Cloud).
    *   Ensure the Ollama model specified with `--ollama_model` has been pulled (e.g., `ollama list` to see available models, `ollama pull <modelname>` to get it).
*   **Embedding Model Mismatch:** The `--embedding_model` used with this script *must* be the same model that was used to generate the embeddings stored in your Qdrant collection. Using a different model will lead to poor or meaningless search results.
*   **No Relevant Documents Found:**
    *   Your query might not have good matches in the Qdrant collection.
    *   The `top_k` value might be too low.
    *   The `filter_filename` might be too restrictive or incorrect.
    *   The embedding model might not be effective for your data/query type.
*   **Ollama Errors:** Ollama might return errors if the model name is incorrect, the prompt is too long for its context window, or the service itself has issues. The script attempts to print these errors.

This README should provide a comprehensive guide for users.
