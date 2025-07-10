# DOCX Document Chunker for RAG

This Python script processes `.docx` files, particularly technical documentation, and splits their content into smaller, manageable text chunks. These chunks, along with associated metadata (like heading hierarchy), are designed to be suitable for ingestion into vector databases for Retrieval Augmented Generation (RAG) applications.

## Features

*   Parses `.docx` files using the `python-docx` library.
*   **Hierarchical Chunking:** Primarily groups content under document headings (H1-H6).
*   **Secondary Splitting:** If a section under a heading is too large, it's further broken down using a recursive character splitting strategy.
*   **Table Handling:** Extracts text from tables and includes it in a structured string format within chunks.
*   **Code Block Detection:** Heuristically identifies code blocks (based on paragraph styles or monospace fonts) and includes their content, marked appropriately.
*   **Metadata Generation:** Each chunk includes metadata:
    *   `source_filename`: The original DOCX filename.
    *   `heading_hierarchy`: A list of heading texts defining the chunk's context.
    *   `doc_chunk_id`: A sequential ID for the chunk within the document.
    *   `estimated_char_count`: Approximate character length of the chunk's text content.
*   **Configurable:** Chunking behavior can be adjusted via command-line arguments.
*   **JSON Output:** Outputs chunks and their metadata as a JSON file.

## Requirements

*   Python 3.8+
*   `python-docx` library: Install using pip:
    ```bash
    pip install python-docx
    ```

## Usage

Run the script from the command line:

```bash
python docx_chunker.py <input_docx_file> <output_json_file> [options]
```

**Positional Arguments:**

*   `input_docx_file`: Path to the input `.docx` document to be chunked.
*   `output_json_file`: Path where the output JSON file containing the chunks will be saved.

**Optional Arguments:**

*   `--heading_prefix TEXT`: Prefix for identifying heading styles (e.g., "Heading" for "Heading 1", "Heading 2").
    *   Default: `"Heading"`
*   `--code_styles TEXT`: Comma-separated list of paragraph style names to be treated as code blocks (case-insensitive).
    *   Default: `"Code,CodeBlock,SourceCode,Courier,CodeText"`
*   `--mono_fonts TEXT`: Comma-separated list of monospace font names to help identify code blocks if no specific style is used (case-insensitive).
    *   Default: `"Courier New,Consolas,Lucida Console,Menlo,Monaco,Fixedsys"`
*   `--max_section_chars INT`: Maximum characters for a heading-defined section before it undergoes secondary recursive splitting. Set to `0` for no limit (each heading section becomes one chunk unless empty).
    *   Default: `4000`
*   `--target_chunk_chars INT`: Target character size for chunks created by the recursive splitter. The actual chunk size may vary.
    *   Default: `1000`
*   `--chunk_overlap_chars INT`: Character overlap between consecutively recursively split chunks.
    *   Default: `100`
*   `--split_separators TEXT`: Pipe (`|`) separated list of strings to use as separators for recursive splitting, in order of preference. Use `\\n` for newline characters.
    *   Default: `"\\n\\n\\n|\\n\\n|\\n|. |? |! | |"` (Note: the last separator is an empty string for character-level splitting as a last resort).

**Example:**

```bash
python docx_chunker.py "my_technical_manual.docx" "manual_chunks.json" --target_chunk_chars 750 --chunk_overlap_chars 75
```

This command will:
1.  Process `my_technical_manual.docx`.
2.  Group content by heading styles (e.g., "Heading 1", "Heading 2").
3.  If any such section's text exceeds 4000 characters (default `max_section_chars`), it will be split further.
4.  The further splitting will aim for chunks of around 750 characters (`target_chunk_chars`) with a 75-character overlap (`chunk_overlap_chars`).
5.  The resulting chunks will be saved in `manual_chunks.json`.

## Chunking Strategy Explained

1.  **Document Parsing:** The script reads the `.docx` file element by element (paragraphs and tables).
2.  **Heading-Based Sectioning:**
    *   It identifies paragraphs styled as headings (e.g., "Heading 1", "Heading 2", etc.) based on the `--heading_prefix`.
    *   All content (paragraphs, tables, code blocks) following a heading is considered part of that heading's section, respecting the document's hierarchy. For example, content after an H2 is part of that H2 section until a new H1 or H2 is encountered.
    *   The text of the heading itself is included at the beginning of its section's content.
3.  **Content Extraction within Sections:**
    *   **Paragraphs:** Text is extracted directly.
    *   **Tables:** Table content is linearized into a string, with rows separated by newlines and cells by " | ". Tables are enclosed in `[TABLE START]` and `[TABLE END]` markers.
    *   **Code Blocks:** Identified heuristically (by style name or monospace font). Code content is enclosed in `[CODE BLOCK START]` and `[CODE BLOCK END]` markers.
4.  **Secondary Splitting (Recursive Character Splitting):**
    *   After a full section's content is aggregated, if its total character count exceeds `--max_section_chars`, the script applies a recursive splitting algorithm to this section's text.
    *   This algorithm tries to split the text using the provided `--split_separators` (from most significant, like paragraph breaks, to least significant, like spaces).
    *   It aims to create sub-chunks of roughly `--target_chunk_chars` length, with `--chunk_overlap_chars` to maintain context between these sub-chunks.
    *   If a section's content is already below `--max_section_chars`, it typically forms a single chunk (unless it's empty).
5.  **Metadata:** Each final chunk, whether it's a whole section or a sub-chunk from splitting, gets associated metadata:
    *   `source_filename`: Name of the DOCX file.
    *   `heading_hierarchy`: A list of heading texts showing its place in the document structure (e.g., `["Main Topic", "Sub-section Title"]`).
    *   `doc_chunk_id`: A unique sequential ID for that chunk within the processed document.
    *   `estimated_char_count`: The character length of the chunk's text content.

## Output JSON Format

The output JSON file will contain a single list, where each item is an object representing a chunk:

```json
[
  {
    "content": "Text content of the first chunk...",
    "metadata": {
      "source_filename": "example.docx",
      "heading_hierarchy": ["Chapter 1: Introduction"],
      "doc_chunk_id": 1,
      "estimated_char_count": 985
    }
  },
  {
    "content": "Text content of the second chunk, possibly with overlap from the first...",
    "metadata": {
      "source_filename": "example.docx",
      "heading_hierarchy": ["Chapter 1: Introduction"],
      "doc_chunk_id": 2,
      "estimated_char_count": 950
    }
  },
  {
    "content": "[TABLE START]\nHeader A | Header B\nData 1A | Data 1B\n[TABLE END]",
    "metadata": {
      "source_filename": "example.docx",
      "heading_hierarchy": ["Chapter 2: System Components", "Section 2.1: Tables"],
      "doc_chunk_id": 3,
      "estimated_char_count": 75
    }
  }
]
```

## Considerations for RAG

*   **Chunk Size (`--target_chunk_chars`):** This is a critical parameter. It should be small enough to fit within your language model's context window (along with prompts and other context) but large enough to contain meaningful semantic units. Experiment to find what works best for your model and data.
*   **Overlap (`--chunk_overlap_chars`):** Overlap helps maintain context between chunks, reducing the chance that a piece of information relevant to a query is split right at a chunk boundary. Typical values might be 10-20% of the chunk size.
*   **Metadata:** The `heading_hierarchy` metadata is particularly useful. You can potentially include this information when providing context to your RAG model, or use it to help users understand the source of retrieved information.
*   **Table and Code Formatting:** The `[TABLE START/END]` and `[CODE BLOCK START/END]` markers are simple text representations. Your RAG system might need custom parsing or prompting if you want the LLM to specifically "understand" these as tables or code. Alternatively, for tables, you might explore converting them to Markdown format within the chunk if your LLM handles Markdown well.

## Limitations

*   **DOCX Complexity:** `.docx` files can be very complex. This script handles common structures like paragraphs, headings (based on styles), and tables. It may not perfectly interpret very intricate layouts, embedded objects (other than basic text extraction), or highly unusual styling.
*   **Code Block Detection:** Code block detection is heuristic and relies on common style names or font formatting. It might not catch all code blocks or might misidentify some formatted text as code.
*   **Recursive Splitter:** The `recursive_character_split` function is a rule-based implementation. While it aims to be effective, its behavior can vary with different texts and separator configurations.
*   **No Page Numbers:** The `python-docx` library does not inherently understand page numbers, as DOCX is a flow-based format. Thus, page number metadata is not included.
*   **Language:** Primarily designed for English-like text structures regarding sentence terminators (`.`, `?`, `!`) if used in separators.
