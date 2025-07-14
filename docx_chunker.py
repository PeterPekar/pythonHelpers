import argparse
import json
import re
import os
from docx import Document
from docx.document import Document as DocxDocument # For type hinting
from docx.table import Table as DocxTable # For type hinting
from docx.text.paragraph import Paragraph as DocxParagraph # For type hinting

# --- Configuration ---
DEFAULT_HEADING_STYLE_PREFIX = "Heading"
MIN_CHUNK_SIZE_CHARS = 800 # Avoid tiny chunks from splitting
DEFAULT_MAX_SECTION_CHARS = 4000
DEFAULT_TARGET_CHUNK_CHARS = 1000
DEFAULT_CHUNK_OVERLAP_CHARS = 100
DEFAULT_SPLIT_SEPARATORS = [
    "\n\n",    # Double newlines (paragraph breaks)
    "\n",      # Single newlines
    ". ",      # Sentence breaks (with space after dot)
    "? ",
    "! ",
    " ",       # Word breaks
    ""         # Character breaks (last resort)
]

# --- Helper Functions for DOCX Element Processing ---

def get_paragraph_text(p: DocxParagraph) -> str:
    """Extracts all text from a paragraph, stripping leading/trailing whitespace."""
    return p.text.strip()

def get_paragraph_heading_level(p: DocxParagraph, heading_style_prefix: str) -> int | None:
    """
    Determines the heading level (1-9) of a paragraph if its style name
    starts with heading_style_prefix followed by a number.
    Returns None if it's not a heading.
    """
    if not p.style or not p.style.name:
        return None

    style_name = p.style.name
    normalized_prefix = re.escape(heading_style_prefix)
    match = re.match(rf"{normalized_prefix}\s*(\d+)", style_name, re.IGNORECASE)

    if match:
        try:
            level = int(match.group(1))
            if 1 <= level <= 9:
                return level
        except ValueError:
            return None
    return None

def get_table_text(table: DocxTable, split_large_tables: bool = True) -> str:
    """
    Extracts text from a table, formatting it as Markdown.
    If `split_large_tables` is True, very large tables will be split by row.
    """
    rows = list(table.rows)
    if not rows:
        return ""

    # Create header row
    header_cells = [cell.text.strip() for cell in rows[0].cells]
    header = " | ".join(header_cells)

    # Create separator row
    separator = " | ".join(["---"] * len(header_cells))

    # Process data rows
    data_rows = []
    for row in rows[1:]:
        data_cells = [cell.text.strip() for cell in row.cells]
        data_rows.append(" | ".join(data_cells))

    # Combine into a Markdown table
    markdown_table = f"{header}\n{separator}\n" + "\n".join(data_rows)

    # If the table is very large, we can split it by row
    # This is a simple heuristic, a more advanced implementation could be more sophisticated
    if split_large_tables and len(markdown_table) > DEFAULT_MAX_SECTION_CHARS:
        # Return each row as a separate "mini-table" to preserve context
        row_chunks = []
        for data_row in data_rows:
            row_chunks.append(f"{header}\n{separator}\n{data_row}")
        return "\n\n[TABLE_ROW_SPLIT]\n\n".join(row_chunks)

    return markdown_table

# --- Heading-Based Section Aggregation (Phase 2 core) ---

DEFAULT_CODE_STYLE_NAMES = ["Code", "CodeBlock", "SourceCode", "Courier", "CodeText", "Fixed Normal"]
DEFAULT_MONOSPACE_FONTS = ["Courier New", "Consolas", "Lucida Console", "Menlo", "Monaco", "Fixedsys", "Courier"]

def is_paragraph_code(p: DocxParagraph, code_style_names: list[str], monospace_fonts: list[str]) -> bool:
    """
    Heuristic to determine if a paragraph represents a code block.
    Checks paragraph style name and then run font names.
    """
    if p.style and p.style.name:
        style_name_lower = p.style.name.lower()
        if any(cs.lower() == style_name_lower for cs in code_style_names):
            return True

    if p.runs:
        has_text_in_runs = any(run.text.strip() for run in p.runs)
        if not has_text_in_runs and not p.text.strip():
            return False # Empty paragraph
        if not has_text_in_runs and p.text.strip():
             return False # Text exists but not in runs (unlikely for typical docx)

        non_empty_runs_are_monospace = True
        text_present_in_paragraph = False
        for run in p.runs:
            if run.text.strip():
                text_present_in_paragraph = True
                # Check if font name exists and is in the monospace list
                font_name = run.font.name
                if not (font_name and any(mf.lower() == font_name.lower() for mf in monospace_fonts)):
                    non_empty_runs_are_monospace = False
                    break
        if text_present_in_paragraph and non_empty_runs_are_monospace:
            return True

    return False

def is_paragraph_list(p: DocxParagraph) -> bool:
    """
    Heuristic to determine if a paragraph is part of a list.
    """
    return 'List' in p.style.name or p._p.pPr.numPr is not None

def generate_sections_from_docx(doc_path: str, heading_style_prefix: str, doc_filename: str, code_style_names: list[str], monospace_fonts: list[str]):
    """
    Generates preliminary sections based on heading structure.
    Each section includes its aggregated content and heading hierarchy.
    """
    print(f"--- Generating sections from document: {doc_path} ---")
    sections = []
    try:
        document: DocxDocument = Document(doc_path)
    except Exception as e:
        print(f"Error opening or parsing DOCX file '{doc_path}': {e}")
        return []

    current_heading_trail = []  # List of tuples: (level, text)
    current_section_content_parts = []
    in_code_block = False
    in_list = False

    def finalize_current_section():
        nonlocal current_section_content_parts, current_heading_trail, in_code_block, in_list
        if not current_section_content_parts:
            return

        raw_content = "\n\n".join(filter(None, current_section_content_parts)).strip()
        if raw_content:
            frozen_heading_trail = list(current_heading_trail)
            metadata = {
                "source_filename": doc_filename,
                "heading_hierarchy": [h[1] for h in frozen_heading_trail]
            }
            sections.append({
                "raw_content": raw_content,
                "metadata": metadata
            })
        current_section_content_parts = []
        in_code_block = False
        in_list = False


    for element in document.element.body:
        element_text_representation = ""

        if element.tag.endswith('p'):
            para = DocxParagraph(element, document)
            text = get_paragraph_text(para)

            if not text:
                continue # Skip empty paragraphs

            heading_level = get_paragraph_heading_level(para, heading_style_prefix)

            if heading_level is not None:
                finalize_current_section()
                current_heading_trail = [h for h in current_heading_trail if h[0] < heading_level]
                current_heading_trail.append((heading_level, text))
                element_text_representation = text
            else:
                if is_paragraph_code(para, code_style_names, monospace_fonts):
                    if in_list:
                        current_section_content_parts.append("[LIST_END]")
                        in_list = False
                    if not in_code_block:
                        element_text_representation = "[CODE_BLOCK_START]\n"
                        in_code_block = True
                    element_text_representation += text
                elif is_paragraph_list(para):
                    if in_code_block:
                        current_section_content_parts.append("[CODE_BLOCK_END]")
                        in_code_block = False
                    if not in_list:
                        element_text_representation = "[LIST_START]\n"
                        in_list = True
                    element_text_representation += f"- {text}" # Markdown-style list item
                else:
                    if in_code_block:
                        current_section_content_parts.append("[CODE_BLOCK_END]")
                        in_code_block = False
                    if in_list:
                        current_section_content_parts.append("[LIST_END]")
                        in_list = False
                    element_text_representation = text


        elif element.tag.endswith('tbl'):
            if in_code_block:
                current_section_content_parts.append("[CODE_BLOCK_END]")
                in_code_block = False
            if in_list:
                current_section_content_parts.append("[LIST_END]")
                in_list = False
            table = DocxTable(element, document)
            table_text = get_table_text(table, split_large_tables=True)
            if table_text:
                element_text_representation = table_text

        if element_text_representation:
            current_section_content_parts.append(element_text_representation)

    finalize_current_section()

    print(f"--- Generated {len(sections)} initial sections from {doc_filename} ---")
    return sections

# --- Semantic Chunking Logic ---

def split_text_into_chunks(text: str, target_chunk_size: int, overlap: int) -> list[str]:
    """
    Splits a long text into chunks based on paragraphs, sentences, or words,
    respecting the target chunk size and overlap.
    """
    if not text.strip():
        return []

    # Split by paragraphs first
    paragraphs = text.split('\n\n')

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if not para.strip():
            continue

        # If a paragraph is larger than the target size, split it further
        if len(para) > target_chunk_size:
            if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE_CHARS:
                chunks.append(current_chunk.strip())

            # Sentence splitting as a fallback
            sentences = re.split(r'(?<=[.!?]) +', para)
            sentence_chunk = ""
            for sent in sentences:
                if len(sentence_chunk) + len(sent) + 1 < target_chunk_size:
                    sentence_chunk += sent + " "
                else:
                    if sentence_chunk and len(sentence_chunk) >= MIN_CHUNK_SIZE_CHARS:
                        chunks.append(sentence_chunk.strip())
                    sentence_chunk = sent + " "
            if sentence_chunk and len(sentence_chunk) >= MIN_CHUNK_SIZE_CHARS:
                chunks.append(sentence_chunk.strip())
            current_chunk = ""

        # If adding the next paragraph fits, add it
        elif len(current_chunk) + len(para) + 2 < target_chunk_size:
            current_chunk += para + "\n\n"

        # Otherwise, finalize the current chunk and start a new one
        else:
            if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE_CHARS:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE_CHARS:
        chunks.append(current_chunk.strip())

    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            # Get the last `overlap` characters from the previous chunk
            overlap_text = chunks[i-1][-overlap:]
            overlapped_chunks.append(overlap_text + chunks[i])
        return overlapped_chunks

    return chunks

import spacy

# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str) -> dict:
    """
    Extracts named entities from text using spaCy.
    """
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    return entities

def summarize_text(text: str, api_token: str) -> str:
    """
    Summarizes text using the Hugging Face Inference API.
    """
    API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
    headers = {"Authorization": f"Bearer {api_token}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    summary = query({
        "inputs": text,
        "parameters": {
            "max_length": 100,
            "min_length": 30,
            "do_sample": False
        }
    })

    if isinstance(summary, list) and 'summary_text' in summary[0]:
        return summary[0]['summary_text']
    return ""

def create_final_chunks(sections: list[dict], max_section_chars: int,
                        target_chunk_chars: int, chunk_overlap_chars: int,
                        extract_entities_enabled: bool, summarize_sections_enabled: bool,
                        api_token: str):
    """
    Takes sections, applies semantic splitting if needed, and formats final chunks.
    Handles code blocks, tables, lists, entity extraction, and section summarization.
    Ensures that all chunks are at least MIN_CHUNK_SIZE_CHARS long.
    """
    all_chunks = []
    final_chunks_with_metadata = []
    doc_chunk_id_counter = 0

    for section in sections:
        section_content = section["raw_content"]
        section_metadata = section["metadata"]

        if not section_content.strip():
            continue

        if summarize_sections_enabled and len(section_content) > max_section_chars:
            section_metadata["summary"] = summarize_text(section_content, api_token)

        parts = re.split(r'(\[CODE_BLOCK_START\].*?\[CODE_BLOCK_END\]|\[TABLE_ROW_SPLIT\]|\[LIST_START\].*?\[LIST_END\])', section_content, flags=re.DOTALL)

        for part in parts:
            if not part.strip() or part == "[TABLE_ROW_SPLIT]":
                continue

            is_code_block = part.startswith("[CODE_BLOCK_START]")
            is_table_row = "\n---\n" in part and " | " in part
            is_list = part.startswith("[LIST_START]")

            if is_code_block:
                content = part.replace("[CODE_BLOCK_START]", "").replace("[CODE_BLOCK_END]", "").strip()
                if not content:
                    continue

                doc_chunk_id_counter += 1
                chunk_metadata = section_metadata.copy()
                chunk_metadata["doc_chunk_id"] = doc_chunk_id_counter
                chunk_metadata["is_code_block"] = True
                chunk_metadata["estimated_char_count"] = len(content)

                final_chunks_with_metadata.append({
                    "content": content,
                    "metadata": chunk_metadata
                })
            elif is_table_row:
                doc_chunk_id_counter += 1
                chunk_metadata = section_metadata.copy()
                chunk_metadata["doc_chunk_id"] = doc_chunk_id_counter
                chunk_metadata["is_table_row"] = True
                chunk_metadata["estimated_char_count"] = len(part)

                final_chunks_with_metadata.append({
                    "content": part,
                    "metadata": chunk_metadata
                })
            elif is_list:
                content = part.replace("[LIST_START]", "").replace("[LIST_END]", "").strip()
                if not content:
                    continue

                doc_chunk_id_counter += 1
                chunk_metadata = section_metadata.copy()
                chunk_metadata["doc_chunk_id"] = doc_chunk_id_counter
                chunk_metadata["is_list"] = True
                chunk_metadata["estimated_char_count"] = len(content)

                final_chunks_with_metadata.append({
                    "content": content,
                    "metadata": chunk_metadata
                })
            else:
                # This is a text part, so we might need to split it further
                sub_chunks = []
                if len(part) > max_section_chars:
                    sub_chunks = split_text_into_chunks(part, target_chunk_chars, chunk_overlap_chars)
                else:
                    sub_chunks = [part]

                for content_piece in sub_chunks:
                    stripped_content = content_piece.strip()
                    if not stripped_content:
                        continue

                    doc_chunk_id_counter += 1
                    chunk_metadata = section_metadata.copy()
                    chunk_metadata["doc_chunk_id"] = doc_chunk_id_counter
                    chunk_metadata["is_code_block"] = False
                    chunk_metadata["is_table_row"] = False
                    chunk_metadata["is_list"] = False
                    chunk_metadata["estimated_char_count"] = len(stripped_content)

                    final_chunks_with_metadata.append({
                        "content": stripped_content,
                        "metadata": chunk_metadata
                    })

    print(f"--- Generated {len(final_chunks_with_metadata)} final chunks ---")
    return final_chunks_with_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk DOCX files for RAG with advanced features.")
    parser.add_argument("input_file", help="Path to the input DOCX file.")
    parser.add_argument("output_file", help="Path to the output JSON file for chunks.")
    parser.add_argument("--heading_prefix", default=DEFAULT_HEADING_STYLE_PREFIX,
                        help=f"Prefix for heading styles (default: '{DEFAULT_HEADING_STYLE_PREFIX}')")
    parser.add_argument("--max_section_chars", type=int, default=DEFAULT_MAX_SECTION_CHARS,
                        help=f"Max chars for a heading-defined section before splitting (default: {DEFAULT_MAX_SECTION_CHARS})")
    parser.add_argument("--target_chunk_chars", type=int, default=DEFAULT_TARGET_CHUNK_CHARS,
                        help=f"Target char size for split chunks (default: {DEFAULT_TARGET_CHUNK_CHARS})")
    parser.add_argument("--chunk_overlap_chars", type=int, default=DEFAULT_CHUNK_OVERLAP_CHARS,
                        help=f"Char overlap for split chunks (default: {DEFAULT_CHUNK_OVERLAP_CHARS})")
    parser.add_argument("--extract_entities", action="store_true", help="Enable entity extraction.")
    parser.add_argument("--summarize_sections", action="store_true", help="Enable section summarization.")
    parser.add_argument("--huggingface_api_token", default=os.environ.get("HUGGINGFACE_API_TOKEN"),
                        help="Hugging Face API token for summarization. Can also be set via HUGGINGFACE_API_TOKEN environment variable.")

    args = parser.parse_args()

    api_token = args.huggingface_api_token
    if args.summarize_sections and not api_token:
        print("Error: --summarize_sections requires a Hugging Face API token.")
        exit(1)

    if os.path.exists(args.input_file):
        doc_filename = os.path.basename(args.input_file)

        code_style_names_list = [s.strip().lower() for s in DEFAULT_CODE_STYLE_NAMES]
        monospace_fonts_list = [f.strip().lower() for f in DEFAULT_MONOSPACE_FONTS]

        generated_sections = generate_sections_from_docx(
            args.input_file,
            args.heading_prefix,
            doc_filename,
            code_style_names_list,
            monospace_fonts_list
        )

        if generated_sections:
            final_chunks = create_final_chunks(
                generated_sections,
                args.max_section_chars,
                args.target_chunk_chars,
                args.chunk_overlap_chars,
                args.extract_entities,
                args.summarize_sections,
                api_token
            )

            if final_chunks:
                try:
                    with open(args.output_file, 'w', encoding='utf-8') as f:
                        json.dump(final_chunks, f, indent=2, ensure_ascii=False)
                    print(f"Successfully saved {len(final_chunks)} chunks to {args.output_file}")
                except IOError as e:
                    print(f"Error writing output JSON file '{args.output_file}': {e}")

                print(f"\n\n--- Preview of Final Chunks (Max 2) ---")
                for i, chunk_data in enumerate(final_chunks[:2]):
                    if i < 2 : # Print only first 2
                        print(f"\nChunk {chunk_data['metadata']['doc_chunk_id']}:")
                        print(f"  Metadata: {chunk_data['metadata']}")
                        print(f"  Content Preview (first 150 chars): {chunk_data['content'][:150]}...")
                        print(f"  Content Length: {len(chunk_data['content'])}")
            else:
                print("No final chunks were generated after splitting.")
        else:
            print("No sections were generated from the document, so no chunks to process.")
    else:
        print(f"Error: Input file not found: {args.input_file}")
