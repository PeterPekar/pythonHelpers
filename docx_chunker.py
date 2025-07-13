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
MIN_CHUNK_SIZE_CHARS = 50 # Avoid tiny chunks from splitting
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

def get_table_text(table: DocxTable) -> str:
    """
    Extracts text from a table, formatting it row by row, cell by cell.
    """
    table_text_parts = []
    for row in table.rows:
        row_text_parts = []
        for cell in row.cells:
            cell_text = " ".join([get_paragraph_text(p) for p in cell.paragraphs]).strip()
            row_text_parts.append(cell_text)

        if any(row_text_parts): # Only add row if it has content
            table_text_parts.append(" | ".join(row_text_parts))

    if table_text_parts:
        return "\n".join(table_text_parts)
    return ""

# --- Heading-Based Section Aggregation (Phase 2 core) ---

def generate_sections_from_docx(doc_path: str, heading_style_prefix: str, doc_filename: str):
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

    def finalize_current_section():
        nonlocal current_section_content_parts, current_heading_trail
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
                element_text_representation = text

        elif element.tag.endswith('tbl'):
            table = DocxTable(element, document)
            table_text = get_table_text(table)
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
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Sentence splitting as a fallback
            sentences = re.split(r'(?<=[.!?]) +', para)
            sentence_chunk = ""
            for sent in sentences:
                if len(sentence_chunk) + len(sent) + 1 < target_chunk_size:
                    sentence_chunk += sent + " "
                else:
                    if sentence_chunk:
                        chunks.append(sentence_chunk.strip())
                    sentence_chunk = sent + " "
            if sentence_chunk:
                chunks.append(sentence_chunk.strip())
            current_chunk = ""

        # If adding the next paragraph fits, add it
        elif len(current_chunk) + len(para) + 2 < target_chunk_size:
            current_chunk += para + "\n\n"

        # Otherwise, finalize the current chunk and start a new one
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            # Get the last `overlap` characters from the previous chunk
            overlap_text = chunks[i-1][-overlap:]
            overlapped_chunks.append(overlap_text + chunks[i])
        return overlapped_chunks

    return chunks

def create_final_chunks(sections: list[dict], max_section_chars: int,
                        target_chunk_chars: int, chunk_overlap_chars: int):
    """
    Takes sections, applies semantic splitting if needed, and formats final chunks.
    """
    final_chunks_with_metadata = []
    doc_chunk_id_counter = 0

    for section in sections:
        section_content = section["raw_content"]
        section_metadata = section["metadata"]

        if not section_content.strip():
            continue

        sub_chunks = []
        if len(section_content) > max_section_chars:
            # If the section is too large, split it semantically
            sub_chunks = split_text_into_chunks(section_content, target_chunk_chars, chunk_overlap_chars)
        else:
            sub_chunks = [section_content]

        for content_piece in sub_chunks:
            stripped_content = content_piece.strip()
            if not stripped_content:
                continue

            doc_chunk_id_counter += 1
            chunk_metadata = section_metadata.copy()
            chunk_metadata["doc_chunk_id"] = doc_chunk_id_counter
            chunk_metadata["estimated_char_count"] = len(stripped_content)

            final_chunks_with_metadata.append({
                "content": stripped_content,
                "metadata": chunk_metadata
            })

    print(f"--- Generated {len(final_chunks_with_metadata)} final chunks ---")
    return final_chunks_with_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk DOCX files for RAG.")
    parser.add_argument("input_file", help="Path to the input DOCX file.")
    parser.add_argument("output_file", help="Path to the output JSON file for chunks.")
    parser.add_argument("--heading_prefix", default=DEFAULT_HEADING_STYLE_PREFIX,
                        help=f"Prefix for heading styles (default: '{DEFAULT_HEADING_STYLE_PREFIX}')")
    parser.add_argument("--max_section_chars", type=int, default=DEFAULT_MAX_SECTION_CHARS,
                        help=f"Max chars for a heading-defined section before recursive split (default: {DEFAULT_MAX_SECTION_CHARS})")
    parser.add_argument("--target_chunk_chars", type=int, default=DEFAULT_TARGET_CHUNK_CHARS,
                        help=f"Target char size for recursively split chunks (default: {DEFAULT_TARGET_CHUNK_CHARS})")
    parser.add_argument("--chunk_overlap_chars", type=int, default=DEFAULT_CHUNK_OVERLAP_CHARS,
                        help=f"Char overlap for recursively split chunks (default: {DEFAULT_CHUNK_OVERLAP_CHARS})")

    args = parser.parse_args()

    if os.path.exists(args.input_file):
        doc_filename = os.path.basename(args.input_file)

        generated_sections = generate_sections_from_docx(
            args.input_file,
            args.heading_prefix,
            doc_filename
        )

        if generated_sections:
            final_chunks = create_final_chunks(
                generated_sections,
                args.max_section_chars,
                args.target_chunk_chars,
                args.chunk_overlap_chars,
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
