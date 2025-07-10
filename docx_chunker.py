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
DEFAULT_CODE_STYLE_NAMES = ["Code", "CodeBlock", "SourceCode", "Courier", "CodeText", "Fixed Normal"]
DEFAULT_MONOSPACE_FONTS = ["Courier New", "Consolas", "Lucida Console", "Menlo", "Monaco", "Fixedsys", "Courier"]

MIN_CHUNK_SIZE_CHARS = 50 # Avoid tiny chunks from splitting
DEFAULT_MAX_SECTION_CHARS = 4000
DEFAULT_TARGET_CHUNK_CHARS = 1000
DEFAULT_CHUNK_OVERLAP_CHARS = 100
DEFAULT_SPLIT_SEPARATORS = [
    "\n\n\n",  # Triple newlines
    "\n\n",    # Double newlines (paragraph breaks)
    "\n",      # Single newlines
    ". ",      # Sentence breaks (with space after dot)
    "? ",
    "! ",
    # ", ",    # Clause breaks - can be too aggressive
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

def get_table_text(table: DocxTable, code_style_names: list[str], monospace_fonts: list[str]) -> str:
    """
    Extracts text from a table, formatting it row by row, cell by cell.
    Identifies and tags code content within cells.
    """
    table_text_parts = ["[TABLE START]"]
    for row_idx, row in enumerate(table.rows):
        row_text_parts = []
        for cell_idx, cell in enumerate(row.cells):
            cell_content_parts = []
            for p_idx, p in enumerate(cell.paragraphs):
                p_text = get_paragraph_text(p)
                if p_text:
                    if is_paragraph_code(p, code_style_names, monospace_fonts):
                        cell_content_parts.append(f"[CODE BLOCK START]\n{p_text}\n[CODE BLOCK END]")
                    else:
                        cell_content_parts.append(p_text)
            row_text_parts.append(" ".join(cell_content_parts).strip()) # Join paragraphs in cell with space

        if any(rtp for rtp in row_text_parts): # Only add row if it has content
            table_text_parts.append(" | ".join(row_text_parts))

    if len(table_text_parts) > 1: # Has at least one row of content
        table_text_parts.append("[TABLE END]")
        return "\n".join(table_text_parts)
    return "" # Return empty if table (other than marker) is empty


# --- Heading-Based Section Aggregation (Phase 2 core) ---

def generate_sections_from_docx(doc_path: str, heading_style_prefix: str,
                                code_style_names: list[str], monospace_fonts: list[str],
                                doc_filename: str):
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


    for element_idx, element in enumerate(document.element.body):
        element_text_representation = ""

        if element.tag.endswith('p'):
            para = DocxParagraph(element, document)
            text = get_paragraph_text(para)

            if not text and not any(run.element.xpath('.//w:drawing') or run.element.xpath('.//w:pict') for run in para.runs):
                continue # Skip empty paragraphs unless they contain images/drawings

            heading_level = get_paragraph_heading_level(para, heading_style_prefix)

            if heading_level is not None:
                finalize_current_section()
                current_heading_trail = [h for h in current_heading_trail if h[0] < heading_level]
                current_heading_trail.append((heading_level, text))
                element_text_representation = text
            else:
                if is_paragraph_code(para, code_style_names, monospace_fonts):
                    element_text_representation = f"[CODE BLOCK START]\n{text}\n[CODE BLOCK END]"
                else:
                    element_text_representation = text

        elif element.tag.endswith('tbl'):
            table = DocxTable(element, document)
            table_text = get_table_text(table, code_style_names, monospace_fonts)
            if table_text:
                element_text_representation = table_text

        if element_text_representation:
            current_section_content_parts.append(element_text_representation)

    finalize_current_section()

    print(f"--- Generated {len(sections)} initial sections from {doc_filename} ---")
    return sections

# --- Recursive Character Splitting and Final Output (Phase 3 core) ---

def recursive_character_split(text: str, separators: list[str],
                              target_size: int, overlap_size: int
                              ) -> list[str]:
    """
    Recursively splits text into chunks aiming for target_size.
    """
    final_chunks = []
    if not text.strip(): return []

    if len(text) <= target_size:
        if len(text) >= MIN_CHUNK_SIZE_CHARS: final_chunks.append(text)
        elif text.strip(): final_chunks.append(text) # Keep very short if it's the only content
        return final_chunks

    current_separator_for_splitting = ""
    for sep in separators:
        if sep == "": # Last resort
            current_separator_for_splitting = sep
            break
        if sep in text:
            current_separator_for_splitting = sep
            break
    else: # No separator found, should use ""
        current_separator_for_splitting = ""

    if current_separator_for_splitting == "": # Base case: split by length if no separators work
        chunk_step = target_size - overlap_size
        if chunk_step <= 0: chunk_step = max(1, target_size // 2) # Ensure step is positive

        for i in range(0, len(text), chunk_step):
            chunk = text[i:i + target_size]
            if len(chunk.strip()) >= MIN_CHUNK_SIZE_CHARS:
                final_chunks.append(chunk)
            elif i == 0 and chunk.strip(): # if it's the first and only chunk and very small
                final_chunks.append(chunk)
        return final_chunks

    # Split by the chosen separator
    splits = text.split(current_separator_for_splitting)

    good_chunks_from_this_level = []
    buffer = ""
    for i, part in enumerate(splits):
        # Add back separator if it's not the first part and separator is not empty
        part_to_add_to_buffer = (current_separator_for_splitting + part) if (i > 0 and current_separator_for_splitting) else part

        if len(buffer) + len(part_to_add_to_buffer) <= target_size:
            buffer += part_to_add_to_buffer
        else:
            # Finalize buffer if it's meaningful
            if len(buffer.strip()) >= MIN_CHUNK_SIZE_CHARS:
                good_chunks_from_this_level.append(buffer)

            # Start new buffer. If part_to_add_to_buffer itself is too large, it'll be handled by recursion.
            # If part_to_add_to_buffer is small, it starts the new buffer.
            buffer = part # Start new buffer with the current part (without leading separator initially)
                         # If this part is also too large, recursion will handle it.
                         # If the part itself is small, it's the start of a new chunk.

    # Add any remaining content in buffer
    if len(buffer.strip()) >= MIN_CHUNK_SIZE_CHARS:
        good_chunks_from_this_level.append(buffer)
    elif not good_chunks_from_this_level and buffer.strip(): # If it's the only piece and small
        good_chunks_from_this_level.append(buffer)


    # Recursively process any chunks from this level that are still too large
    for chunk in good_chunks_from_this_level:
        if len(chunk) > target_size:
            # Find the index of the current separator to pass the rest for recursion
            next_separators = separators
            try:
                idx_sep = separators.index(current_separator_for_splitting)
                if idx_sep < len(separators) -1 :
                     next_separators = separators[idx_sep+1:]
                else: # No more separators, next recursion must use length based split
                     next_separators = [""]
            except ValueError: # current_separator_for_splitting was ""
                next_separators = [""] # Force length-based split

            final_chunks.extend(recursive_character_split(chunk, next_separators, target_size, overlap_size))
        elif len(chunk.strip()) >= MIN_CHUNK_SIZE_CHARS:
            final_chunks.append(chunk)
        elif not final_chunks and chunk.strip(): # Keep if it's the only result and has content
             final_chunks.append(chunk)

    # Apply overlap (simplified)
    if overlap_size > 0 and len(final_chunks) > 1:
        overlapped_chunks = [final_chunks[0]]
        for i in range(1, len(final_chunks)):
            overlap_text = final_chunks[i-1][-overlap_size:]
            current_chunk_text = final_chunks[i]
            if not current_chunk_text.startswith(overlap_text): # Avoid duplicating overlap
                overlapped_chunks.append(overlap_text + current_chunk_text)
            else:
                overlapped_chunks.append(current_chunk_text)
        return [c for c in overlapped_chunks if c.strip()] # Ensure no empty strings after overlap

    return [c for c in final_chunks if c.strip()]


def create_final_chunks(sections: list[dict], max_section_chars: int,
                        target_chunk_chars: int, chunk_overlap_chars: int,
                        split_separators: list[str]):
    """
    Takes sections, applies recursive splitting if needed, and formats final chunks.
    """
    final_chunks_with_metadata = []
    doc_chunk_id_counter = 0

    for section_idx, section in enumerate(sections):
        section_content = section["raw_content"]
        section_metadata = section["metadata"]

        sub_chunks = []
        if not section_content.strip(): # Skip empty sections
            continue

        if max_section_chars > 0 and len(section_content) > max_section_chars :
            # print(f"  Section {section_idx+1} needs splitting (len: {len(section_content)}). Target: {target_chunk_chars}")
            sub_chunks = recursive_character_split(
                section_content,
                split_separators,
                target_chunk_chars,
                chunk_overlap_chars
            )
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
    parser.add_argument("--code_styles", default=",".join(DEFAULT_CODE_STYLE_NAMES),
                        help=f"Comma-separated paragraph style names for code blocks (default: '{','.join(DEFAULT_CODE_STYLE_NAMES)}')")
    parser.add_argument("--mono_fonts", default=",".join(DEFAULT_MONOSPACE_FONTS),
                        help=f"Comma-separated monospace font names for code block detection (default: '{','.join(DEFAULT_MONOSPACE_FONTS)}')")
    parser.add_argument("--max_section_chars", type=int, default=DEFAULT_MAX_SECTION_CHARS,
                        help=f"Max chars for a heading-defined section before recursive split (0 for no limit) (default: {DEFAULT_MAX_SECTION_CHARS})")
    parser.add_argument("--target_chunk_chars", type=int, default=DEFAULT_TARGET_CHUNK_CHARS,
                        help=f"Target char size for recursively split chunks (default: {DEFAULT_TARGET_CHUNK_CHARS})")
    parser.add_argument("--chunk_overlap_chars", type=int, default=DEFAULT_CHUNK_OVERLAP_CHARS,
                        help=f"Char overlap for recursively split chunks (default: {DEFAULT_CHUNK_OVERLAP_CHARS})")
    parser.add_argument("--split_separators", default="|".join(DEFAULT_SPLIT_SEPARATORS).replace("\n","\\n"),
                        help="Pipe-separated list of separators for recursive splitting, e.g. '\\n\\n\\n|\\n\\n|\\n|. | |'")


    args = parser.parse_args()

    code_style_names_list = [s.strip().lower() for s in args.code_styles.split(',')]
    monospace_fonts_list = [f.strip().lower() for f in args.mono_fonts.split(',')]

    custom_separators_str = args.split_separators.split('|')
    parsed_separators = [s.replace("\\n", "\n") for s in custom_separators_str]


    if os.path.exists(args.input_file):
        doc_filename = os.path.basename(args.input_file)

        generated_sections = generate_sections_from_docx(
            args.input_file,
            args.heading_prefix,
            code_style_names_list,
            monospace_fonts_list,
            doc_filename
        )

        if generated_sections:
            final_chunks = create_final_chunks(
                generated_sections,
                args.max_section_chars,
                args.target_chunk_chars,
                args.chunk_overlap_chars,
                parsed_separators
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
