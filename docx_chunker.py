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
DEFAULT_CODE_STYLE_NAMES = ["Code", "CodeBlock", "SourceCode", "Courier", "CodeText"]
DEFAULT_MONOSPACE_FONTS = ["Courier New", "Consolas", "Lucida Console", "Menlo", "Monaco", "Fixedsys"]

# --- Helper Functions for DOCX Element Processing (from Phase 1) ---

def get_paragraph_text(p: DocxParagraph) -> str:
    """Extracts all text from a paragraph, stripping leading/trailing whitespace."""
    return p.text.strip()

def get_paragraph_heading_level(p: DocxParagraph, heading_style_prefix: str) -> int | None:
    """
    Determines the heading level (1-9) of a paragraph if its style name
    starts with heading_style_prefix followed by a number.
    Returns None if it's not a heading.
    """
    if not p.style or not p.style.name: # Ensure style and style.name exist
        return None

    style_name = p.style.name
    normalized_prefix = re.escape(heading_style_prefix)
    match = re.match(rf"{normalized_prefix}\s*(\d+)", style_name, re.IGNORECASE)

    if match:
        try:
            level = int(match.group(1))
            if 1 <= level <= 9: # DOCX supports up to 9 heading levels
                return level
        except ValueError:
            return None # Should not happen if regex matches \d+
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
        if not has_text_in_runs and not p.text.strip(): # Empty paragraph
            return False
        if not has_text_in_runs and p.text.strip(): # Paragraph has text but not in runs? Unlikely.
             return False


        non_empty_runs_are_monospace = True
        text_present_in_paragraph = False
        for run in p.runs:
            if run.text.strip(): # Only consider runs with actual text
                text_present_in_paragraph = True
                if not (run.font and run.font.name and any(mf.lower() == run.font.name.lower() for mf in monospace_fonts)):
                    non_empty_runs_are_monospace = False
                    break
        if text_present_in_paragraph and non_empty_runs_are_monospace: # Ensure there was text to check font for
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
            row_text_parts.append(" ".join(cell_content_parts).strip())

        if any(rtp for rtp in row_text_parts):
            table_text_parts.append(" | ".join(row_text_parts))

    if len(table_text_parts) > 1:
        table_text_parts.append("[TABLE END]")
        return "\n".join(table_text_parts)
    return ""


# --- Phase 2: Heading-Based Section Aggregation --- (Content moved to the final integrated script)
# generate_sections_from_docx and its helpers are now part of the complete script below.


# --- Phase 3: Recursive Character Splitting and Final Output ---

DEFAULT_MAX_SECTION_CHARS = 4000
DEFAULT_TARGET_CHUNK_CHARS = 1000
DEFAULT_CHUNK_OVERLAP_CHARS = 100
DEFAULT_SPLIT_SEPARATORS = [
    "\n\n\n",  # Triple newlines (often separate very distinct blocks)
    "\n\n",    # Double newlines (paragraph breaks)
    "\n",      # Single newlines
    ". ",      # Sentence breaks (with space)
    "? ",
    "! ",
    # ", ",      # Clause breaks - can be too aggressive
    " ",       # Word breaks
    ""         # Character breaks (last resort)
]
MIN_CHUNK_SIZE_CHARS = 50 # Avoid tiny chunks from splitting

def recursive_character_split(text: str, separators: list[str],
                              target_size: int, overlap_size: int
                              ) -> list[str]:
    """
    Recursively splits text into chunks aiming for target_size.
    Inspired by LangChain's RecursiveCharacterTextSplitter.
    """
    final_chunks = []

    if not text.strip(): # Handle empty or whitespace-only text
        return []

    if len(text) <= target_size and len(text) >= MIN_CHUNK_SIZE_CHARS:
        final_chunks.append(text)
        return final_chunks
    elif len(text) < MIN_CHUNK_SIZE_CHARS: # Too small to be a meaningful chunk alone
        # This case should ideally be handled by the caller by merging small prior/post chunks
        # For now, if it's the only thing, return it if not empty.
        if text.strip():
             final_chunks.append(text)
        return final_chunks


    current_separator = ""
    effective_separator_for_split = "" # Store the actual separator used for splitting
    for sep in separators:
        if sep == "": # Use empty string as last resort
            current_separator = sep
            effective_separator_for_split = sep # No actual character to split by, implies fixed length
            break
        if sep in text:
            current_separator = sep
            effective_separator_for_split = sep
            break
    else: # If no separator found in text (except possibly ""), use ""
        current_separator = ""
        effective_separator_for_split = ""


    # Logic for splitting based on chosen separator
    if effective_separator_for_split != "":
        splits = text.split(effective_separator_for_split)
    else: # Forced character split if no other separator worked or was chosen
        # If splitting by char, chunk_step should account for overlap to avoid re-generating same chunk
        chunk_step = target_size - overlap_size
        if chunk_step <=0: chunk_step = target_size # avoid issues if overlap is too large

        for i in range(0, len(text), chunk_step):
            chunk = text[i:i + target_size]
            if len(chunk.strip()) >= MIN_CHUNK_SIZE_CHARS:
                final_chunks.append(chunk)
        return final_chunks # Character split is the final strategy, no further recursion on these parts

    # Process splits from a found separator
    current_doc_buffer = ""
    for i, s_part in enumerate(splits):
        part_to_add = s_part
        # Add back the separator if this is not the first part after a split by a non-empty separator
        if i > 0 and effective_separator_for_split:
            part_to_add = effective_separator_for_split + s_part

        if len(current_doc_buffer) + len(part_to_add) <= target_size:
            current_doc_buffer += part_to_add
        else:
            # Finalize current buffer if it's meaningful
            if len(current_doc_buffer.strip()) >= MIN_CHUNK_SIZE_CHARS:
                final_chunks.append(current_doc_buffer)

            # Start new buffer with current part; if it's too large, it will be recursed upon
            current_doc_buffer = s_part # Store the part *without* the leading separator for now
                                        # as it might be the start of a new large chunk to be split

    # Add any remaining content in buffer
    if len(current_doc_buffer.strip()) >= MIN_CHUNK_SIZE_CHARS:
        final_chunks.append(current_doc_buffer)

    # Post-process the chunks: recurse on any that are still too large, apply overlap
    processed_chunks = []
    for chunk in final_chunks:
        if len(chunk) > target_size:
            # Determine next set of separators for recursion
            next_separators = separators
            if current_separator != "": # If we used a specific separator
                try:
                    sep_idx = separators.index(current_separator)
                    next_separators = separators[sep_idx + 1:] # Try next ones
                    if not next_separators : next_separators = [""] # Must have at least char split
                except ValueError: # Should not happen if current_separator was from separators
                    next_separators = [""]

            processed_chunks.extend(recursive_character_split(chunk, next_separators, target_size, overlap_size))
        elif len(chunk.strip()) >= MIN_CHUNK_SIZE_CHARS: # Ensure chunk is not just whitespace
            processed_chunks.append(chunk)

    final_chunks = processed_chunks

    # Apply overlap
    if overlap_size > 0 and len(final_chunks) > 1:
        overlapped_chunks = [final_chunks[0]]
        for i in range(1, len(final_chunks)):
            # Get overlap from the previous *original* chunk (before it might have been re-chunked due to its own overlap)
            # This is tricky. For simplicity, let's take from the immediate previous in current final_chunks.
            overlap_text = final_chunks[i-1][-overlap_size:]
            current_chunk_text = final_chunks[i]

            # Avoid prepending overlap if it's already the start of the current chunk
            if not current_chunk_text.startswith(overlap_text):
                 overlapped_chunks.append(overlap_text + current_chunk_text)
            else:
                 overlapped_chunks.append(current_chunk_text) # Already contains overlap

        return [c for c in overlapped_chunks if len(c.strip()) >= MIN_CHUNK_SIZE_CHARS] # Filter out tiny chunks again

    return [c for c in final_chunks if len(c.strip()) >= MIN_CHUNK_SIZE_CHARS]


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

        # print(f"\nProcessing section {section_idx+1} (len: {len(section_content)} chars) under: {' > '.join(section_metadata['heading_hierarchy'] if section_metadata['heading_hierarchy'] else ['(No Heading)']) }")

        if len(section_content) > max_section_chars and max_section_chars > 0 : # max_section_chars = 0 means no secondary split
            # print(f"  Section needs splitting (current size: {len(section_content)}, max: {max_section_chars}). Target chunk: {target_chunk_chars}")
            sub_chunks = recursive_character_split(
                section_content,
                split_separators,
                target_chunk_chars,
                chunk_overlap_chars
            )
            # print(f"  Split into {len(sub_chunks)} sub-chunks.")
        else:
            sub_chunks = [section_content] if section_content.strip() else []

        for content_piece in sub_chunks:
            stripped_content = content_piece.strip()
            if not stripped_content: # Skip empty or whitespace-only chunks
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
                        help="Pipe-separated list of separators for recursive splitting, e.g. '\\n\\n|\\n|.'")


    args = parser.parse_args()

    code_style_names_list = [s.strip().lower() for s in args.code_styles.split(',')]
    monospace_fonts_list = [f.strip().lower() for f in args.mono_fonts.split(',')]

    parsed_separators = [sep.replace("\\n", "\n") for sep in args.split_separators.split('|')]


    if os.path.exists(args.input_file):
        doc_filename = os.path.basename(args.input_file)

        # Phase 2 logic is now part of generate_sections_from_docx, called from main
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
                    print(f"\nChunk {i+1}:")
                    print(f"  Metadata: {chunk_data['metadata']}")
                    print(f"  Content Preview (first 150 chars): {chunk_data['content'][:150]}...")
                    print(f"  Content Length: {len(chunk_data['content'])}")
            else:
                print("No final chunks were generated after splitting.")
        else:
            print("No sections were generated from the document, so no chunks to process.")
    else:
        print(f"Error: Input file not found: {args.input_file}")
