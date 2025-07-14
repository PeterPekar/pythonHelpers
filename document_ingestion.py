import os
import re
from docx import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

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

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Ingest a .docx file and process it into structured data.")
    parser.add_argument("file_path", help="The path to the .docx file to ingest.")
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File not found at {args.file_path}")
    elif not args.file_path.lower().endswith('.docx'):
        print(f"Error: File is not a .docx file: {args.file_path}")
    else:
        # Ingest the document
        pipeline_output = ingest_document(args.file_path)

        # Print the output
        print(json.dumps(pipeline_output, indent=2))