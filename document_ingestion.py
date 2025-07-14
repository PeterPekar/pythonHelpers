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
    # Create a dummy docx for testing
    doc = Document()
    doc.add_heading('This is a title', 0)
    p = doc.add_paragraph('A plain paragraph having some ')
    p.add_run('bold').bold = True
    p.add_run(' and some ')
    p.add_run('italic.').italic = True

    doc.add_heading('This is a heading', level=1)
    doc.add_paragraph('Intense quote', style='Intense Quote')

    doc.add_paragraph(
        'first item in unordered list', style='List Bullet'
    )
    doc.add_paragraph(
        'first item in ordered list', style='List Number'
    )
    file_path = "sample_document.docx"
    doc.save(file_path)

    # Ingest the document
    pipeline_output = ingest_document(file_path)

    # Print the output
    import json
    print(json.dumps(pipeline_output, indent=2))

    # Clean up the dummy file
    os.remove(file_path)
