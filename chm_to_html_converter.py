import argparse
import os
import subprocess
import tempfile
import shutil
from html.parser import HTMLParser
import sys

class BodyExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_body = False
        self.body_content = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == 'body':
            self.in_body = True

    def handle_endtag(self, tag):
        if tag.lower() == 'body':
            self.in_body = False

    def handle_data(self, data):
        if self.in_body:
            self.body_content.append(data)

    def get_body_content(self):
        return "".join(self.body_content)

def decompile_chm(chm_path, output_dir):
    """
    Decompiles a CHM file using hh.exe.
    """
    print(f"Decompiling {chm_path} to {output_dir}...")
    try:
        # Ensure hh.exe is available. It's typically in C:\Windows\hh.exe
        # Or can be found via system path.
        # Forcing a specific path might be needed if not in PATH on some systems,
        # but usually hh.exe is accessible.
        result = subprocess.run(
            ["hh.exe", "-decompile", output_dir, chm_path],
            capture_output=True,
            text=True,
            check=False  # We'll check manually for better error reporting
        )
        if result.returncode != 0:
            # hh.exe on success might return 1 if a window was auto-closed.
            # More reliable is to check if output_dir was populated.
            # However, hh.exe is quirky. A common success indicator is if files appear.
            # If stdout/stderr has specific error messages, that's more telling.
            print(f"Warning: hh.exe exited with code {result.returncode}.")
            if result.stdout:
                print(f"hh.exe stdout:\n{result.stdout}")
            if result.stderr:
                print(f"hh.exe stderr:\n{result.stderr}")
            # Check if any files were extracted as a basic success heuristic
            if not os.listdir(output_dir):
                 raise Exception(f"hh.exe failed to decompile. No files found in {output_dir}.")
            else:
                print("hh.exe seems to have extracted files despite non-zero exit code. Proceeding cautiously.")

        print("Decompilation possibly completed.")
        return True
    except FileNotFoundError:
        print("Error: hh.exe not found. Please ensure it is installed and in your system's PATH.")
        print("On most Windows systems, hh.exe is part of the OS.")
        return False
    except Exception as e:
        print(f"An error occurred during decompilation: {e}")
        return False

def extract_html_bodies_from_dir(source_dir):
    """
    Extracts and concatenates the body content of all HTML files in a directory.
    HTML files are processed in alphabetical order of their paths for consistency.
    """
    print(f"Extracting HTML content from {source_dir}...")
    html_contents = []

    file_paths = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith((".html", ".htm")):
                file_paths.append(os.path.join(root, file))

    file_paths.sort() # Process files in a consistent order

    for filepath in file_paths:
        print(f"Processing: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            parser = BodyExtractor()
            parser.feed(content)
            body = parser.get_body_content()
            if body:
                # Add a separator or header indicating the source file
                relative_path = os.path.relpath(filepath, source_dir)
                html_contents.append(f"\n<hr>\n<h2>Content from: {relative_path}</h2>\n{body}\n")
            else:
                print(f"No body content extracted from {filepath}")
        except Exception as e:
            print(f"Could not read or parse file {filepath}: {e}")

    if not html_contents:
        print("Warning: No HTML content was extracted.")
        return ""

    return "".join(html_contents)

def create_combined_html(bodies_content, title):
    """
    Wraps the combined HTML body content in a basic HTML structure.
    """
    if not bodies_content:
        bodies_content = "<p>No content could be extracted from the CHM file.</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: sans-serif; line-height: 1.6; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        hr {{ margin-top: 20px; margin-bottom: 20px; border: 0; border-top: 1px solid #eee; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    {bodies_content}
</body>
</html>
"""

def main():
    parser = argparse.ArgumentParser(description="Convert a CHM file to a single HTML file.")
    parser.add_argument("chm_file", help="Path to the input CHM file.")
    parser.add_argument("-o", "--output", help="Path to the output HTML file. Defaults to <chm_filename>.html next to the CHM file.")

    if len(sys.argv) == 1: # No arguments provided
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    chm_file_path = os.path.abspath(args.chm_file)

    if not os.path.exists(chm_file_path):
        print(f"Error: CHM file not found at {chm_file_path}")
        sys.exit(1)

    if args.output:
        output_file_path = os.path.abspath(args.output)
    else:
        base_name = os.path.splitext(os.path.basename(chm_file_path))[0]
        output_file_path = os.path.join(os.path.dirname(chm_file_path), f"{base_name}.html")

    # Create a temporary directory for decompiled files
    temp_dir = tempfile.mkdtemp(prefix="chm_extract_")
    print(f"Created temporary directory: {temp_dir}")

    try:
        if not decompile_chm(chm_file_path, temp_dir):
            # Error message already printed by decompile_chm
            sys.exit(1)

        # Check if decompilation actually produced any files
        if not os.listdir(temp_dir):
            print(f"Error: Decompilation did not produce any files in {temp_dir}. The CHM might be empty, protected, or an issue occurred with hh.exe.")
            sys.exit(1)

        combined_body_content = extract_html_bodies_from_dir(temp_dir)

        chm_filename = os.path.basename(chm_file_path)
        output_title = f"Combined Content - {chm_filename}"

        final_html = create_combined_html(combined_body_content, output_title)

        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(final_html)
            print(f"Successfully converted CHM to HTML: {output_file_path}")
        except IOError as e:
            print(f"Error writing output file {output_file_path}: {e}")
            sys.exit(1)

    finally:
        if os.path.exists(temp_dir):
            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
