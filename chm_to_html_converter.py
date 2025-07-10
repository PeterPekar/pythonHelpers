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
        self.reset()
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
        return "".join(self.body_content).strip()

class HHCParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.topic_map = {}
        self.current_title = None
        self.current_path = None
        self.in_object = False
        self.is_sitemap_object = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag.lower() == 'object':
            self.in_object = True
            if attrs_dict.get('type', '').lower() == 'text/sitemap':
                self.is_sitemap_object = True
        elif tag.lower() == 'param' and self.is_sitemap_object:
            param_name = attrs_dict.get('name', '').lower()
            param_value = attrs_dict.get('value', '')
            if param_name == 'name':
                self.current_title = param_value
            elif param_name == 'local':
                self.current_path = param_value

    def handle_endtag(self, tag):
        if tag.lower() == 'object':
            if self.is_sitemap_object and self.current_title and self.current_path:
                # Normalize path separators for consistency
                normalized_path = os.path.normpath(self.current_path).replace('\\', '/')
                self.topic_map[normalized_path] = self.current_title
            self.current_title = None
            self.current_path = None
            self.in_object = False
            self.is_sitemap_object = False

    def get_topic_map(self):
        return self.topic_map

def find_hhc_file(directory, chm_base_name):
    """Finds the HHC file in the directory."""
    hhc_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".hhc"):
                hhc_files.append(os.path.join(root, file))

    if not hhc_files:
        return None

    # Prioritize common names
    preferred_names = [
        os.path.join(directory, "toc.hhc"),
        os.path.join(directory, "index.hhc"),
        os.path.join(directory, f"{chm_base_name}.hhc") # CHM base name + .hhc
    ]
    # Normalize paths for comparison
    normalized_hhc_files = {os.path.normcase(f): f for f in hhc_files}

    for preferred_path_str in preferred_names:
        normalized_preferred_path = os.path.normcase(preferred_path_str)
        if normalized_preferred_path in normalized_hhc_files:
            print(f"Found preferred HHC file: {normalized_hhc_files[normalized_preferred_path]}")
            return normalized_hhc_files[normalized_preferred_path]

    # Fallback to the first one found if no preferred name matches
    print(f"Using first HHC file found: {hhc_files[0]}")
    return hhc_files[0]


def decompile_chm(chm_path, output_dir):
    """
    Decompiles a CHM file using hh.exe.
    """
    print(f"Decompiling {chm_path} to {output_dir}...")
    try:
        result = subprocess.run(
            ["hh.exe", "-decompile", output_dir, chm_path],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0:
            print(f"Warning: hh.exe exited with code {result.returncode}.")
            if result.stdout:
                print(f"hh.exe stdout:\n{result.stdout}")
            if result.stderr:
                print(f"hh.exe stderr:\n{result.stderr}")
            if not os.listdir(output_dir):
                 raise Exception(f"hh.exe failed to decompile. No files found in {output_dir}.")
            else:
                print("hh.exe seems to have extracted files despite non-zero exit code. Proceeding cautiously.")
        print("Decompilation possibly completed.")
        return True
    except FileNotFoundError:
        print("Error: hh.exe not found. Please ensure it is installed and in your system's PATH.")
        return False
    except Exception as e:
        print(f"An error occurred during decompilation: {e}")
        return False

def extract_html_bodies_from_dir(source_dir, topic_map=None):
    """
    Extracts and concatenates the body content of all HTML files in a directory.
    Uses topic_map to add headers based on HHC file entries.
    """
    print(f"Extracting HTML content from {source_dir}...")
    if topic_map is None:
        topic_map = {}

    html_page_contents = []

    file_paths = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith((".html", ".htm")):
                file_paths.append(os.path.join(root, file))

    file_paths.sort()

    for filepath in file_paths:
        # print(f"Processing: {filepath}") # Can be verbose
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            body_parser = BodyExtractor()
            body_parser.feed(content)
            body_content = body_parser.get_body_content()

            if body_content or topic_map: # Add header even if body is empty but topic exists
                # Determine header
                relative_path = os.path.relpath(filepath, source_dir).replace('\\', '/')
                topic_title = topic_map.get(relative_path)

                header_text = ""
                if topic_title:
                    header_text = topic_title
                else:
                    # Fallback if no title in map (e.g. use filename, but make it less prominent)
                    # Only add fallback header if there's actual body content
                    if body_content:
                        header_text = f"Page: {os.path.basename(filepath)}"

                # Construct header HTML, only if header_text is not empty
                header_html = ""
                if header_text:
                    # Use h2 for topics from HHC, h3 for fallback page titles
                    header_tag = "h2" if topic_title else "h3"
                    header_html = f"\n<hr>\n<{header_tag}>{header_text}</{header_tag}>\n"

                # Append header (if any) and body content (if any)
                if header_html or body_content:
                    html_page_contents.append(header_html)
                    if body_content:
                         html_page_contents.append(body_content + "\n")
                    else: # If only header (empty page from ToC)
                         html_page_contents.append("<p><em>(This topic might be empty or link to a missing page)</em></p>\n")

            # else: # No body content and no topic map to infer title
                # print(f"No body content extracted from {filepath}")
        except Exception as e:
            print(f"Could not read or parse file {filepath}: {e}")

    if not html_page_contents:
        print("Warning: No HTML content was extracted.")
        return ""

    return "".join(html_page_contents)

def create_combined_html(all_body_content, title):
    """
    Wraps the combined HTML body content in a basic HTML structure.
    """
    if not all_body_content:
        all_body_content = "<p>No content could be extracted or processed from the CHM file.</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: sans-serif; line-height: 1.6; margin: 20px; }}
        h1 {{ font-size: 1.8em; border-bottom: 2px solid #ccc; padding-bottom: 0.3em; margin-bottom: 0.7em;}}
        h2 {{ font-size: 1.4em; color: #333; margin-top: 1.5em; margin-bottom: 0.5em; border-bottom: 1px solid #eee; padding-bottom: 0.2em;}}
        h3 {{ font-size: 1.1em; color: #555; margin-top: 1.2em; margin-bottom: 0.4em;}}
        p {{ margin-bottom: 1em; }}
        hr {{ margin-top: 25px; margin-bottom: 25px; border: 0; border-top: 1px dashed #ccc; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    {all_body_content}
</body>
</html>
"""

def main():
    parser = argparse.ArgumentParser(description="Convert a CHM file to a single HTML file, using HHC for topic titles.")
    parser.add_argument("chm_file", help="Path to the input CHM file.")
    parser.add_argument("-o", "--output", help="Path to the output HTML file. Defaults to <chm_filename>.html next to the CHM file.")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    chm_file_path = os.path.abspath(args.chm_file)
    chm_base_name = os.path.splitext(os.path.basename(chm_file_path))[0]

    if not os.path.exists(chm_file_path):
        print(f"Error: CHM file not found at {chm_file_path}")
        sys.exit(1)

    if args.output:
        output_file_path = os.path.abspath(args.output)
    else:
        output_file_path = os.path.join(os.path.dirname(chm_file_path), f"{chm_base_name}.html")

    temp_dir = tempfile.mkdtemp(prefix="chm_extract_")
    print(f"Created temporary directory: {temp_dir}")

    topic_map = {}
    try:
        if not decompile_chm(chm_file_path, temp_dir):
            sys.exit(1)

        if not os.listdir(temp_dir):
            print(f"Error: Decompilation did not produce any files in {temp_dir}.")
            sys.exit(1)

        hhc_file_path = find_hhc_file(temp_dir, chm_base_name)
        if hhc_file_path:
            print(f"Parsing HHC file: {hhc_file_path}")
            try:
                with open(hhc_file_path, 'r', encoding='utf-8', errors='ignore') as f_hhc:
                    hhc_content = f_hhc.read()
                hhc_parser = HHCParser()
                hhc_parser.feed(hhc_content)
                topic_map = hhc_parser.get_topic_map()
                if not topic_map:
                    print("Warning: HHC parsing did not yield any topic mappings.")
                else:
                    print(f"Successfully parsed {len(topic_map)} topics from HHC file.")
            except Exception as e:
                print(f"Error parsing HHC file {hhc_file_path}: {e}")
        else:
            print("Warning: No HHC file found. Headers will default to filenames.")

        combined_body_content = extract_html_bodies_from_dir(temp_dir, topic_map)

        output_title = f"Combined Content - {chm_base_name}" # Main H1 title for the whole document
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
