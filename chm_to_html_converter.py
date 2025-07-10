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
        self.current_level = 0  # Depth of nesting
        self.in_li = False # Are we inside an <li> element that could define a level for an object?
        self.object_level = 0 # Level associated with the current object being processed

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        tag_lower = tag.lower()

        if tag_lower == 'ul':
            self.current_level += 1
        elif tag_lower == 'li':
            self.in_li = True
            # The object tag defining the topic is usually a direct child of <li> or inside it.
            # We associate the current list level with this item.
            self.object_level = self.current_level -1 # уровень списка - это уровень вложенности li
            if self.object_level < 0: self.object_level = 0


        elif tag_lower == 'object' and self.in_li: # Object must be within an LI to get its level
            if attrs_dict.get('type', '').lower() == 'text/sitemap':
                # Reset title/path for this new object
                self.current_title = None
                self.current_path = None
        elif tag_lower == 'param' and self.in_li: # Params are relevant if we are in an LI context
            # Check if we are inside an object that is a sitemap entry
            # This requires knowing if the parent object was a sitemap object.
            # This simple parser doesn't track parent tags easily.
            # Assume params are for the current sitemap object if title/path are being built.
            param_name = attrs_dict.get('name', '').lower()
            param_value = attrs_dict.get('value', '')
            if param_name == 'name':
                self.current_title = param_value
            elif param_name == 'local':
                self.current_path = param_value

            # If both title and path are found for the current object, map them
            if self.current_title and self.current_path:
                normalized_path = os.path.normpath(self.current_path).replace('\\', '/')
                self.topic_map[normalized_path] = {
                    'title': self.current_title,
                    'level': self.object_level # Use the level of the parent <li>
                }
                # Reset for next potential object within the same <li> or next <li>
                self.current_title = None
                self.current_path = None


    def handle_endtag(self, tag):
        tag_lower = tag.lower()
        if tag_lower == 'ul':
            if self.current_level > 0: # Should not go below 0
                self.current_level -= 1
        elif tag_lower == 'li':
            self.in_li = False
            # Reset title/path when exiting an LI, as they are scoped to an object within an LI
            self.current_title = None
            self.current_path = None
            self.object_level = 0 # Reset object level

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

    preferred_names = [
        os.path.join(directory, "toc.hhc"),
        os.path.join(directory, "index.hhc"),
        os.path.join(directory, f"{chm_base_name}.hhc")
    ]
    normalized_hhc_files = {os.path.normcase(f): f for f in hhc_files}

    for preferred_path_str in preferred_names:
        normalized_preferred_path = os.path.normcase(preferred_path_str)
        if normalized_preferred_path in normalized_hhc_files:
            print(f"Found preferred HHC file: {normalized_hhc_files[normalized_preferred_path]}")
            return normalized_hhc_files[normalized_preferred_path]

    print(f"Using first HHC file found: {hhc_files[0]}")
    return hhc_files[0]

def decompile_chm(chm_path, output_dir):
    print(f"Decompiling {chm_path} to {output_dir}...")
    try:
        result = subprocess.run(
            ["hh.exe", "-decompile", output_dir, chm_path],
            capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            print(f"Warning: hh.exe exited with code {result.returncode}.")
            if result.stdout: print(f"hh.exe stdout:\n{result.stdout}")
            if result.stderr: print(f"hh.exe stderr:\n{result.stderr}")
            if not os.listdir(output_dir):
                 raise Exception(f"hh.exe failed to decompile. No files found in {output_dir}.")
            else:
                print("hh.exe seems to have extracted files. Proceeding cautiously.")
        print("Decompilation possibly completed.")
        return True
    except FileNotFoundError:
        print("Error: hh.exe not found. Ensure it's in PATH.")
        return False
    except Exception as e:
        print(f"An error occurred during decompilation: {e}")
        return False

def extract_html_bodies_from_dir(source_dir, topic_map=None):
    print(f"Extracting HTML content from {source_dir}...")
    if topic_map is None: topic_map = {}

    html_page_contents = []
    file_paths = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith((".html", ".htm")):
                file_paths.append(os.path.join(root, file))
    file_paths.sort()

    for filepath in file_paths:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            body_parser = BodyExtractor()
            body_parser.feed(content)
            body_content = body_parser.get_body_content()

            relative_path = os.path.relpath(filepath, source_dir).replace('\\', '/')
            topic_info = topic_map.get(relative_path)

            header_html = ""
            if topic_info:
                title = topic_info.get('title', os.path.basename(filepath))
                level = topic_info.get('level', 0) # Default to level 0 if not specified
                h_tag_number = min(level + 1, 6) # level 0 -> H1, level 1 -> H2, ..., capped at H6
                header_html = f"\n<hr>\n<h{h_tag_number}>{title}</h{h_tag_number}>\n"
            elif body_content: # Only add fallback if there's body content
                # Fallback if no title in map (e.g. use filename)
                header_html = f"\n<hr>\n<h3>Page: {os.path.basename(filepath)}</h3>\n"

            if header_html or body_content:
                html_page_contents.append(header_html)
                if body_content:
                    html_page_contents.append(body_content + "\n")
                elif topic_info : # Topic exists in map but page body is empty
                    html_page_contents.append("<p><em>(This topic might be empty or link to a missing page)</em></p>\n")

        except Exception as e:
            print(f"Could not read or parse file {filepath}: {e}")

    if not html_page_contents:
        print("Warning: No HTML content was extracted.")
        return ""
    return "".join(html_page_contents)

def create_combined_html(all_body_content, title):
    if not all_body_content:
        all_body_content = "<p>No content could be extracted or processed.</p>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: sans-serif; line-height: 1.6; margin: 20px; color: #333; }}
        h1 {{ font-size: 2em; border-bottom: 2px solid #666; padding-bottom: 0.3em; margin-top: 0.5em; margin-bottom: 0.7em; color: #111;}}
        h2 {{ font-size: 1.75em; color: #222; margin-top: 1.5em; margin-bottom: 0.5em; border-bottom: 1px solid #ccc; padding-bottom: 0.2em;}}
        h3 {{ font-size: 1.5em; color: #333; margin-top: 1.3em; margin-bottom: 0.4em;}}
        h4 {{ font-size: 1.25em; color: #444; margin-top: 1.2em; margin-bottom: 0.4em;}}
        h5 {{ font-size: 1.1em; color: #555; margin-top: 1.1em; margin-bottom: 0.3em;}}
        h6 {{ font-size: 1em; color: #666; margin-top: 1em; margin-bottom: 0.3em; font-style: italic;}}
        p {{ margin-bottom: 1em; }}
        hr {{ margin-top: 25px; margin-bottom: 25px; border: 0; border-top: 1px dashed #ccc; }}
        em {{ color: #777; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    {all_body_content}
</body>
</html>
"""

def main():
    parser = argparse.ArgumentParser(description="Convert CHM to single HTML with hierarchical headings from HHC.")
    parser.add_argument("chm_file", help="Path to the input CHM file.")
    parser.add_argument("-o", "--output", help="Path to output HTML. Defaults to <chm_filename>.html.")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    chm_file_path = os.path.abspath(args.chm_file)
    chm_base_name = os.path.splitext(os.path.basename(chm_file_path))[0]

    if not os.path.exists(chm_file_path):
        print(f"Error: CHM file not found: {chm_file_path}")
        sys.exit(1)

    output_file_path = args.output if args.output else os.path.join(os.path.dirname(chm_file_path), f"{chm_base_name}.html")
    output_file_path = os.path.abspath(output_file_path)

    temp_dir = tempfile.mkdtemp(prefix="chm_extract_")
    print(f"Created temporary directory: {temp_dir}")

    topic_map = {}
    try:
        if not decompile_chm(chm_file_path, temp_dir): sys.exit(1)
        if not os.listdir(temp_dir):
            print(f"Error: Decompilation produced no files in {temp_dir}.")
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
                if not topic_map: print("Warning: HHC parsing yielded no topic mappings.")
                else: print(f"Successfully parsed {len(topic_map)} topics from HHC file.")
            except Exception as e:
                print(f"Error parsing HHC file {hhc_file_path}: {e}")
        else:
            print("Warning: No HHC file found. Headers will default to filenames.")

        combined_body_content = extract_html_bodies_from_dir(temp_dir, topic_map)

        # The main H1 is for the document itself. Topics from HHC will start from H1 (level 0) or H2 (level 0 from parser +1)
        document_title = f"Documentation: {chm_base_name}"
        final_html = create_combined_html(combined_body_content, document_title)

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
