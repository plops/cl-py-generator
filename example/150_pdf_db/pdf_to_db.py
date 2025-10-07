import argparse
import sqlite_utils
import subprocess
import time
import json
from pathlib import Path
from multiprocessing import Pool
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

def process_pdf(pdf_path):
    """Processes a single PDF file."""
    pdf_path = Path(pdf_path)
    start_time = time.time()

    try:
        print(f"Processing PDF: {pdf_path}")

        # pdftotext
        pdftotext_process = subprocess.run(
            ["pdftotext", "-raw", "-enc", "UTF-8", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            check=True
        )
        pdftotext_output = pdftotext_process.stdout
        pdftotext_size = len(pdftotext_output.encode("utf-8"))
        processing_time = time.time() - start_time

        # pdfinfo
        pdfinfo_process = subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True, check=True)
        pdfinfo_output = pdfinfo_process.stdout
        pdfinfo_data = {}
        for line in pdfinfo_output.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                pdfinfo_data[key.strip()] = value.strip()

        # pdfinfo -url
        pdfinfo_url_data = []
        try:
            pdfinfo_url_process = subprocess.run(["pdfinfo", "-url", "-enc", "UTF-8", str(pdf_path)],
                                                 capture_output=True, text=True, check=True)
            if pdfinfo_url_process.returncode == 0:
                lines = pdfinfo_url_process.stdout.strip().split('\n')
                if len(lines) > 1:
                    for line in lines[1:]:
                        parts = line.split()
                        if len(parts) == 3:
                            pdfinfo_url_data.append({'Page': int(parts[0]), 'Type': parts[1], 'URL': parts[2]})
        except Exception as e1:
            print(f"Could not get URL info for {pdf_path}: {e1}")
            pass

        return {
            "path": str(pdf_path),
            "text": pdftotext_output,
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "file_size": pdf_path.stat().st_size,
            "processing_time": processing_time,
            "text_size": pdftotext_size,
            "metadata": json.dumps(pdfinfo_data),
            "url_info": json.dumps(pdfinfo_url_data),
            "type": "pdf"
        }
    except (subprocess.CalledProcessError, FileNotFoundError) as e2:
        print(f"Error processing {pdf_path}: {e2}")
        return None
    except Exception as e3:
        print(f"An unexpected error occurred with {pdf_path}: {e3}")
        return None

def process_epub(epub_path):
    """Processes a single EPUB file."""
    epub_path = Path(epub_path)
    start_time = time.time()

    try:
        print(f"Processing EPUB: {epub_path}")
        book = epub.read_epub(str(epub_path))
        full_text = []

        # Extract text from all document items
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            # Extract text from the body, or the whole document if no body is found
            if soup.body:
                full_text.append(soup.body.get_text())
            else:
                full_text.append(soup.get_text())

        content = "\n".join(full_text)
        text_size = len(content.encode("utf-8"))
        processing_time = time.time() - start_time

        # Extract metadata
        metadata = {
            'title': book.get_metadata('DC', 'title'),
            'creator': book.get_metadata('DC', 'creator'),
            'identifier': book.get_metadata('DC', 'identifier'),
            'language': book.get_metadata('DC', 'language'),
            'publisher': book.get_metadata('DC', 'publisher'),
            'description': book.get_metadata('DC', 'description')
        }

        return {
            "path": str(epub_path),
            "text": content,
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "file_size": epub_path.stat().st_size,
            "processing_time": processing_time,
            "text_size": text_size,
            "metadata": json.dumps(metadata),
            "url_info": None, # EPUBs don't have a direct equivalent to pdfinfo -url
            "type": "epub"
        }
    except Exception as e:
        print(f"Error processing {epub_path}: {e}")
        return None

def process_file(file_path):
    """Determines the file type and calls the appropriate processor."""
    if file_path.suffix.lower() == ".pdf":
        return process_pdf(file_path)
    elif file_path.suffix.lower() == ".epub":
        return process_epub(file_path)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_paths", nargs="+", help="Path(s) to search for PDF and EPUB files.")
    parser.add_argument("--db_path", default="documents.db", help="Path to the SQLite database.")
    args = parser.parse_args()

    db = sqlite_utils.Database(args.db_path)

    # Find both PDF and EPUB files
    files_to_process = []
    for input_path in args.input_paths:
        path_obj = Path(input_path)
        if path_obj.is_dir():
            files_to_process.extend(path_obj.rglob("*.pdf"))
            files_to_process.extend(path_obj.rglob("*.epub"))
        elif path_obj.is_file():
            if path_obj.suffix.lower() in [".pdf", ".epub"]:
                files_to_process.append(path_obj)

    print(f"Found {len(files_to_process)} files to process.")

    with Pool() as pool:
        results = pool.imap_unordered(process_file, files_to_process)
        print("Processing finished. Now inserting into database.")

        successful_results = []
        for result in results:
            if result:
                successful_results.append(result)

        if successful_results:
            # Create table if it doesn't exist and insert data
            db["documents"].insert_all(successful_results, pk="path", replace=True)
            print(f"Successfully inserted {len(successful_results)} documents into the database.")
        else:
            print("No documents were successfully processed.")

if __name__ == "__main__":
    main()
