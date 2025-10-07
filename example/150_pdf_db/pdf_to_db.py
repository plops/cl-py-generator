
import argparse
import sqlite_utils
import subprocess
import time
import json
import logging
from pathlib import Path
from multiprocessing import Pool
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

# --- Logger Setup ---
logger = logging.getLogger(__name__)

def setup_logging(log_file="processing.log"):
    """Configures logging to console and file."""
    logger.setLevel(logging.INFO)  # Set the lowest level to capture all messages

    # Prevent adding handlers multiple times if the function is called again
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler - writes detailed logs to a file
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Console Handler - writes colored logs to the console
    try:
        # Use colorlog if available for better readability
        import colorlog
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            })
        console_handler = colorlog.StreamHandler()
    except ImportError:
        # Fallback to standard StreamHandler if colorlog is not installed
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

def process_pdf(pdf_path):
    """Processes a single PDF file."""
    pdf_path = Path(pdf_path)
    start_time = time.time()
    logger.info(f"Processing PDF: {pdf_path}")

    try:
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
            logger.warning(f"Could not get URL info for {pdf_path}: {e1}")
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
        logger.error(f"Failed to process PDF {pdf_path}: {e2}")
        return None
    except Exception as e3:
        logger.error(f"An unexpected error occurred with PDF {pdf_path}: {e3}")
        return None

def process_epub(epub_path):
    """Processes a single EPUB file with improved error handling."""
    epub_path = Path(epub_path)
    start_time = time.time()
    logger.info(f"Processing EPUB: {epub_path}")
    full_text = []

    try:
        book = epub.read_epub(str(epub_path))

        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            try:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                if soup.body:
                    text = soup.body.get_text(separator='\n')
                else:
                    text = soup.get_text(separator='\n')
                full_text.append(text)
            except (KeyError, ValueError, AttributeError) as e:
                logger.warning(f"Error extracting text from item {item.file_name} in EPUB {epub_path}: {e}")
                continue  # Skip to the next item

        content = "\n".join(full_text)
        text_size = len(content.encode("utf-8"))
        processing_time = time.time() - start_time

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
            "url_info": None,
            "type": "epub"
        }
    except Exception as e:
        logger.error(f"Failed to process EPUB {epub_path}: {e}")
        return None

def process_file(file_path):
    """Determines the file type and calls the appropriate processor."""
    file_path = Path(file_path)
    if file_path.suffix.lower() == ".pdf":
        return process_pdf(file_path)
    elif file_path.suffix.lower() == ".epub":
        return process_epub(file_path)
    logger.warning(f"Skipping unsupported file type: {file_path}")
    return None

def main():
    parser = argparse.ArgumentParser(description="Process PDF and EPUB files and store their content in a SQLite database.")
    parser.add_argument("input_paths", nargs="+", help="Path(s) to search for PDF and EPUB files.")
    parser.add_argument("--db_path", default="documents.db", help="Path to the SQLite database.")
    parser.add_argument("--log_file", default="processing.log", help="Path to the log file.")
    args = parser.parse_args()

    # Configure logging at the start of the main function
    setup_logging(args.log_file)

    logger.info("Script started.")
    db = sqlite_utils.Database(args.db_path)

    files_to_process = []
    for input_path in args.input_paths:
        path_obj = Path(input_path)
        if path_obj.is_dir():
            files_to_process.extend(path_obj.rglob("*.pdf"))
            files_to_process.extend(path_obj.rglob("*.epub"))
        elif path_obj.is_file():
            if path_obj.suffix.lower() in [".pdf", ".epub"]:
                files_to_process.append(path_obj)

    logger.info(f"Found {len(files_to_process)} files to process.")

    if not files_to_process:
        logger.warning("No files found to process. Exiting.")
        return

    with Pool() as pool:
        logger.info(f"Starting processing pool with {pool._processes} workers.")
        results = pool.imap_unordered(process_file, files_to_process)
        
        successful_results = []
        for result in results:
            if result:
                successful_results.append(result)

        logger.info("Processing finished. Now handling database insertion.")

        if successful_results:
            try:
                db["documents"].insert_all(successful_results, pk="path", replace=True)
                logger.info(f"Successfully inserted or replaced {len(successful_results)} documents in the database.")
            except Exception as e:
                logger.error(f"Failed to insert records into the database: {e}")
        else:
            logger.warning("No documents were successfully processed to be inserted into the database.")
    
    logger.info("Script finished.")

if __name__ == "__main__":
    main()
