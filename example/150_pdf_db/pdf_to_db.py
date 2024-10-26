
import argparse
import sqlite_utils
import subprocess
import time
import json
from pathlib import Path
from multiprocessing import Pool

def process_pdf(pdf_path):
    """Processes a single PDF file."""
    pdf_path = Path(pdf_path)
    start_time = time.time()

    try:
        print(pdf_path)

        # pdftotext
        pdftotext_process = subprocess.run(
            ["pdftotext", "-raw", "-enc", "UTF-8", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            check=True  # Raise CalledProcessError if command fails
        )
        pdftotext_output = pdftotext_process.stdout
        pdftotext_size = len(pdftotext_output.encode("utf-8"))
        pdftotext_time = time.time() - start_time

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
                if len(lines) > 1:  # Check if there are actual URLs (not just the header)
                    for line in lines[1:]:  # Skip the header line
                        parts = line.split()
                        if len(parts) == 3:  # Avoid lines with potential errors (e.g. blank)
                            pdfinfo_url_data.append({'Page': int(parts[0]), 'Type': parts[1], 'URL': parts[2]})
        except Exception as e:
            print(pdf_path, e)
            pass

        return {
            "path": str(pdf_path),
            "text": pdftotext_output,
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "pdf_size": pdf_path.stat().st_size,
            "pdftotext_time": pdftotext_time,
            "text_size": pdftotext_size,
            "pdfinfo": json.dumps(pdfinfo_data),
            "pdfinfo_url": json.dumps(pdfinfo_url_data)
        }
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error processing {pdf_path}: {e}")
        return None
    except Exception as e:
        print(e)
        return None
    return None

parser = argparse.ArgumentParser()
parser.add_argument("input_paths", nargs="+", help="Path(s) to search for PDF files.")
parser.add_argument("--db_path", default="pdfs.db", help="Path to the SQLite database.")
args = parser.parse_args()

db = sqlite_utils.Database(args.db_path)

pdf_files = []
for input_path in args.input_paths:
    path_obj = Path(input_path)
    if path_obj.is_dir():
        pdf_files.extend(path_obj.rglob("*.pdf"))
    elif path_obj.is_file() and path_obj.suffix == ".pdf":
        pdf_files.append(path_obj)




with Pool() as pool:
    results = pool.imap_unordered(process_pdf, pdf_files)

    print("processing finished")

    # Filter out failed runs returning None
    # successful_results = [result for result in results if result]

    successful_results = []
    for result in results:
        try:
            if result:
                successful_results.append(result)
        except Exception as e:
            print(e)
            pass

    db["pdfs"].insert_all(successful_results, pk="path", replace=True)

#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("input_paths", nargs="+", help="Path(s) to search for PDF files.")
#     parser.add_argument("--db_path", default="pdfs.db", help="Path to the SQLite database.")
#     args = parser.parse_args()
#
#     db = sqlite_utils.Database(args.db_path)
#
#     pdf_files = []
#     for input_path in args.input_paths:
#         path_obj = Path(input_path)
#         if path_obj.is_dir():
#             pdf_files.extend(path_obj.rglob("*.pdf"))
#         elif path_obj.is_file() and path_obj.suffix == ".pdf":
#             pdf_files.append(path_obj)
#
#
#
#
#     with Pool() as pool:
#         results = pool.imap_unordered(process_pdf, pdf_files)
#
#         print("processing finished")
#
#         # Filter out failed runs returning None
#         successful_results = [result for result in results if result]
#
#
#         db["pdfs"].insert_all(successful_results, pk="path", replace=True)
#
#
#
# if __name__ == "__main__":
#     main()
