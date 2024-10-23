import os
import time
from pathlib import Path
import pandas as pd
import sqlite3
from sqlite_utils.db import Database
import subprocess
from multiprocessing.pool import ThreadPool

def find_pdf_files(root_path):
    pdf_files = []
    for path in Path(root_path).rglob('*.pdf'):
        if path.is_file():
            pdf_files.append(path)
    print(f'found {len(pdf_files)} pdf files')
    return pdf_files

def get_pdfinfo_metadata(pdf_path):
    try:
        result = subprocess.run(['pdfinfo', str(pdf_path)], capture_output=True, text=True, check=False)
        return result.stdout
    except Exception as e:
        return f"Error: {e}"

def extract_pdfinfo_data(metadata_str):
    data = {}
    for line in metadata_str.strip().split('\n'):
        key, value = line.split(':')
        key = key.strip()
        value = value.strip()
        if key == 'Title' or key == 'Author':
            value = value.replace('"', '\\"')  # Escape double quotes
        data[key] = value
    return data

def get_pdftotext_content(pdf_path):
    try:
        start_time = time.time()
        result = subprocess.run(['pdftotext', '-raw', '-enc', 'UTF-8', str(pdf_path), '-'], capture_output=True, text=True)
        end_time = time.time()
        return {
            'content': result.stdout,
            'execution_time': end_time - start_time
        }
    except Exception as e:
        return f"Error: {e}"

def store_pdf_data(db_path, pdf_path):
    db = Database(db_path)
    
    # Extract PDF info metadata
    pdfinfo_metadata = get_pdfinfo_metadata(pdf_path)
    pdfinfo_data = extract_pdfinfo_data(pdfinfo_metadata)
    
    # Get PDF content and execution time
    pdftotext_result = get_pdftotext_content(pdf_path)

    print(f"{pdf_path} {len(pdftotext_result)}")
    
    # Store data in SQLite database
    db['pdfs'].insert({
        'path': str(pdf_path),
        'content': pdfinfo_data.get('Content', ''),
        'title': pdfinfo_data.get('Title', ''),
        'author': pdfinfo_data.get('Author', ''),
        'creation_date': pdfinfo_data.get('CreationDate', ''),
        'modification_date': pdfinfo_data.get('ModDate', ''),
        'size_bytes_pdf': os.path.getsize(pdf_path),
        'time_execution_pdftotext': pdftotext_result.get('execution_time', 0),
        'size_bytes_pdftotext_result': len(pdftotext_result.get('content', '')),
        'date_created': pd.to_datetime('now')
    }, replace=True)
    
    # Store pdfinfo metadata in a separate table
    db['pdfs_info'].insert({
        'path': str(pdf_path),
        'metadata': pdfinfo_metadata,
        'date_updated': pd.to_datetime('now')
    }, replace=True)

def process_pdf_files(pdf_files, db_path):
    db = Database(db_path)
    
    if not db['pdfs'].exists():
        db['pdfs'].create({
            'path': str,
            'content': str,
            'title': str,
            'author': str,
            'creation_date': str,
            'modification_date': str,
            'size_bytes_pdf': int,
            'time_execution_pdftotext': float,
            'size_bytes_pdftotext_result': int,
            'date_created': pd.Timestamp
        })
    
    if not db['pdfs_info'].exists():
        db['pdfs_info'].create({
            'path': str,
            'metadata': str,
            'date_updated': pd.Timestamp
        })
    
    with ThreadPool() as pool:
        for pdf_path in pdf_files:
            print(pdf_path)
            pool.apply_async(store_pdf_data, args=(db_path, pdf_path))
    
    pool.close()
    pool.join()

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Process PDF files and store metadata in a SQLite database.')
    parser.add_argument('input_paths', nargs='+', help='One or more input paths to search for PDF files.')
    parser.add_argument('--db-path', default='pdfs.db', help='Path to the SQLite database file (default: pdfs.db)')

    args = parser.parse_args()

    pdf_files = []
    for root_path in args.input_paths:
        pdf_files.extend(find_pdf_files(root_path))

    if not pdf_files:
        print("No PDF files found.")
        return

    process_pdf_files(pdf_files, args.db_path)

if __name__ == '__main__':
    main()
