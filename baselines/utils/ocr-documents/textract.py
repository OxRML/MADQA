#!/usr/bin/env python3
"""
Minimalistic PDF OCR script using Amazon Textract with synchronous processing.
Downloads PDFs from the OxRML/MADQA dataset (externally hosted) and processes
them via the Textract API.  Output is JSONL with one page per line.
"""

import argparse
import json
import os
import sys
import io
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datasets import load_dataset, DownloadManager
from pdf2image import convert_from_path


def ocr_pdf_page(image_bytes, region="us-east-1", aws_access_key_id=None, aws_secret_access_key=None):
    """Process a single PDF page image using Amazon Textract synchronous API."""
    client_kwargs = {'region_name': region}
    if aws_access_key_id and aws_secret_access_key:
        client_kwargs['aws_access_key_id'] = aws_access_key_id
        client_kwargs['aws_secret_access_key'] = aws_secret_access_key
    
    textract = boto3.client('textract', **client_kwargs)
    
    # Use synchronous detect_document_text API
    response = textract.detect_document_text(
        Document={'Bytes': image_bytes}
    )
    
    # Extract text from blocks
    text_lines = []
    for block in response['Blocks']:
        if block['BlockType'] == 'LINE':
            text_lines.append(block['Text'])
    
    return "\n".join(text_lines)


def ocr_pdf(pdf_path, region="us-east-1", aws_access_key_id=None, aws_secret_access_key=None):
    """Process PDF by splitting into pages and using synchronous Textract API."""
    # Convert PDF to images (one per page)
    images = convert_from_path(pdf_path, dpi=200)
    total_pages = len(images)
    
    pages_list = []
    
    for page_num, image in enumerate(images, start=1):
        # Convert PIL Image to bytes (PNG format)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        # Process the page with Textract
        text = ocr_pdf_page(img_bytes, region, aws_access_key_id, aws_secret_access_key)
        
        pages_list.append({
            "file": str(pdf_path),
            "service": "textract",
            "page_number": page_num,
            "total_pages": total_pages,
            "text": text
        })
    
    return pages_list


def process_pdf_wrapper(pdf_file, pdf_name, region, aws_access_key_id=None, aws_secret_access_key=None):
    """Wrapper to process a single PDF file and return pages."""
    try:
        pages = ocr_pdf(pdf_file, region, aws_access_key_id, aws_secret_access_key)
        # Update pages to use the dataset filename instead of temp path
        for page in pages:
            page["file"] = pdf_name
        return {"success": True, "file": pdf_name, "pages": pages}
    except Exception as e:
        return {"success": False, "file": pdf_name, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="OCR PDFs from HuggingFace dataset using Amazon Textract synchronous API"
    )
    parser.add_argument("-o", "--output", default="ocr_output.jsonl", 
                       help="Output JSONL file path (default: ocr_output.jsonl)")
    parser.add_argument("-r", "--region", default="us-east-1", 
                       help="AWS region (default: us-east-1)")
    parser.add_argument("-w", "--workers", type=int, default=24,
                       help="Number of parallel workers (default: 24)")
    parser.add_argument("--aws-access-key-id", 
                       help="AWS Access Key ID")
    parser.add_argument("--aws-secret-access-key", 
                       help="AWS Secret Access Key")
    
    args = parser.parse_args()
    
    # Load document URLs from the HuggingFace dataset
    dataset_name = "OxRML/MADQA"
    print(f"Loading document URLs from {dataset_name}...", file=sys.stderr)
    
    try:
        docs = load_dataset(dataset_name, "documents", split="links")
        doc_urls = {r["document"]: r["url"] for r in docs}
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(doc_urls)} documents", file=sys.stderr)
    
    # Download PDFs
    dm = DownloadManager()
    pdf_files = []
    print("Downloading PDFs...", file=sys.stderr)
    
    for pdf_name, url in sorted(doc_urls.items()):
        try:
            pdf_path = dm.download(url)
            pdf_files.append((pdf_path, pdf_name))
        except Exception as e:
            print(f"Error downloading {pdf_name}: {e}", file=sys.stderr)
    
    if not pdf_files:
        print("Error: No PDF files downloaded", file=sys.stderr)
        sys.exit(1)
    
    print(f"Downloaded {len(pdf_files)} PDFs", file=sys.stderr)
    
    # Process PDFs in parallel and write pages to JSONL
    total_pages = 0
    processed_files = 0
    write_lock = Lock()
    
    print(f"Processing {len(pdf_files)} PDFs with {args.workers} workers...", file=sys.stderr)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit all PDF processing tasks
            future_to_pdf = {
                executor.submit(
                    process_pdf_wrapper, 
                    pdf_file, 
                    pdf_name, 
                    args.region,
                    args.aws_access_key_id,
                    args.aws_secret_access_key
                ): pdf_name 
                for pdf_file, pdf_name in pdf_files
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_pdf):
                pdf_name = future_to_pdf[future]
                result = future.result()
                
                if result["success"]:
                    pages = result["pages"]
                    
                    # Thread-safe writing to output file
                    with write_lock:
                        for page in pages:
                            f.write(json.dumps(page, ensure_ascii=False) + '\n')
                    
                    total_pages += len(pages)
                    processed_files += 1
                    print(f"✓ Completed {pdf_name} ({len(pages)} pages)", file=sys.stderr)
                else:
                    print(f"✗ Error processing {pdf_name}: {result['error']}", file=sys.stderr)
    
    print(f"\nResults written to {args.output}", file=sys.stderr)
    print(f"Processed {processed_files}/{len(pdf_files)} files, {total_pages} pages total", file=sys.stderr)


if __name__ == "__main__":
    main()

