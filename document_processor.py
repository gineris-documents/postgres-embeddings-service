import os
import sys
import logging
import json
import io
import http.server
import socketserver
import psycopg2
import psycopg2.extras
import time
import numpy as np
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import re
from google.auth import default
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PORT = int(os.environ.get('PORT', 8080))

# PostgreSQL Configuration
PG_HOST = os.environ.get("PG_HOST")
PG_DATABASE = os.environ.get("PG_DATABASE", "postgres")
PG_USER = os.environ.get("PG_USER")
PG_PASSWORD = os.environ.get("PG_PASSWORD")

# Google Drive API configuration
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = os.environ.get("SERVICE_ACCOUNT_FILE", "service-account.json")

# Global variable to store the model
model = None

def get_drive_service():
    """Create and return a Google Drive service."""
    try:
        # First try to use service account file
        if os.path.exists(SERVICE_ACCOUNT_FILE) and os.path.getsize(SERVICE_ACCOUNT_FILE) > 0:
            creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
            service = build('drive', 'v3', credentials=creds)
            logger.info("Successfully connected to Google Drive API using service account")
            return service
        else:
            # Fall back to application default credentials
            logger.warning(f"Service account file '{SERVICE_ACCOUNT_FILE}' not found or empty, using default credentials")
            credentials, _ = default(scopes=SCOPES)
            service = build('drive', 'v3', credentials=credentials)
            logger.info("Successfully connected to Google Drive API using default credentials")
            return service
    except Exception as e:
        logger.error(f"Failed to connect to Google Drive API: {str(e)}")
        raise

def download_file_from_drive(service, file_id):
    """Download a file from Google Drive."""
    try:
        request = service.files().get_media(fileId=file_id)
        file_content = io.BytesIO()
        downloader = MediaIoBaseDownload(file_content, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
            logger.info(f"Download progress: {int(status.progress() * 100)}%")
        
        file_content.seek(0)
        logger.info(f"Successfully downloaded file with ID: {file_id}")
        return file_content
    except HttpError as error:
        logger.error(f"Error downloading file from Google Drive: {error}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error downloading file: {str(e)}")
        raise

def get_postgres_connection():
    """Create and return a connection to PostgreSQL."""
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            database=PG_DATABASE,
            user=PG_USER,
            password=PG_PASSWORD
        )
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
        raise

def load_model():
    """Load the sentence-transformers model."""
    global model
    if model is not None:
        return model
        
    try:
        logger.info("Loading sentence-transformers model...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def get_pending_documents():
    """Get pending documents from the tracking table."""
    conn = get_postgres_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    try:
        cursor.execute(
            """
            SELECT * FROM ai_data.document_tracking 
            WHERE status = 'pending'
            ORDER BY created_at ASC
            LIMIT 1
            """
        )
        pending_docs = cursor.fetchall()
        logger.info(f"Found {len(pending_docs)} pending documents")
        return pending_docs
    except Exception as e:
        logger.error(f"Error fetching pending documents: {str(e)}")
        return []
    finally:
        cursor.close()
        conn.close()

def update_document_status(doc_id, status, error_message=None):
    """Update the status of a document in the tracking table."""
    conn = get_postgres_connection()
    cursor = conn.cursor()
    
    try:
        if error_message:
            cursor.execute(
                """
                UPDATE ai_data.document_tracking 
                SET status = %s, error_message = %s, processed_at = NOW()
                WHERE id = %s
                """,
                (status, error_message, doc_id)
            )
        else:
            cursor.execute(
                """
                UPDATE ai_data.document_tracking 
                SET status = %s, processed_at = NOW()
                WHERE id = %s
                """,
                (status, doc_id)
            )
        
        conn.commit()
        logger.info(f"Updated document {doc_id} status to {status}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error updating document status: {str(e)}")
    finally:
        cursor.close()
        conn.close()

def delete_previous_document_data(drive_file_id):
    """Delete document and embeddings data for a previous version."""
    if not drive_file_id:
        logger.info("No previous document ID provided, skipping deletion")
        return
    
    conn = get_postgres_connection()
    cursor = conn.cursor()
    
    try:
        # First, get the document ID from the documents table
        cursor.execute(
            """
            SELECT id FROM ai_data.documents 
            WHERE drive_file_id = %s
            """,
            (drive_file_id,)
        )
        document_ids = cursor.fetchall()
        
        if not document_ids:
            logger.info(f"No documents found with drive_file_id: {drive_file_id}")
            return
        
        # Delete embeddings for each document
        for doc_id in document_ids:
            cursor.execute(
                """
                DELETE FROM ai_data.document_embeddings 
                WHERE document_id = %s OR drive_file_id = %s
                """,
                (doc_id[0], drive_file_id)
            )
            logger.info(f"Deleted embeddings for document ID {doc_id[0]}")
        
        # Delete the document records
        cursor.execute(
            """
            DELETE FROM ai_data.documents 
            WHERE drive_file_id = %s
            """,
            (drive_file_id,)
        )
        logger.info(f"Deleted {cursor.rowcount} documents with drive_file_id: {drive_file_id}")
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error deleting previous document data: {str(e)}")
    finally:
        cursor.close()
        conn.close()

def extract_with_layout_analysis(page):
    """Extract text preserving layout structure using text blocks."""
    try:
        # Get text with layout information
        blocks = page.get_text("dict")
        
        page_text = ""
        for block in blocks.get("blocks", []):
            if "lines" in block:
                block_text = ""
                for line in block["lines"]:
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                    if line_text.strip():
                        block_text += line_text.strip() + "\n"
                
                if block_text.strip():
                    page_text += block_text + "\n"
        
        return page_text
    except Exception as e:
        logger.warning(f"Layout analysis failed: {str(e)}")
        return ""

def extract_with_coordinates(page):
    """Extract text using coordinate-based analysis for tables."""
    try:
        # Get all text with coordinates
        text_dict = page.get_text("rawdict")
        
        # Group text by approximate Y coordinate (rows)
        rows = {}
        
        for block in text_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    y_coord = int(line["bbox"][1])  # Y coordinate
                    if y_coord not in rows:
                        rows[y_coord] = []
                    
                    for span in line.get("spans", []):
                        x_coord = int(span["bbox"][0])  # X coordinate
                        text = span.get("text", "").strip()
                        if text:
                            rows[y_coord].append((x_coord, text))
        
        # Sort rows by Y coordinate and build text
        page_text = ""
        for y in sorted(rows.keys()):
            # Sort by X coordinate within each row
            row_items = sorted(rows[y], key=lambda x: x[0])
            row_text = " | ".join([item[1] for item in row_items])
            if row_text.strip():
                page_text += row_text + "\n"
        
        return page_text
    except Exception as e:
        logger.warning(f"Coordinate extraction failed: {str(e)}")
        return ""

def extract_with_advanced_ocr(page, file_content):
    """Advanced OCR with preprocessing for financial documents."""
    try:
        # Convert page to high-resolution image
        pix = page.get_pixmap(matrix=fitz.Matrix(4.0, 4.0))  # 4x resolution
        img_data = pix.tobytes("png")
        
        # Open with PIL for preprocessing
        image = Image.open(io.BytesIO(img_data))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Enhance contrast for better OCR
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # OCR with multiple configurations
        configs = [
            '--oem 3 --psm 6',  # Uniform block of text
            '--oem 3 --psm 4',  # Single column of text
            '--oem 3 --psm 8',  # Single word
            '--oem 3 --psm 11', # Sparse text
        ]
        
        best_text = ""
        max_length = 0
        
        for config in configs:
            try:
                ocr_text = pytesseract.image_to_string(image, lang='eng', config=config)
                if len(ocr_text) > max_length:
                    max_length = len(ocr_text)
                    best_text = ocr_text
            except Exception as ocr_error:
                logger.warning(f"OCR config {config} failed: {str(ocr_error)}")
                continue
        
        return best_text
    except Exception as e:
        logger.warning(f"Advanced OCR failed: {str(e)}")
        return ""

def clean_and_structure_financial_text(text):
    """Advanced cleaning and structuring for financial text."""
    
    # Step 1: Fix spacing issues common in financial PDFs
    text = re.sub(r'([a-zA-Z])([A-Z][a-zA-Z])', r'\1 \2', text)  # Add spaces between camelCase
    text = re.sub(r'(\d)\s*%', r'\1%', text)  # Fix percentage formatting
    text = re.sub(r'\$\s*(\d)', r'$\1', text)  # Fix currency formatting
    text = re.sub(r'\(\s*(\d+[,.]?\d*)\s*\)', r'(\1)', text)  # Fix negative numbers
    
    # Step 2: Fix common financial statement patterns
    text = re.sub(r'(\w+)(\([\d,.]+\))', r'\1 \2', text)  # Space before parenthetical numbers
    text = re.sub(r'(\d+[,.]\d+)(\w)', r'\1 \2', text)  # Space after numbers before words
    
    # Step 3: Preserve important financial terms
    financial_terms = [
        'Total Assets', 'Total Liabilities', 'Total Equity', 'Net Income', 'Net Loss',
        'Gross Profit', 'Operating Income', 'Operating Loss', 'Current Assets',
        'Non-Current Assets', 'Current Liabilities', 'Long Term Loans',
        'Retained Earnings', 'Total Sales', 'Cost of Sales', 'Income Statement',
        'Balance Sheet', 'Cash Flow', 'Profit Loss'
    ]
    
    for term in financial_terms:
        # Ensure proper spacing around financial terms
        pattern = re.escape(term).replace(r'\ ', r'\s*')
        text = re.sub(f'({pattern})', term, text, flags=re.IGNORECASE)
    
    # Step 4: Clean excessive whitespace but preserve structure
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r' {3,}', '  ', text)  # Max 2 consecutive spaces
    text = re.sub(r'\t+', ' ', text)  # Replace tabs with spaces
    
    # Step 5: Add structure markers for better chunking
    text = re.sub(r'\n(Balance Sheet|Income Statement|Statement of|Cash Flow)', r'\n\n=== \1 ===\n', text, flags=re.IGNORECASE)
    
    return text.strip()

def extract_text_comprehensive(file_content):
    """Comprehensive text extraction using multiple methods."""
    try:
        # Convert BytesIO to bytes for PyMuPDF
        if hasattr(file_content, 'getvalue'):
            pdf_bytes = file_content.getvalue()
        else:
            file_content.seek(0)
            pdf_bytes = file_content.read()
        
        # Open PDF with PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        all_text = ""
        extraction_stats = {
            'direct': 0,
            'layout': 0,
            'coordinate': 0,
            'table': 0,
            'ocr': 0
        }
        
        logger.info(f"Processing PDF with {len(doc)} pages using comprehensive extraction")
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            logger.info(f"Processing page {page_num + 1}")
            
            page_texts = {}
            
            # Method 1: Direct text extraction
            try:
                direct_text = page.get_text()
                page_texts['direct'] = direct_text
                extraction_stats['direct'] += len(direct_text)
                logger.info(f"Page {page_num + 1} - Direct: {len(direct_text)} chars")
            except Exception as e:
                logger.warning(f"Direct extraction failed: {str(e)}")
                page_texts['direct'] = ""
            
            # Method 2: Layout-aware extraction
            try:
                layout_text = extract_with_layout_analysis(page)
                page_texts['layout'] = layout_text
                extraction_stats['layout'] += len(layout_text)
                logger.info(f"Page {page_num + 1} - Layout: {len(layout_text)} chars")
            except Exception as e:
                logger.warning(f"Layout extraction failed: {str(e)}")
                page_texts['layout'] = ""
            
            # Method 3: Coordinate-based extraction (good for tables)
            try:
                coord_text = extract_with_coordinates(page)
                page_texts['coordinate'] = coord_text
                extraction_stats['coordinate'] += len(coord_text)
                logger.info(f"Page {page_num + 1} - Coordinate: {len(coord_text)} chars")
            except Exception as e:
                logger.warning(f"Coordinate extraction failed: {str(e)}")
                page_texts['coordinate'] = ""
            
            # Method 4: Table extraction
            try:
                tables = page.find_tables()
                table_text = ""
                for table in tables:
                    df = table.extract()
                    for row in df:
                        row_text = " | ".join([str(cell) if cell else "" for cell in row])
                        table_text += row_text + "\n"
                page_texts['table'] = table_text
                extraction_stats['table'] += len(table_text)
                logger.info(f"Page {page_num + 1} - Table: {len(table_text)} chars")
            except Exception as e:
                logger.warning(f"Table extraction failed: {str(e)}")
                page_texts['table'] = ""
            
            # Method 5: OCR as backup
            try:
                ocr_text = extract_with_advanced_ocr(page, file_content)
                page_texts['ocr'] = ocr_text
                extraction_stats['ocr'] += len(ocr_text)
                logger.info(f"Page {page_num + 1} - OCR: {len(ocr_text)} chars")
            except Exception as e:
                logger.warning(f"OCR extraction failed: {str(e)}")
                page_texts['ocr'] = ""
            
            # Choose the best extraction method for this page
            best_method = 'direct'
            best_length = len(page_texts['direct'])
            
            for method, text in page_texts.items():
                if len(text) > best_length:
                    best_length = len(text)
                    best_method = method
            
            # Combine methods if beneficial
            if best_length < 200:  # If best method is still poor, try combining
                combined_text = ""
                for method in ['table', 'coordinate', 'layout', 'direct', 'ocr']:
                    if page_texts[method] and len(page_texts[method]) > 50:
                        combined_text += f"\n--- {method.upper()} EXTRACTION ---\n"
                        combined_text += page_texts[method] + "\n"
                
                if len(combined_text) > best_length:
                    best_method = 'combined'
                    page_texts['combined'] = combined_text
                    best_length = len(combined_text)
            
            chosen_text = page_texts.get(best_method, page_texts['direct'])
            
            logger.info(f"Page {page_num + 1}: Using {best_method} extraction ({len(chosen_text)} chars)")
            
            all_text += f"\n=== PAGE {page_num + 1} ({best_method.upper()}) ===\n"
            all_text += chosen_text + "\n"
        
        doc.close()
        
        logger.info(f"Extraction stats: {extraction_stats}")
        
        # Clean and structure the text
        cleaned_text = clean_and_structure_financial_text(all_text)
        logger.info(f"Final text length: {len(cleaned_text)} characters")
        
        return cleaned_text
        
    except Exception as e:
        logger.error(f"Comprehensive extraction failed: {str(e)}")
        # Ultimate fallback
        return extract_text_from_pdf_fallback(file_content)

def extract_text_from_pdf_fallback(file_content):
    """Ultimate fallback using pypdf."""
    try:
        from pypdf import PdfReader
        file_content.seek(0)
        reader = PdfReader(file_content)
        text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n=== PAGE {i + 1} (FALLBACK) ===\n"
                text += page_text + "\n"
        logger.info(f"Fallback extraction: {len(text)} characters")
        return text
    except Exception as e:
        logger.error(f"Even fallback extraction failed: {str(e)}")
        return ""

def extract_text_from_pdf(file_content):
    """Main PDF text extraction function."""
    return extract_text_comprehensive(file_content)

def intelligent_financial_chunking(text):
    """Advanced chunking for financial documents with context preservation."""
    try:
        chunk_size = 3000  # Larger chunks for financial context
        chunk_overlap = 500  # Substantial overlap
        
        # Split by major sections first
        sections = re.split(r'\n=== (PAGE \d+|Balance Sheet|Income Statement|Statement of|Cash Flow)', text)
        
        chunks = []
        current_chunk = ""
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            
            # If this is a section header, include it
            if i > 0 and re.match(r'(PAGE \d+|Balance Sheet|Income Statement|Statement of|Cash Flow)', section):
                section = f"\n=== {section} ===\n"
            
            # Check if adding this section exceeds chunk size
            if len(current_chunk) + len(section) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Create meaningful overlap
                    if chunk_overlap > 0:
                        lines = current_chunk.split('\n')
                        overlap_lines = []
                        char_count = 0
                        
                        for line in reversed(lines):
                            if char_count + len(line) <= chunk_overlap:
                                overlap_lines.insert(0, line)
                                char_count += len(line) + 1
                            else:
                                break
                        
                        current_chunk = '\n'.join(overlap_lines) if overlap_lines else ""
                    else:
                        current_chunk = ""
                
                # Handle oversized sections
                if len(section) > chunk_size:
                    # Split by paragraphs or lines
                    lines = section.split('\n')
                    temp_chunk = current_chunk
                    
                    for line in lines:
                        if len(temp_chunk) + len(line) + 1 <= chunk_size:
                            temp_chunk = temp_chunk + '\n' + line if temp_chunk else line
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = line
                    
                    current_chunk = temp_chunk
                else:
                    current_chunk = current_chunk + section if current_chunk else section
            else:
                current_chunk = current_chunk + section if current_chunk else section
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Filter and validate chunks
        valid_chunks = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) >= 200:  # Minimum meaningful size for financial data
                valid_chunks.append(chunk)
                logger.info(f"Chunk {len(valid_chunks)}: {len(chunk)} characters")
            else:
                logger.info(f"Skipped small chunk {i}: {len(chunk)} characters")
        
        logger.info(f"Created {len(valid_chunks)} financial chunks")
        
        if not valid_chunks:
            logger.error("No valid chunks created - using fallback")
            return simple_chunking_fallback(text)
        
        return valid_chunks
        
    except Exception as e:
        logger.error(f"Financial chunking failed: {str(e)}")
        return simple_chunking_fallback(text)

def simple_chunking_fallback(text):
    """Simple fallback chunking with larger sizes for financial docs."""
    chunk_size = 3000
    chunk_overlap = 500
    
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip() and len(chunk.strip()) >= 200:
            chunks.append(chunk)
    
    logger.info(f"Fallback chunking: {len(chunks)} chunks")
    return chunks

def split_text_into_chunks(text):
    """Main chunking function."""
    return intelligent_financial_chunking(text)

def process_document(doc_info):
    """Process a document from Google Drive with comprehensive extraction."""
    try:
        logger.info(f"Processing document: {doc_info['file_name']} (ID: {doc_info['id']})")
        
        # Load model if not already loaded
        global model
        if model is None:
            model = load_model()
        
        # Get Drive service
        drive_service = get_drive_service()
        
        # Check if this is a replacement document
        if doc_info["previous_drive_file_id"]:
            logger.info(f"This is a replacement document for {doc_info['previous_drive_file_id']}")
            delete_previous_document_data(doc_info["previous_drive_file_id"])
        
        # Download file from Google Drive
        file_content = download_file_from_drive(drive_service, doc_info["drive_file_id"])
        
        # Extract text using comprehensive method
        text = extract_text_from_pdf(file_content)
        logger.info(f"Extracted {len(text)} characters from {doc_info['file_name']}")
        
        if len(text) < 500:  # Still too low for an 8-page financial document
            logger.error(f"Insufficient text extracted from {doc_info['file_name']}: {len(text)} characters")
            update_document_status(doc_info["id"], "error", f"Only {len(text)} characters extracted - likely extraction failure")
            return False
        
        # Split text into chunks
        chunks = split_text_into_chunks(text)
        logger.info(f"Created {len(chunks)} chunks")
        
        if len(chunks) == 0:
            logger.error(f"No chunks generated from {doc_info['file_name']}")
            update_document_status(doc_info["id"], "error", "No chunks could be generated")
            return False
        
        # Log detailed chunk analysis
        total_chunk_chars = sum(len(chunk) for chunk in chunks)
        avg_chunk_size = total_chunk_chars // len(chunks)
        logger.info(f"Chunk analysis: {len(chunks)} chunks, {total_chunk_chars} total chars, {avg_chunk_size} avg size")
        
        # Validate chunk quality
        financial_chunks = 0
        for chunk in chunks:
            has_numbers = bool(re.search(r'\d+[,.]\d+', chunk))
            has_financial_terms = bool(re.search(r'(income|loss|profit|asset|liability|equity|revenue|expense)', chunk, re.IGNORECASE))
            if has_numbers and has_financial_terms:
                financial_chunks += 1
        
        logger.info(f"Quality check: {financial_chunks}/{len(chunks)} chunks contain financial data")
        
        # Create document record in database
        conn = get_postgres_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO ai_data.documents
                    (client_id, document_name, year, month, day, class, subclass, 
                     drive_file_id, total_chunks, is_new)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        doc_info["client_id"], 
                        doc_info["file_name"],
                        doc_info["year"],
                        doc_info["month"],
                        doc_info["day"],
                        doc_info["class"],
                        doc_info["subclass"],
                        doc_info["drive_file_id"],
                        len(chunks),
                        True
                    )
                )
                document_id = cursor.fetchone()[0]
                
                # Update tracking table
                cursor.execute(
                    """
                    UPDATE ai_data.document_tracking
                    SET documents_id = %s
                    WHERE id = %s
                    """,
                    (document_id, doc_info["id"])
                )
                
                # Process and store embeddings
                for i, chunk_text in enumerate(chunks):
                    # Generate embedding
                    embedding = model.encode(chunk_text)
                    
                    # Store in PostgreSQL
                    cursor.execute(
                        """
                        INSERT INTO ai_data.document_embeddings
                        (client_id, document_name, chunk_index, chunk_text, year, month, day,
                         class, subclass, document_id, drive_file_id, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            doc_info["client_id"],
                            doc_info["file_name"],
                            i,
                            chunk_text,
                            doc_info["year"],
                            doc_info["month"],
                            doc_info["day"],
                            doc_info["class"],
                            doc_info["subclass"],
                            document_id,
                            doc_info["drive_file_id"],
                            embedding.tolist()
                        )
                    )
                
                conn.commit()
                logger.info(f"Successfully processed {doc_info['file_name']}: {len(chunks)} chunks, {total_chunk_chars} chars, {financial_chunks} financial chunks")
                
                update_document_status(doc_info["id"], "processed")
                return True
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error processing document: {str(e)}")
            update_document_status(doc_info["id"], "error", str(e))
            return False
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        update_document_status(doc_info["id"], "error", str(e))
        return False

def process_pending_documents():
    """Process all pending documents."""
    pending_docs = get_pending_documents()
    logger.info(f"Found {len(pending_docs)} pending documents")
    
    documents_processed = 0
    errors = 0
    
    for doc in pending_docs:
        success = process_document(doc)
        if success:
            documents_processed += 1
        else:
            errors += 1
    
    return {
        "documents_found": len(pending_docs),
        "documents_processed": documents_processed,
        "errors": errors
    }

class DocumentProcessorHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests - for health checks."""
        logger.info(f"Received GET request to {self.path}")
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        response = {
            "status": "healthy",
            "service": "advanced-financial-document-processor",
            "database_connected": True,
            "model_loaded": model is not None
        }
        
        self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def do_POST(self):
        """Handle POST requests - for document processing."""
        logger.info(f"Received POST request to {self.path}")
        
        try:
            results = process_pending_documents()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                "success": True,
                "results": results
            }
            
            self.wfile.write(json.dumps(response).encode('utf-8'))
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                "success": False,
                "error": str(e)
            }
            
            self.wfile.write(json.dumps(response).encode('utf-8'))

def run_server():
    """Run the HTTP server."""
    try:
        logger.info("Starting Advanced Financial Document Processor")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"Environment: PORT={PORT}, PG_HOST={PG_HOST}, PG_DATABASE={PG_DATABASE}")
        
        httpd = socketserver.TCPServer(("", PORT), DocumentProcessorHandler)
        logger.info(f"Advanced processor listening on port {PORT}")
        httpd.serve_forever()
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_server()
