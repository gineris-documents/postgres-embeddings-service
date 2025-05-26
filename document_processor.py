import os
import json
import time
import logging
import io
import re
import numpy as np
import http.server
import socketserver
from google.cloud import storage
from sentence_transformers import SentenceTransformer
import psycopg2
import psycopg2.extras
import google.auth
from google.auth import default
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ID = os.environ.get("PROJECT_ID", "pdf-to-pinecone")
BUCKET_NAME = f"{PROJECT_ID}-chunks"

# PostgreSQL Configuration for Google Cloud SQL
PG_HOST = os.environ.get("PG_HOST", "34.66.180.234")
PG_DATABASE = os.environ.get("PG_DATABASE", "gineris_dev")
PG_USER = os.environ.get("PG_USER", "admin")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "H@nnib@lMO2015")

# Google Drive API configuration
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = os.environ.get("SERVICE_ACCOUNT_FILE", "service-account.json")

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

def init_clients():
    """Initialize GCS, DB, and model clients."""
    # Initialize GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    logger.info(f"Successfully connected to GCS bucket: {BUCKET_NAME}")
    
    # Initialize Drive API
    drive_service = get_drive_service()
    
    # Initialize embedding model
    logger.info("Initializing SentenceTransformer model: all-MiniLM-L6-v2")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Model initialized successfully")
    
    return bucket, drive_service, model

def get_postgres_connection():
    """Create and return a connection to PostgreSQL."""
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            database=PG_DATABASE,
            user=PG_USER,
            password=PG_PASSWORD
        )
        logger.info(f"Successfully connected to PostgreSQL database: {PG_DATABASE}")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
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

def extract_text_with_pymupdf(file_content):
    """Extract text using PyMuPDF with OCR fallback."""
    try:
        # Open PDF with PyMuPDF
        pdf_document = fitz.open("pdf", file_content.read())
        full_text = ""
        page_texts = []
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            
            # Try direct text extraction first
            page_text = page.get_text()
            
            # If no text found, try OCR
            if not page_text.strip():
                logger.info(f"Page {page_num + 1} has no direct text, attempting OCR")
                try:
                    # Convert page to image
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))
                    
                    # Perform OCR
                    page_text = pytesseract.image_to_string(image)
                    logger.info(f"OCR extracted {len(page_text)} characters from page {page_num + 1}")
                except Exception as ocr_error:
                    logger.warning(f"OCR failed for page {page_num + 1}: {str(ocr_error)}")
                    page_text = ""
            
            if page_text.strip():
                # Clean up text spacing and formatting
                page_text = clean_extracted_text(page_text)
                page_texts.append(page_text)
                full_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
        pdf_document.close()
        logger.info(f"PyMuPDF extracted {len(full_text)} characters from {len(page_texts)} pages")
        return full_text, page_texts
        
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed: {str(e)}")
        # Fallback to basic extraction
        return extract_text_fallback(file_content)

def clean_extracted_text(text):
    """Clean up extracted text to preserve financial data formatting."""
    if not text:
        return ""
    
    # Remove excessive whitespace but preserve line breaks
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double newline
    
    # Preserve currency formatting
    text = re.sub(r'(\d+),\s*(\d{3})', r'\1,\2', text)  # Fix broken thousands separators
    text = re.sub(r'\$\s+(\d)', r'$\1', text)  # Fix broken dollar signs
    
    return text.strip()

def extract_text_fallback(file_content):
    """Fallback text extraction method."""
    try:
        from pypdf import PdfReader
        file_content.seek(0)
        reader = PdfReader(file_content)
        text = ""
        page_texts = []
        
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                page_text = clean_extracted_text(page_text)
                page_texts.append(page_text)
                text += page_text + "\n"
        
        logger.info(f"Fallback extraction got {len(text)} characters")
        return text, page_texts
    except Exception as e:
        logger.error(f"Fallback extraction also failed: {str(e)}")
        return "", []

def identify_document_type(text, filename):
    """Identify the type of financial document."""
    text_lower = text.lower()
    filename_lower = filename.lower()
    
    # Tax return patterns
    if any(pattern in text_lower for pattern in ['form 1120', '1120-s', 'tax return', 'form 1065', 'schedule k-1']):
        return 'tax_return'
    
    # Financial statement patterns
    if any(pattern in text_lower for pattern in ['balance sheet', 'income statement', 'profit and loss', 'p&l', 'cash flow statement']):
        return 'financial_statement'
    
    # Invoice patterns
    if any(pattern in text_lower for pattern in ['invoice', 'bill to', 'invoice number', 'amount due']):
        return 'invoice'
    
    # Contract patterns
    if any(pattern in text_lower for pattern in ['agreement', 'contract', 'terms and conditions', 'whereas']):
        return 'contract'
    
    # Default based on filename or content
    if 'tax' in filename_lower:
        return 'tax_return'
    elif any(pattern in filename_lower for pattern in ['financial', 'statement', 'p&l', 'income']):
        return 'financial_statement'
    
    return 'unknown'

def extract_financial_summary(text, document_type):
    """Extract structured financial data based on document type."""
    summary_data = {
        'document_type': document_type,
        'extracted_metrics': {}
    }
    
    if document_type == 'tax_return':
        summary_data['extracted_metrics'] = extract_tax_metrics(text)
    elif document_type == 'financial_statement':
        summary_data['extracted_metrics'] = extract_financial_metrics(text)
    
    # Convert to JSON string for storage
    return json.dumps(summary_data, indent=2)

def extract_tax_metrics(text):
    """Extract key tax metrics from tax documents."""
    metrics = {}
    
    # Common tax patterns
    patterns = {
        'ordinary_business_income': [
            r'ordinary business income.*?[\$\(]?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'line 22.*?[\$\(]?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        ],
        'total_income': [
            r'total income.*?[\$\(]?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'gross income.*?[\$\(]?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        ],
        'net_income': [
            r'net income.*?[\$\(]?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'net profit.*?[\$\(]?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        ],
        'tax_owed': [
            r'tax owed.*?[\$\(]?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'amount due.*?[\$\(]?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        ]
    }
    
    for metric, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    # Take the last match (usually the total)
                    value = matches[-1].replace(',', '')
                    metrics[metric] = float(value)
                    break
                except (ValueError, IndexError):
                    continue
    
    return metrics

def extract_financial_metrics(text):
    """Extract key financial metrics from financial statements."""
    metrics = {}
    
    patterns = {
        'revenue': [
            r'revenue.*?[\$\(]?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'sales.*?[\$\(]?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        ],
        'net_income': [
            r'net income.*?[\$\(]?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'net profit.*?[\$\(]?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        ],
        'total_assets': [
            r'total assets.*?[\$\(]?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        ],
        'total_liabilities': [
            r'total liabilities.*?[\$\(]?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        ]
    }
    
    for metric, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    value = matches[-1].replace(',', '')
                    metrics[metric] = float(value)
                    break
                except (ValueError, IndexError):
                    continue
    
    return metrics

def split_text_into_chunks(text):
    """Split text into chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

def process_document(doc_info, bucket, drive_service, model):
    """Process a document from Google Drive with enhanced text extraction."""
    try:
        # Check if this is a replacement document
        if doc_info["previous_drive_file_id"]:
            logger.info(f"This is a replacement document for {doc_info['previous_drive_file_id']}")
            delete_previous_document_data(doc_info["previous_drive_file_id"])
        
        # Download file from Google Drive
        file_content = download_file_from_drive(drive_service, doc_info["drive_file_id"])
        
        # Extract text using enhanced method
        full_document_text, page_texts = extract_text_with_pymupdf(file_content)
        logger.info(f"Extracted {len(full_document_text)} characters of text from {doc_info['file_name']}")
        
        if len(full_document_text) == 0:
            logger.error(f"No text extracted from {doc_info['file_name']}")
            update_document_status(doc_info["id"], "error", "No text could be extracted from the PDF")
            return False
        
        # Identify document type
        document_type = identify_document_type(full_document_text, doc_info["file_name"])
        logger.info(f"Identified document type: {document_type}")
        
        # Extract financial summary
        financial_summary = extract_financial_summary(full_document_text, document_type)
        logger.info(f"Generated financial summary with {len(financial_summary)} characters")
        
        # Split text into chunks
        chunks = split_text_into_chunks(full_document_text)
        logger.info(f"Split text into {len(chunks)} chunks")
        
        if len(chunks) == 0:
            logger.error(f"No chunks generated from {doc_info['file_name']}")
            update_document_status(doc_info["id"], "error", "No chunks could be generated from the PDF text")
            return False
        
        # Create document record in database
        conn = get_postgres_connection()
        try:
            with conn.cursor() as cursor:
                # Insert document with full text and financial summary
                cursor.execute(
                    """
                    INSERT INTO ai_data.documents
                    (client_id, document_name, year, month, day, class, subclass, 
                     drive_file_id, total_chunks, is_new, full_document_text, financial_summary)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                        True,  # Assuming all new documents are marked as new
                        full_document_text,  # Store full document text
                        financial_summary    # Store financial summary
                    )
                )
                document_id = cursor.fetchone()[0]
                logger.info(f"Created document record with ID: {document_id}")
                
                # Update the document_tracking table with document_id
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
                            embedding.tolist()  # Convert numpy array to list for PostgreSQL
                        )
                    )
                
                conn.commit()
                logger.info(f"Successfully processed document {doc_info['file_name']} with {len(chunks)} chunks, full text ({len(full_document_text)} chars), and financial summary")
                
                # Update document status to 'processed'
                update_document_status(doc_info["id"], "processed")
                return True
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Error processing document: {str(e)}")
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
    bucket, drive_service, model = init_clients()
    
    # Get pending documents
    pending_docs = get_pending_documents()
    logger.info(f"Found {len(pending_docs)} pending documents")
    
    documents_processed = 0
    errors = 0
    
    # Process each document
    for doc in pending_docs:
        logger.info(f"Processing document: {doc['file_name']} (ID: {doc['id']})")
        success = process_document(doc, bucket, drive_service, model)
        if success:
            documents_processed += 1
        else:
            errors += 1
    
    return {
        "documents_found": len(pending_docs),
        "documents_processed": documents_processed,
        "errors": errors
    }

class SimpleHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests - for health checks."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {'status': 'healthy'}
        self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def do_POST(self):
        """Handle POST requests - for document processing."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Process documents
            results = process_pending_documents()
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'success': True, 'results': results}
            self.wfile.write(json.dumps(response).encode('utf-8'))
        except Exception as e:
            logger.error(f"Error in POST handler: {str(e)}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'success': False, 'error': str(e)}
            self.wfile.write(json.dumps(response).encode('utf-8'))

def run_server():
    """Run the HTTP server."""
    port = int(os.environ.get('PORT', 8080))
    server = socketserver.TCPServer(('0.0.0.0', port), SimpleHTTPRequestHandler)
    logger.info(f'Starting server on port {port}')
    server.serve_forever()

if __name__ == "__main__":
    run_server()
