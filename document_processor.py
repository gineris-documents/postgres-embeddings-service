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

def clean_extracted_text(text: str) -> str:
    """Clean up extracted text by removing excessive whitespace and fixing common issues."""
    
    # Remove excessive line breaks (more than 2 consecutive)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Fix hyphenated words split across lines
    text = re.sub(r'-\s*\n\s*', '', text)
    
    # Remove excessive spaces
    text = re.sub(r' +', ' ', text)
    
    # Remove common OCR artifacts
    text = re.sub(r'[|]', 'I', text)  # Common OCR mistake
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # Add space between camelCase
    
    return text.strip()

def extract_text_from_pdf_fallback(file_content):
    """Fallback to original pypdf extraction method."""
    try:
        from pypdf import PdfReader
        file_content.seek(0)  # Reset file pointer
        reader = PdfReader(file_content)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        logger.error(f"Fallback extraction also failed: {str(e)}")
        return ""

def extract_text_from_pdf(file_content):
    """Enhanced PDF text extraction with OCR fallback."""
    try:
        # Convert BytesIO to bytes for PyMuPDF
        if hasattr(file_content, 'getvalue'):
            pdf_bytes = file_content.getvalue()
        else:
            file_content.seek(0)
            pdf_bytes = file_content.read()
        
        # Open PDF with PyMuPDF (much better than pypdf)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        all_text = ""
        pages_with_ocr = 0
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # First, try direct text extraction
            page_text = page.get_text()
            
            # If we get very little text, the page might be scanned
            if len(page_text.strip()) < 50:  # Threshold for "sparse text"
                logger.info(f"Page {page_num + 1}: Little text found ({len(page_text.strip())} chars), trying OCR...")
                
                try:
                    # Convert page to image with higher resolution for better OCR
                    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2x resolution
                    img_data = pix.tobytes("png")
                    
                    # Use OCR on the image
                    image = Image.open(io.BytesIO(img_data))
                    ocr_text = pytesseract.image_to_string(image, lang='eng')
                    
                    if len(ocr_text.strip()) > len(page_text.strip()):
                        page_text = ocr_text
                        pages_with_ocr += 1
                        logger.info(f"Page {page_num + 1}: OCR extracted {len(ocr_text.strip())} characters")
                    else:
                        logger.info(f"Page {page_num + 1}: Direct extraction was better")
                        
                except Exception as ocr_error:
                    logger.warning(f"OCR failed for page {page_num + 1}: {str(ocr_error)}")
            else:
                logger.info(f"Page {page_num + 1}: Direct extraction got {len(page_text.strip())} characters")
            
            all_text += page_text + "\n"
        
        doc.close()
        
        if pages_with_ocr > 0:
            logger.info(f"Used OCR on {pages_with_ocr} out of {len(doc)} pages")
        
        # Clean up the text
        cleaned_text = clean_extracted_text(all_text)
        logger.info(f"Final cleaned text length: {len(cleaned_text)} characters")
        
        return cleaned_text
        
    except Exception as e:
        logger.error(f"Error in enhanced text extraction: {str(e)}")
        # Fallback to original method
        logger.info("Falling back to pypdf extraction...")
        return extract_text_from_pdf_fallback(file_content)

def simple_chunking_fallback(text):
    """Simple fallback chunking method."""
    chunk_size = 1000
    chunk_overlap = 200
    
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def split_text_into_chunks(text):
    """Improved text chunking that respects sentence and paragraph boundaries."""
    try:
        chunk_size = 1000
        chunk_overlap = 200
        
        # First, split by paragraphs (double newlines)
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) + 2 > chunk_size:  # +2 for \n\n
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Handle overlap by taking last sentences of previous chunk
                    if chunk_overlap > 0:
                        sentences = current_chunk.split('. ')
                        overlap_text = ""
                        for sentence in reversed(sentences):
                            if len(overlap_text) + len(sentence) <= chunk_overlap:
                                overlap_text = sentence + ". " + overlap_text if overlap_text else sentence
                            else:
                                break
                        current_chunk = overlap_text.strip()
                    else:
                        current_chunk = ""
                
                # If the paragraph itself is too long, split it
                if len(paragraph) > chunk_size:
                    # Split long paragraphs at sentence boundaries
                    sentences = paragraph.split('. ')
                    temp_chunk = current_chunk
                    
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) + 2 <= chunk_size:
                            temp_chunk = temp_chunk + ". " + sentence if temp_chunk else sentence
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence
                    
                    current_chunk = temp_chunk
                else:
                    current_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            else:
                # Add paragraph to current chunk
                current_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Filter out very small chunks
        final_chunks = [chunk for chunk in chunks if len(chunk.strip()) >= 50]
        
        logger.info(f"Created {len(final_chunks)} meaningful chunks from {len(chunks)} total chunks")
        return final_chunks
        
    except Exception as e:
        logger.error(f"Error in improved chunking: {str(e)}")
        # Fallback to simple chunking
        return simple_chunking_fallback(text)

def process_document(doc_info):
    """Process a document from Google Drive."""
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
        
        # Extract text from PDF using enhanced method
        text = extract_text_from_pdf(file_content)
        logger.info(f"Extracted {len(text)} characters of text from {doc_info['file_name']}")
        
        if len(text) == 0:
            logger.error(f"No text extracted from {doc_info['file_name']}")
            update_document_status(doc_info["id"], "error", "No text could be extracted from the PDF")
            return False
        
        # Split text into chunks using improved method
        chunks = split_text_into_chunks(text)
        logger.info(f"Split text into {len(chunks)} chunks")
        
        if len(chunks) == 0:
            logger.error(f"No chunks generated from {doc_info['file_name']}")
            update_document_status(doc_info["id"], "error", "No chunks could be generated from the PDF text")
            return False
        
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
                        True  # Assuming all new documents are marked as new
                    )
                )
                document_id = cursor.fetchone()[0]
                
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
                logger.info(f"Successfully processed document {doc_info['file_name']} with {len(chunks)} chunks")
                
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
    # Get pending documents (just one at a time to avoid timeouts)
    pending_docs = get_pending_documents()
    logger.info(f"Found {len(pending_docs)} pending documents")
    
    documents_processed = 0
    errors = 0
    
    # Process each document
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
        
        # Don't load the model on health checks to avoid timeouts
        response = {
            "status": "healthy",
            "service": "postgres-embeddings-service",
            "database_connected": True,
            "model_loaded": model is not None
        }
        
        self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def do_POST(self):
        """Handle POST requests - for document processing."""
        logger.info(f"Received POST request to {self.path}")
        
        try:
            # Process pending documents
            results = process_pending_documents()
            
            # Send response
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
        # Print startup message
        logger.info("Starting document processor service")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"Environment variables: PORT={PORT}, PG_HOST={PG_HOST}, PG_DATABASE={PG_DATABASE}")
        
        # Start the server
        httpd = socketserver.TCPServer(("", PORT), DocumentProcessorHandler)
        logger.info(f"Server listening on port {PORT}")
        httpd.serve_forever()
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_server()
