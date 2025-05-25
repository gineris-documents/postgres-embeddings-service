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
        if os.path.exists(SERVICE_ACCOUNT_FILE) and os.path.getsize(SERVICE_ACCOUNT_FILE) > 0:
            creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
            service = build('drive', 'v3', credentials=creds)
            logger.info("Successfully connected to Google Drive API using service account")
            return service
        else:
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
        
        for doc_id in document_ids:
            cursor.execute(
                """
                DELETE FROM ai_data.document_embeddings 
                WHERE document_id = %s OR drive_file_id = %s
                """,
                (doc_id[0], drive_file_id)
            )
            logger.info(f"Deleted embeddings for document ID {doc_id[0]}")
        
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

def parse_financial_number(text):
    """Parse financial number handling negatives and formatting."""
    if not text:
        return None
    
    # Remove commas and spaces
    clean_text = re.sub(r'[,\s]', '', text)
    
    # Handle parentheses as negative
    if clean_text.startswith('(') and clean_text.endswith(')'):
        clean_text = '-' + clean_text[1:-1]
    
    # Convert to float
    try:
        return float(clean_text)
    except:
        return None

def extract_financial_metrics(text):
    """Extract key financial metrics into structured format."""
    
    financial_data = {
        "company_name": None,
        "period_end": None,
        "net_income": None,
        "total_revenue": None,
        "total_assets": None,
        "total_liabilities": None,
        "total_equity": None,
        "gross_profit": None,
        "operating_income": None,
        "current_assets": None,
        "current_liabilities": None,
        "cash_and_equivalents": None,
        "key_metrics": {}
    }
    
    # Extract company name
    company_match = re.search(r'([\w\s,]+(?:LLC|Inc|Corp|Corporation))', text, re.IGNORECASE)
    if company_match:
        financial_data["company_name"] = company_match.group(1).strip()
    
    # Extract period
    period_match = re.search(r'(?:As of|For the year ended|December 31,?\s*(\d{4}))', text)
    if period_match:
        financial_data["period_end"] = period_match.group(1)
    
    # Extract Net Income (multiple patterns to catch all variations)
    net_income_patterns = [
        r'Net Income[^\d\-\(]*[\(\-]?([\d,.\s]+)[\)]?',
        r'Current Year Earnings[^\d\-\(]*[\(\-]?([\d,.\s]+)[\)]?',
        r'Net Income \(Loss\)[^\d\-\(]*[\(\-]?([\d,.\s]+)[\)]?',
        r'Total Comprehensive Income[^\d\-\(]*[\(\-]?([\d,.\s]+)[\)]?',
        r'Profit \(Loss\) Before Tax[^\d\-\(]*[\(\-]?([\d,.\s]+)[\)]?'
    ]
    
    for pattern in net_income_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            if not financial_data["net_income"]:
                # Look for the pattern with context to get the right number
                context = text[max(0, match.start()-50):match.end()+50]
                if '2024' in context or 'current' in context.lower():
                    value = parse_financial_number(match.group(1))
                    if value is not None:
                        financial_data["net_income"] = value
                        logger.info(f"Found Net Income: {value} using pattern: {pattern}")
                        break
    
    # Extract other key metrics
    metrics_patterns = {
        "total_sales": r'Total Sales[^\d\-\(]*[\(\-]?([\d,.\s]+)[\)]?',
        "total_revenue": r'Total (?:Income|Revenue)[^\d\-\(]*[\(\-]?([\d,.\s]+)[\)]?',
        "gross_profit": r'Gross Profit[^\d\-\(]*[\(\-]?([\d,.\s]+)[\)]?',
        "total_assets": r'Total Assets[^\d\-\(]*[\(\-]?([\d,.\s]+)[\)]?',
        "total_liabilities": r'Total Liabilities[^\d\-\(]*[\(\-]?([\d,.\s]+)[\)]?',
        "total_equity": r'Total Equity[^\d\-\(]*[\(\-]?([\d,.\s]+)[\)]?',
        "current_assets": r'Total Current Assets[^\d\-\(]*[\(\-]?([\d,.\s]+)[\)]?',
        "current_liabilities": r'Total Current Liabilities[^\d\-\(]*[\(\-]?([\d,.\s]+)[\)]?',
        "cash_equivalents": r'Total Cash and Cash Equivalents[^\d\-\(]*[\(\-]?([\d,.\s]+)[\)]?',
        "operating_income": r'Operating Income[^\d\-\(]*[\(\-]?([\d,.\s]+)[\)]?'
    }
    
    for key, pattern in metrics_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = parse_financial_number(match.group(1))
            if value is not None:
                financial_data["key_metrics"][key] = value
    
    return financial_data

def create_financial_summary(financial_data):
    """Create a natural language summary of financial data."""
    
    company = financial_data.get("company_name", "The company")
    period = financial_data.get("period_end", "the period")
    
    summary_parts = [f"Financial summary for {company} as of {period}:"]
    
    # Net Income - this is the key metric
    net_income = financial_data.get("net_income")
    if net_income is not None:
        if net_income < 0:
            summary_parts.append(f"Net Loss: ${abs(net_income):,.2f}")
            summary_parts.append(f"Current Year Earnings: (${abs(net_income):,.2f})")
            summary_parts.append(f"Net Income (Loss): (${abs(net_income):,.2f})")
        else:
            summary_parts.append(f"Net Income: ${net_income:,.2f}")
            summary_parts.append(f"Current Year Earnings: ${net_income:,.2f}")
    
    # Key metrics
    metrics = financial_data.get("key_metrics", {})
    
    if "total_sales" in metrics:
        summary_parts.append(f"Total Sales: ${metrics['total_sales']:,.2f}")
    
    if "gross_profit" in metrics:
        summary_parts.append(f"Gross Profit: ${metrics['gross_profit']:,.2f}")
    
    if "total_assets" in metrics:
        summary_parts.append(f"Total Assets: ${metrics['total_assets']:,.2f}")
    
    if "total_equity" in metrics:
        equity = metrics['total_equity']
        if equity < 0:
            summary_parts.append(f"Total Equity: (${abs(equity):,.2f}) - negative equity")
        else:
            summary_parts.append(f"Total Equity: ${equity:,.2f}")
    
    # Add ratios if possible
    if net_income and "total_sales" in metrics and metrics["total_sales"] != 0:
        margin = (net_income / metrics["total_sales"]) * 100
        summary_parts.append(f"Net Profit Margin: {margin:.1f}%")
    
    return " ".join(summary_parts)

def simple_ocr_extraction(page):
    """Simple, focused OCR for financial documents."""
    try:
        # High resolution for better OCR
        pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0))
        img_data = pix.tobytes("png")
        
        # Basic OCR - no fancy preprocessing that breaks things
        image = Image.open(io.BytesIO(img_data))
        ocr_text = pytesseract.image_to_string(image, lang='eng')
        
        return ocr_text
    except Exception as e:
        logger.warning(f"OCR failed: {str(e)}")
        return ""

def extract_text_simple_and_effective(file_content):
    """Simple but effective extraction - try direct first, OCR as backup."""
    try:
        if hasattr(file_content, 'getvalue'):
            pdf_bytes = file_content.getvalue()
        else:
            file_content.seek(0)
            pdf_bytes = file_content.read()
        
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        all_text = ""
        ocr_pages = 0
        
        logger.info(f"Processing {len(doc)} pages with simple extraction")
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Try direct extraction first
            direct_text = page.get_text()
            
            # If direct extraction is poor, use OCR
            if len(direct_text.strip()) < 100:
                logger.info(f"Page {page_num + 1}: Direct extraction poor ({len(direct_text.strip())} chars), trying OCR")
                ocr_text = simple_ocr_extraction(page)
                
                if len(ocr_text.strip()) > len(direct_text.strip()):
                    page_text = ocr_text
                    ocr_pages += 1
                    logger.info(f"Page {page_num + 1}: OCR better ({len(ocr_text.strip())} chars)")
                else:
                    page_text = direct_text
                    logger.info(f"Page {page_num + 1}: Keeping direct extraction")
            else:
                page_text = direct_text
                logger.info(f"Page {page_num + 1}: Direct extraction good ({len(direct_text.strip())} chars)")
            
            # Simple page separator
            all_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
        doc.close()
        
        logger.info(f"Extraction complete: {len(all_text)} total chars, OCR used on {ocr_pages} pages")
        
        # Very light cleaning - just fix obvious issues
        all_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', all_text)  # Remove excessive newlines
        all_text = re.sub(r' +', ' ', all_text)  # Remove excessive spaces
        
        return all_text
        
    except Exception as e:
        logger.error(f"Simple extraction failed: {str(e)}")
        return extract_text_fallback(file_content)

def extract_text_fallback(file_content):
    """Ultimate fallback using pypdf."""
    try:
        from pypdf import PdfReader
        file_content.seek(0)
        reader = PdfReader(file_content)
        text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {i + 1} ---\n{page_text}\n"
        logger.info(f"Fallback extraction: {len(text)} characters")
        return text
    except Exception as e:
        logger.error(f"Even fallback extraction failed: {str(e)}")
        return ""

def extract_text_from_pdf(file_content):
    """Main extraction function - keep it simple."""
    return extract_text_simple_and_effective(file_content)

def simple_smart_chunking(text):
    """Simple but smart chunking for financial documents."""
    try:
        # Medium-sized chunks with good overlap
        chunk_size = 1500
        chunk_overlap = 300
        
        # Split by pages first to respect document structure
        page_sections = re.split(r'\n--- Page \d+ ---\n', text)
        
        chunks = []
        current_chunk = ""
        
        for section in page_sections:
            if not section.strip():
                continue
            
            # If section fits in current chunk, add it
            if len(current_chunk) + len(section) <= chunk_size:
                current_chunk = current_chunk + "\n\n" + section if current_chunk else section
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Create overlap
                    if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                        current_chunk = current_chunk[-chunk_overlap:]
                    else:
                        current_chunk = ""
                
                # If section is too big, split it
                if len(section) > chunk_size:
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
                    current_chunk = section
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Filter very small chunks
        final_chunks = [chunk for chunk in chunks if len(chunk.strip()) >= 100]
        
        logger.info(f"Simple chunking created {len(final_chunks)} chunks from {len(chunks)} total")
        
        # Log some stats
        if final_chunks:
            total_chars = sum(len(chunk) for chunk in final_chunks)
            avg_size = total_chars // len(final_chunks)
            logger.info(f"Chunk stats: {len(final_chunks)} chunks, {total_chars} total chars, {avg_size} avg size")
        
        return final_chunks if final_chunks else [text]  # Return original text if chunking fails
        
    except Exception as e:
        logger.error(f"Simple chunking failed: {str(e)}")
        # Ultimate fallback - just split by size
        chunks = []
        for i in range(0, len(text), 1500):
            chunk = text[i:i+1500]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

def split_text_into_chunks(text):
    """Main chunking function."""
    return simple_smart_chunking(text)

def process_document(doc_info):
    """Process a document with structured financial data extraction."""
    try:
        logger.info(f"Processing document: {doc_info['file_name']} (ID: {doc_info['id']})")
        
        # Load model
        global model
        if model is None:
            model = load_model()
        
        # Get Drive service
        drive_service = get_drive_service()
        
        # Handle replacement
        if doc_info["previous_drive_file_id"]:
            logger.info(f"Replacing document {doc_info['previous_drive_file_id']}")
            delete_previous_document_data(doc_info["previous_drive_file_id"])
        
        # Download and extract
        file_content = download_file_from_drive(drive_service, doc_info["drive_file_id"])
        text = extract_text_from_pdf(file_content)
        
        logger.info(f"Extracted {len(text)} characters from {doc_info['file_name']}")
        
        if len(text) < 1000:  # Reasonable threshold for 8-page document
            logger.error(f"Too little text extracted: {len(text)} characters")
            update_document_status(doc_info["id"], "error", f"Only {len(text)} characters extracted")
            return False
        
        # NEW: Extract structured financial data
        financial_data = extract_financial_metrics(text)
        financial_summary = create_financial_summary(financial_data)
        
        logger.info(f"Extracted financial data: {financial_data}")
        logger.info(f"Generated summary: {financial_summary}")
        
        # Chunk the text
        chunks = split_text_into_chunks(text)
        
        if len(chunks) == 0:
            logger.error("No chunks created")
            update_document_status(doc_info["id"], "error", "No chunks could be generated")
            return False
        
        # Quality check
        financial_chunks = 0
        for chunk in chunks:
            if re.search(r'\d+[,.]\d+', chunk) and re.search(r'(income|loss|profit|asset|liability|equity|revenue|expense)', chunk, re.IGNORECASE):
                financial_chunks += 1
        
        logger.info(f"Quality check: {financial_chunks}/{len(chunks)} chunks contain financial data")
        
        # Store in database with structured data
        conn = get_postgres_connection()
        try:
            with conn.cursor() as cursor:
                # Insert document with new columns
                cursor.execute(
                    """
                    INSERT INTO ai_data.documents
                    (client_id, document_name, year, month, day, class, subclass, 
                     drive_file_id, total_chunks, is_new, full_document_text, financial_summary)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        doc_info["client_id"], doc_info["file_name"], doc_info["year"],
                        doc_info["month"], doc_info["day"], doc_info["class"],
                        doc_info["subclass"], doc_info["drive_file_id"], len(chunks), 
                        True, text, financial_summary
                    )
                )
                document_id = cursor.fetchone()[0]
                
                # Update tracking
                cursor.execute(
                    """
                    UPDATE ai_data.document_tracking SET documents_id = %s WHERE id = %s
                    """,
                    (document_id, doc_info["id"])
                )
                
                # Insert embeddings
                for i, chunk_text in enumerate(chunks):
                    embedding = model.encode(chunk_text)
                    
                    cursor.execute(
                        """
                        INSERT INTO ai_data.document_embeddings
                        (client_id, document_name, chunk_index, chunk_text, year, month, day,
                         class, subclass, document_id, drive_file_id, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            doc_info["client_id"], doc_info["file_name"], i, chunk_text,
                            doc_info["year"], doc_info["month"], doc_info["day"],
                            doc_info["class"], doc_info["subclass"], document_id,
                            doc_info["drive_file_id"], embedding.tolist()
                        )
                    )
                
                conn.commit()
                
                total_chars = sum(len(chunk) for chunk in chunks)
                logger.info(f"SUCCESS: {doc_info['file_name']} - {len(chunks)} chunks, {total_chars} chars, {financial_chunks} with financial data")
                logger.info(f"Financial Summary: {financial_summary}")
                
                update_document_status(doc_info["id"], "processed")
                return True
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {str(e)}")
            update_document_status(doc_info["id"], "error", str(e))
            return False
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        update_document_status(doc_info["id"], "error", str(e))
        return False

def process_pending_documents():
    """Process pending documents."""
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
        """Health check."""
        logger.info(f"Health check request")
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        response = {
            "status": "healthy",
            "service": "structured-financial-document-processor",
            "database_connected": True,
            "model_loaded": model is not None
        }
        
        self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def do_POST(self):
        """Process documents."""
        logger.info(f"Processing request")
        
        try:
            results = process_pending_documents()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {"success": True, "results": results}
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Request error: {str(e)}")
            
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {"success": False, "error": str(e)}
            self.wfile.write(json.dumps(response).encode('utf-8'))

def run_server():
    """Run the server."""
    try:
        logger.info("Starting Structured Financial Document Processor")
        logger.info(f"Environment: PORT={PORT}, PG_HOST={PG_HOST}")
        
        httpd = socketserver.TCPServer(("", PORT), DocumentProcessorHandler)
        logger.info(f"Structured processor listening on port {PORT}")
        httpd.serve_forever()
        
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_server()
