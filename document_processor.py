import os
import json
import time
import io
import logging
import base64
import http.server
import socketserver
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Essential imports with error handling
try:
    import psycopg2
    import psycopg2.extras
    logger.info("PostgreSQL libraries imported successfully")
except ImportError as e:
    logger.error(f"Failed to import PostgreSQL libraries: {e}")

try:
    import google.auth
    from google.auth import default
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from googleapiclient.errors import HttpError
    logger.info("Google API libraries imported successfully")
except ImportError as e:
    logger.error(f"Failed to import Google API libraries: {e}")

try:
    import fitz  # PyMuPDF for PDF text extraction
    PYMUPDF_AVAILABLE = True
    logger.info("PyMuPDF imported successfully")
except ImportError as e:
    PYMUPDF_AVAILABLE = False
    logger.warning(f"PyMuPDF not available: {e}")

try:
    from PIL import Image
    PIL_AVAILABLE = True
    logger.info("PIL imported successfully")
except ImportError as e:
    PIL_AVAILABLE = False
    logger.warning(f"PIL not available: {e}")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    logger.info("Tesseract imported successfully")
except ImportError as e:
    TESSERACT_AVAILABLE = False
    logger.warning(f"Tesseract not available: {e}")

try:
    import requests
    REQUESTS_AVAILABLE = True
    logger.info("Requests library imported successfully")
except ImportError as e:
    REQUESTS_AVAILABLE = False
    logger.error(f"Requests library not available: {e}")

# Configuration
PROJECT_ID = os.environ.get("PROJECT_ID", "pdf-to-pinecone")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# PostgreSQL Configuration
PG_HOST = os.environ.get("PG_HOST", "34.66.180.234")
PG_DATABASE = os.environ.get("PG_DATABASE", "postgres")
PG_USER = os.environ.get("PG_USER", "admin")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "H@nnib@lMO2015")

# Google Drive API configuration
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = os.environ.get("SERVICE_ACCOUNT_FILE", "service-account.json")

def get_drive_service():
    """Create and return a Google Drive service."""
    try:
        if os.path.exists(SERVICE_ACCOUNT_FILE) and os.path.getsize(SERVICE_ACCOUNT_FILE) > 0:
            creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
            service = build('drive', 'v3', credentials=creds)
            logger.info("Successfully connected to Google Drive API using service account")
            return service
        else:
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
    except Exception as e:
        logger.error(f"Error downloading file from Google Drive: {str(e)}")
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
        logger.info(f"Successfully connected to PostgreSQL database: {PG_DATABASE}")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
        raise

def check_database_schema():
    """Check if the required database schema exists."""
    conn = get_postgres_connection()
    cursor = conn.cursor()
    
    try:
        # Check if ai_data schema exists
        cursor.execute("""
            SELECT schema_name FROM information_schema.schemata 
            WHERE schema_name = 'ai_data'
        """)
        schema_exists = cursor.fetchone() is not None
        
        if not schema_exists:
            logger.warning("ai_data schema does not exist. Creating it...")
            cursor.execute("CREATE SCHEMA IF NOT EXISTS ai_data")
            conn.commit()
            logger.info("✓ Created ai_data schema")
        
        # Check if documents table exists in ai_data schema
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'ai_data' AND table_name = 'documents'
        """)
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            logger.error("❌ ai_data.documents table does not exist")
            return False
        
        # Check if vision_analysis column exists
        cursor.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_schema = 'ai_data' AND table_name = 'documents' 
            AND column_name = 'vision_analysis'
        """)
        vision_column_exists = cursor.fetchone() is not None
        
        if not vision_column_exists:
            logger.warning("vision_analysis column missing. Adding it...")
            cursor.execute("""
                ALTER TABLE ai_data.documents 
                ADD COLUMN IF NOT EXISTS vision_analysis TEXT
            """)
            conn.commit()
            logger.info("✓ Added vision_analysis column")
        
        logger.info("✓ Database schema check complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database schema check failed: {str(e)}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

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

def extract_pdf_text_content(file_content: io.BytesIO) -> str:
    """Extract text content from PDF using PyMuPDF (same as working Google Drive tool)"""
    if not PYMUPDF_AVAILABLE:
        raise Exception("PyMuPDF not available for PDF text extraction")
    
    try:
        file_content.seek(0)
        pdf_data = file_content.read()
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        
        all_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            
            # If direct extraction fails, try OCR
            if len(page_text.strip()) < 50 and PIL_AVAILABLE and TESSERACT_AVAILABLE:
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))
                    page_text = pytesseract.image_to_string(image, lang="eng")
                    logger.info(f"Used OCR for page {page_num + 1}")
                except Exception as ocr_error:
                    logger.warning(f"OCR failed for page {page_num + 1}: {str(ocr_error)}")
            
            all_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            logger.info(f"Extracted text from page {page_num + 1}: {len(page_text)} characters")
        
        doc.close()
        
        # Clean the text
        clean_text = clean_extracted_text(all_text)
        logger.info(f"Successfully extracted and cleaned text: {len(clean_text)} characters total")
        return clean_text
        
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
        raise

def clean_extracted_text(text: str) -> str:
    """Clean extracted text (same as working Google Drive tool)"""
    import re
    # Remove excessive whitespace
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    text = re.sub(r" +", " ", text)
    return text.strip()

def identify_document_type_from_filename(filename: str) -> str:
    """Identify document type from filename."""
    filename_lower = filename.lower()
    
    if any(term in filename_lower for term in ['1120', '1040', '1065', 'tax', 'return']):
        return 'tax_return'
    elif any(term in filename_lower for term in ['financial', 'statement', 'p&l', 'balance']):
        return 'financial_statement'
    elif any(term in filename_lower for term in ['invoice', 'bill']):
        return 'invoice'
    elif any(term in filename_lower for term in ['contract', 'agreement']):
        return 'contract'
    else:
        return 'unknown'

def get_financial_analysis_prompt(document_type: str, text_content: str) -> str:
    """Get the appropriate prompt for financial analysis based on document type."""
    
    base_prompt = f"""You are an expert financial analyst. Analyze this {document_type} document and extract key financial information in JSON format.

Document content:
{text_content}

Please provide a JSON response with the following structure:
{{
    "document_type": "{document_type}",
    "company_info": {{
        "company_name": "extracted company name",
        "period_covered": "time period of the document",
        "prepared_by": "who prepared this document"
    }},
    "key_figures": {{
        "revenue": "total revenue/sales amount",
        "expenses": "total expenses",
        "net_income": "net income or profit/loss",
        "total_assets": "total assets if available",
        "total_liabilities": "total liabilities if available",
        "equity": "total equity if available",
        "cash": "cash position if available"
    }},
    "summary": "comprehensive summary of the financial position and performance",
    "notable_items": ["list", "of", "important", "observations", "and", "key", "findings"],
    "year_over_year_changes": "any comparisons to prior periods if shown",
    "financial_ratios": "any ratios that can be calculated from the data"
}}

"""
    
    if document_type == 'tax_return':
        return base_prompt + """
Focus specifically on:
- Form type (1120, 1120S, 1040, etc.)
- Tax year
- Taxpayer name and EIN/SSN
- Adjusted Gross Income, Taxable Income, Tax owed/refund
- Schedule information and deductions
- Any unusual items or red flags
"""
    
    elif document_type == 'financial_statement':
        return base_prompt + """
Focus specifically on:
- Statement type (Income Statement, Balance Sheet, Cash Flow)
- Company name and period
- Revenue trends and major revenue sources
- Expense categories and cost structure
- Profitability metrics and margins
- Asset composition and liquidity
- Debt levels and equity structure
- Cash flow from operations, investing, financing
"""
    
    else:
        return base_prompt + """
Focus on:
- Document purpose and key financial data
- Important parties and entities involved
- Critical amounts, dates, and terms
- Any unusual or notable financial items
"""

def analyze_document_with_chatgpt(text_content: str, document_type: str) -> Dict[str, Any]:
    """Analyze document text using ChatGPT API (same approach as working Google Drive tool)"""
    if not REQUESTS_AVAILABLE or not OPENAI_API_KEY:
        raise Exception("Requests library not available or OpenAI API key not configured")
    
    try:
        logger.info(f"Starting ChatGPT analysis for {document_type}...")
        
        # Get appropriate prompt
        prompt = get_financial_analysis_prompt(document_type, text_content)
        
        logger.info(f"Using financial analysis prompt for {document_type}")
        logger.info("Calling OpenAI ChatGPT API...")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        # Try multiple models with fallback
        models_to_try = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
        
        for model_name in models_to_try:
            try:
                logger.info(f"Trying model: {model_name}")
                
                payload = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert financial analyst specializing in reading and interpreting financial statements, tax returns, and business financial documents. Always respond with valid JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.1  # Low temperature for factual analysis
                }
                
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                
                logger.info(f"API Response Status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    logger.info(f"✓ ChatGPT API call successful with {model_name}")
                    logger.info(f"Received response: {len(content)} characters")
                    break
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    logger.error(f"✗ Model {model_name} failed: {error_msg}")
                    if model_name == models_to_try[-1]:
                        raise Exception(error_msg)
                    continue
                    
            except Exception as model_error:
                logger.error(f"✗ Model {model_name} failed: {str(model_error)}")
                if model_name == models_to_try[-1]:  # Last model, re-raise error
                    raise model_error
                continue  # Try next model
        
        # Try to parse as JSON
        try:
            analysis = json.loads(content)
            logger.info("✓ Response parsed as valid JSON")
        except json.JSONDecodeError:
            logger.info("Response is not valid JSON, creating structured response")
            analysis = {
                "document_type": document_type,
                "raw_analysis": content,
                "parsed": False,
                "error": "Response was not valid JSON"
            }
        
        analysis['model_used'] = model_name
        analysis['analysis_method'] = 'text_extraction_chatgpt'
        
        logger.info(f"✓ Successfully analyzed document with ChatGPT")
        return analysis
        
    except Exception as e:
        logger.error(f"✗ Error analyzing document with ChatGPT: {str(e)}")
        logger.error(f"✗ Full error details: {repr(e)}")
        raise

def process_document_with_text_analysis(doc_info: Dict, drive_service) -> bool:
    """Process a document using text extraction + ChatGPT analysis (proven approach)"""
    try:
        logger.info(f"=== TEXT ANALYSIS MODE: Processing {doc_info['file_name']} ===")
        
        # Step 1: Download file
        logger.info("Step 1: Downloading file from Google Drive...")
        file_content = download_file_from_drive(drive_service, doc_info["drive_file_id"])
        logger.info(f"✓ Downloaded {len(file_content.getvalue())} bytes")
        
        # Step 2: Extract text content
        logger.info("Step 2: Extracting text content from PDF...")
        text_content = extract_pdf_text_content(file_content)
        if not text_content or len(text_content.strip()) < 100:
            raise Exception("Insufficient text content extracted from PDF")
        logger.info(f"✓ Extracted {len(text_content)} characters of text")
        
        # Step 3: Document type
        document_type = identify_document_type_from_filename(doc_info["file_name"])
        logger.info(f"✓ Document type: {document_type}")
        
        # Step 4: Analyze with ChatGPT
        logger.info("Step 4: Analyzing content with ChatGPT...")
        analysis_result = analyze_document_with_chatgpt(text_content, document_type)
        logger.info(f"✓ ChatGPT analysis complete")
        
        # Step 5: Store results in database with complete metadata
        logger.info("Step 5: Storing complete results in database...")
        conn = get_postgres_connection()
        try:
            with conn.cursor() as cursor:
                # Check if document exists in ai_data.documents
                cursor.execute(
                    """
                    SELECT id FROM ai_data.documents 
                    WHERE drive_file_id = %s
                    """,
                    (doc_info["drive_file_id"],)
                )
                
                doc_row = cursor.fetchone()
                
                # Prepare comprehensive data
                full_document_text = text_content[:5000] + "..." if len(text_content) > 5000 else text_content
                financial_summary = json.dumps({
                    "analysis_method": "text_extraction_chatgpt",
                    "document_type": document_type,
                    "analysis_result": analysis_result,
                    "processing_timestamp": time.time(),
                    "text_length": len(text_content)
                })
                vision_analysis = json.dumps(analysis_result)  # Store ChatGPT analysis in vision_analysis field
                
                if doc_row:
                    # Update existing document with complete metadata from tracking
                    document_id = doc_row[0]
                    logger.info(f"Updating existing document with ID: {document_id}")
                    
                    cursor.execute(
                        """
                        UPDATE ai_data.documents 
                        SET client_id = %s,
                            document_name = %s,
                            year = %s,
                            month = %s,
                            day = %s,
                            class = %s,
                            subclass = %s,
                            is_new = %s,
                            full_document_text = %s, 
                            financial_summary = %s,
                            vision_analysis = %s,
                            total_chunks = 1,
                            processed_at = NOW()
                        WHERE drive_file_id = %s
                        """,
                        (
                            doc_info.get("client_id"),
                            doc_info.get("file_name"),
                            doc_info.get("year"),
                            doc_info.get("month"),
                            doc_info.get("day"),
                            doc_info.get("class"),
                            doc_info.get("subclass"),
                            doc_info.get("is_new"),
                            full_document_text,
                            financial_summary,
                            vision_analysis,
                            doc_info["drive_file_id"]
                        )
                    )
                    
                    rows_updated = cursor.rowcount
                    logger.info(f"✓ Database UPDATE: {rows_updated} rows updated with complete metadata")
                else:
                    # Insert new document with complete metadata
                    logger.info("Document not found in ai_data.documents, inserting new record...")
                    
                    cursor.execute(
                        """
                        INSERT INTO ai_data.documents 
                        (drive_file_id, client_id, document_name, year, month, day, class, subclass, is_new,
                         full_document_text, financial_summary, vision_analysis, total_chunks, processed_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        RETURNING id
                        """,
                        (
                            doc_info["drive_file_id"],
                            doc_info.get("client_id"),
                            doc_info.get("file_name"),
                            doc_info.get("year"),
                            doc_info.get("month"),
                            doc_info.get("day"),
                            doc_info.get("class"),
                            doc_info.get("subclass"),
                            doc_info.get("is_new"),
                            full_document_text,
                            financial_summary,
                            vision_analysis,
                            1
                        )
                    )
                    
                    document_id = cursor.fetchone()[0]
                    logger.info(f"✓ New document inserted with ID: {document_id}")
                
                # Clean up old embeddings and insert comprehensive embedding
                cursor.execute(
                    """
                    DELETE FROM ai_data.document_embeddings 
                    WHERE drive_file_id = %s
                    """,
                    (doc_info["drive_file_id"],)
                )
                
                # Create embedding with the analysis result
                cursor.execute(
                    """
                    INSERT INTO ai_data.document_embeddings
                    (client_id, document_name, chunk_index, chunk_text, year, month, day,
                     class, subclass, document_id, drive_file_id, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        doc_info.get("client_id"),
                        doc_info.get("file_name"),
                        1,
                        f"Financial Analysis: {str(analysis_result)[:1000]}",
                        doc_info.get("year"),
                        doc_info.get("month"),
                        doc_info.get("day"),
                        doc_info.get("class"),
                        doc_info.get("subclass"),
                        document_id,
                        doc_info["drive_file_id"],
                        [0.0] * 384  # Dummy embedding vector
                    )
                )
                
                conn.commit()
                logger.info(f"✓ Complete data committed to database")
                
                # Update status
                update_document_status(doc_info["id"], "processed")
                logger.info("✓ Status updated to processed")
                
                return True
                
        except Exception as db_error:
            conn.rollback()
            logger.error(f"✗ Database error: {str(db_error)}")
            logger.error(f"✗ Full database error: {repr(db_error)}")
            return False
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"✗ Overall process error: {str(e)}")
        logger.error(f"✗ Full process error: {repr(e)}")
        update_document_status(doc_info["id"], "error", str(e))
        return False

def process_pending_documents():
    """Process all pending documents using text extraction + ChatGPT analysis."""
    try:
        # Check database schema first
        if not check_database_schema():
            logger.error("Database schema check failed, aborting")
            return {
                "documents_found": 0,
                "documents_processed": 0,
                "errors": 1,
                "error": "Database schema check failed"
            }
        
        # Initialize Drive API
        drive_service = get_drive_service()
        
        # Get pending documents
        pending_docs = get_pending_documents()
        logger.info(f"Found {len(pending_docs)} pending documents")
        
        documents_processed = 0
        errors = 0
        
        # Process each document
        for doc in pending_docs:
            logger.info(f"Processing document: {doc['file_name']} (ID: {doc['id']})")
            success = process_document_with_text_analysis(doc, drive_service)
            if success:
                documents_processed += 1
            else:
                errors += 1
        
        return {
            "documents_found": len(pending_docs),
            "documents_processed": documents_processed,
            "errors": errors
        }
    except Exception as e:
        logger.error(f"Error in process_pending_documents: {str(e)}")
        return {
            "documents_found": 0,
            "documents_processed": 0,
            "errors": 1,
            "error": str(e)
        }

class SimpleHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests - for health checks."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {
            'status': 'healthy',
            'approach': 'text-extraction-chatgpt',
            'capabilities': {
                'pymupdf': PYMUPDF_AVAILABLE,
                'pil': PIL_AVAILABLE,
                'tesseract': TESSERACT_AVAILABLE,
                'requests': REQUESTS_AVAILABLE,
                'api_key_configured': bool(OPENAI_API_KEY)
            }
        }
        self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def do_POST(self):
        """Handle POST requests - for document processing."""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
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
    
    try:
        server = socketserver.TCPServer(('0.0.0.0', port), SimpleHTTPRequestHandler)
        logger.info(f'Starting Text-based document processor on port {port}')
        server.serve_forever()
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

if __name__ == "__main__":
    # Test basic functionality before starting server
    logger.info("=== Starting Text-Based Document Processor ===")
    logger.info(f"PyMuPDF available: {PYMUPDF_AVAILABLE}")
    logger.info(f"PIL available: {PIL_AVAILABLE}")
    logger.info(f"Tesseract available: {TESSERACT_AVAILABLE}")
    logger.info(f"Requests available: {REQUESTS_AVAILABLE}")
    logger.info(f"OpenAI API key configured: {bool(OPENAI_API_KEY)}")
    
    # Test OpenAI API key
    if REQUESTS_AVAILABLE and OPENAI_API_KEY:
        try:
            logger.info("Testing OpenAI API connection...")
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello, this is a test. Respond with 'API working'."}],
                "max_tokens": 10
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('choices') and result['choices'][0]['message']['content']:
                    logger.info("✓ OpenAI API key is working")
                else:
                    logger.error("✗ OpenAI API returned empty response")
            else:
                logger.error(f"✗ OpenAI API test failed: HTTP {response.status_code} - {response.text}")
                
        except Exception as api_error:
            logger.error(f"✗ OpenAI API test failed: {str(api_error)}")
    else:
        logger.error("✗ OpenAI API key not configured or requests library not available")
    
    # Test database connection
    try:
        logger.info("Testing database connection...")
        conn = get_postgres_connection()
        conn.close()
        logger.info("✓ Database connection test successful")
    except Exception as e:
        logger.error(f"✗ Database connection test failed: {e}")
    
    # Start server
    logger.info("Starting HTTP server...")
    run_server()
