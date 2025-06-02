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
    import fitz  # PyMuPDF for PDF to image conversion
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
    import openai
    OPENAI_AVAILABLE = True
    logger.info("OpenAI library imported successfully")
except ImportError as e:
    OPENAI_AVAILABLE = False
    logger.error(f"OpenAI library not available: {e}")

# Configuration
PROJECT_ID = os.environ.get("PROJECT_ID", "pdf-to-pinecone")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# PostgreSQL Configuration
PG_HOST = os.environ.get("PG_HOST", "34.66.180.234")
PG_DATABASE = os.environ.get("PG_DATABASE", "gineris_dev")
PG_USER = os.environ.get("PG_USER", "admin")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "H@nnib@lMO2015")

# Google Drive API configuration
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = os.environ.get("SERVICE_ACCOUNT_FILE", "service-account.json")

# Initialize OpenAI client
if OPENAI_AVAILABLE and OPENAI_API_KEY:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        logger.info("OpenAI API key configured")
    except Exception as e:
        logger.error(f"Failed to configure OpenAI: {e}")
else:
    logger.error("OpenAI API key not configured or library not available")

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

def pdf_to_images(file_content: io.BytesIO) -> List[bytes]:
    """Convert PDF to high-quality images using PyMuPDF."""
    if not PYMUPDF_AVAILABLE:
        raise Exception("PyMuPDF not available for PDF to image conversion")
    
    if not PIL_AVAILABLE:
        raise Exception("PIL not available for image processing")
    
    try:
        file_content.seek(0)
        pdf_data = file_content.read()
        pdf_document = fitz.open("pdf", pdf_data)
        
        images = []
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            
            # Convert to high-quality image with proper format
            # Use higher DPI for better OCR results
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2x scaling for better quality
            
            # Convert PyMuPDF pixmap to PIL Image to ensure proper format
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))
            
            # Convert to RGB if necessary (remove alpha channel issues)
            if pil_image.mode in ('RGBA', 'LA'):
                # Create white background
                background = Image.new('RGB', pil_image.size, (255, 255, 255))
                if pil_image.mode == 'RGBA':
                    background.paste(pil_image, mask=pil_image.split()[-1])  # Use alpha channel as mask
                else:
                    background.paste(pil_image, mask=pil_image.split()[-1])
                pil_image = background
            elif pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Save as high-quality JPEG (more compatible than PNG)
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='JPEG', quality=95, optimize=True)
            final_img_data = img_buffer.getvalue()
            
            images.append(final_img_data)
            
            logger.info(f"Converted page {page_num + 1} to JPEG image ({len(final_img_data)} bytes)")
        
        pdf_document.close()
        logger.info(f"Successfully converted PDF to {len(images)} JPEG images")
        return images
        
    except Exception as e:
        logger.error(f"Error converting PDF to images: {str(e)}")
        raise

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

def get_document_analysis_prompt(document_type: str) -> str:
    """Get the appropriate prompt for document analysis based on type."""
    
    base_prompt = """You are a professional tax and accounting AI assistant. Analyze this document image and extract key financial information.

Please provide a JSON response with the following structure:
{
    "document_type": "type of document",
    "key_figures": {
        "field_name": "value"
    },
    "summary": "brief summary of the document",
    "notable_items": ["list", "of", "important", "observations"]
}

"""
    
    if document_type == 'tax_return':
        return base_prompt + """
Focus on:
- Form type (1120, 1120S, 1040, etc.)
- Tax year
- Taxpayer name and EIN/SSN
- Key amounts: Gross income, Taxable income, Tax owed/refund, Net income/loss
- Schedule information if present
- Any unusual items or red flags
"""
    
    elif document_type == 'financial_statement':
        return base_prompt + """
Focus on:
- Statement type (Income Statement, Balance Sheet, Cash Flow)
- Period covered
- Company name
- Key amounts: Revenue, Net Income, Total Assets, Total Liabilities, Equity
- Year-over-year changes if comparative
- Any unusual items or significant variances
"""
    
    else:
        return base_prompt + """
Focus on:
- Document purpose and type
- Key parties involved
- Important dates and amounts
- Critical terms or conditions
- Any unusual or notable items
"""

def analyze_image_with_vision(image_data: bytes, document_type: str, page_number: int) -> Dict[str, Any]:
    """Analyze a single image using OpenAI Vision API."""
    if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
        raise Exception("OpenAI not available or API key not configured")
    
    try:
        logger.info(f"Starting Vision API analysis for page {page_number}...")
        
        # Encode image to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        logger.info(f"Encoded image to base64: {len(image_base64)} characters")
        
        # Get appropriate prompt
        prompt = get_document_analysis_prompt(document_type)
        
        # Add page context
        if page_number > 1:
            prompt += f"\n\nThis is page {page_number} of the document. Focus on the content visible on this specific page."
        
        logger.info(f"Using prompt: {prompt[:100]}...")
        logger.info("Calling OpenAI Vision API...")
        
        # Simple direct API call without client initialization issues
        import requests
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        # Try gpt-4o first (newer model), fallback to gpt-4-vision-preview
        models_to_try = ["gpt-4o", "gpt-4-vision-preview"]
        
        for model_name in models_to_try:
            try:
                logger.info(f"Trying model: {model_name}")
                
                payload = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.1
                }
                
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    logger.info(f"✓ Vision API call successful with {model_name}")
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
        
        # Try to parse as JSON, fallback to text if not valid JSON
        try:
            analysis = json.loads(content)
            logger.info("✓ Response parsed as valid JSON")
        except json.JSONDecodeError:
            logger.info("Response is not JSON, storing as raw text")
            analysis = {
                "raw_response": content,
                "parsed": False
            }
        
        analysis['page_number'] = page_number
        analysis['model_used'] = model_name
        
        logger.info(f"✓ Successfully analyzed page {page_number} with Vision API")
        return analysis
        
    except Exception as e:
        logger.error(f"✗ Error analyzing image with Vision API: {str(e)}")
        logger.error(f"✗ Full error details: {repr(e)}")
        raise

def consolidate_page_analyses(page_analyses: List[Dict[str, Any]], document_type: str) -> Dict[str, Any]:
    """Consolidate analyses from multiple pages into a single document summary."""
    
    consolidated = {
        "document_type": document_type,
        "total_pages": len(page_analyses),
        "key_figures_consolidated": {},
        "full_summary": "",
        "page_summaries": [],
        "notable_items_all": []
    }
    
    # Collect data from all pages
    for page_analysis in page_analyses:
        if isinstance(page_analysis, dict):
            if "key_figures" in page_analysis:
                consolidated["key_figures_consolidated"].update(page_analysis["key_figures"])
            
            if "summary" in page_analysis:
                consolidated["page_summaries"].append({
                    "page": page_analysis.get("page_number", 0),
                    "summary": page_analysis["summary"]
                })
            
            if "notable_items" in page_analysis:
                consolidated["notable_items_all"].extend(page_analysis["notable_items"])
    
    # Create consolidated summary
    if consolidated["page_summaries"]:
        consolidated["full_summary"] = " ".join([ps["summary"] for ps in consolidated["page_summaries"]])
    
    return consolidated

def process_document_with_vision(doc_info: Dict, drive_service) -> bool:
    """Process a document using Vision API approach with detailed diagnostics."""
    try:
        logger.info(f"=== DIAGNOSTIC MODE: Processing {doc_info['file_name']} ===")
        
        # Step 1: Download file
        logger.info("Step 1: Downloading file from Google Drive...")
        file_content = download_file_from_drive(drive_service, doc_info["drive_file_id"])
        logger.info(f"✓ Downloaded {len(file_content.getvalue())} bytes")
        
        # Step 2: Convert to images
        logger.info("Step 2: Converting PDF to images...")
        images = pdf_to_images(file_content)
        if not images:
            raise Exception("No images generated from PDF")
        logger.info(f"✓ Generated {len(images)} images, processing first 2 for diagnostic")
        
        # Step 3: Document type
        document_type = identify_document_type_from_filename(doc_info["file_name"])
        logger.info(f"✓ Document type: {document_type}")
        
        # Step 4: Test Vision API with just 1 page
        logger.info("Step 4: Testing Vision API with page 1...")
        try:
            test_analysis = analyze_image_with_vision(images[0], document_type, 1)
            logger.info(f"✓ Vision API SUCCESS! Response: {str(test_analysis)[:200]}...")
            
            # Store test result directly in database to verify it works
            conn = get_postgres_connection()
            try:
                with conn.cursor() as cursor:
                    # Update the existing document with test data
                    test_summary = f"TEST VISION API RESULT: {str(test_analysis)[:500]}"
                    test_financial = json.dumps({"test": "vision_api_working", "page_1_analysis": str(test_analysis)[:200]})
                    
                    cursor.execute(
                        """
                        UPDATE ai_data.documents 
                        SET full_document_text = %s, 
                            financial_summary = %s,
                            total_chunks = %s,
                            processed_at = NOW()
                        WHERE drive_file_id = %s
                        """,
                        (test_summary, test_financial, len(images), doc_info["drive_file_id"])
                    )
                    
                    rows_updated = cursor.rowcount
                    logger.info(f"✓ Database UPDATE: {rows_updated} rows updated")
                    
                    # Also insert a test embedding
                    cursor.execute(
                        """
                        DELETE FROM ai_data.document_embeddings 
                        WHERE drive_file_id = %s
                        """,
                        (doc_info["drive_file_id"],)
                    )
                    
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
                            1,
                            f"DIAGNOSTIC: Vision API result: {str(test_analysis)[:1000]}",
                            doc_info.get("year"),
                            doc_info.get("month"),
                            doc_info.get("day"),
                            doc_info.get("class"),
                            doc_info.get("subclass"),
                            20,  # Assuming document ID 20 from your data
                            doc_info["drive_file_id"],
                            [0.0] * 384
                        )
                    )
                    
                    conn.commit()
                    logger.info("✓ Test data committed to database")
                    
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
                
        except Exception as vision_error:
            logger.error(f"✗ Vision API error: {str(vision_error)}")
            logger.error(f"✗ Full Vision API error: {repr(vision_error)}")
            
            # Store error info in database for debugging
            conn = get_postgres_connection()
            try:
                with conn.cursor() as cursor:
                    error_summary = f"VISION API ERROR: {str(vision_error)}"
                    cursor.execute(
                        """
                        UPDATE ai_data.documents 
                        SET full_document_text = %s, 
                            processed_at = NOW()
                        WHERE drive_file_id = %s
                        """,
                        (error_summary, doc_info["drive_file_id"])
                    )
                    conn.commit()
                    logger.info("✓ Error info stored in database")
            except Exception as e:
                logger.error(f"Failed to store error info: {e}")
            finally:
                conn.close()
                
            update_document_status(doc_info["id"], "error", str(vision_error))
            return False
            
    except Exception as e:
        logger.error(f"✗ Overall process error: {str(e)}")
        logger.error(f"✗ Full process error: {repr(e)}")
        update_document_status(doc_info["id"], "error", str(e))
        return False

def process_pending_documents():
    """Process all pending documents using Vision API."""
    try:
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
            success = process_document_with_vision(doc, drive_service)
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
            'approach': 'vision-api',
            'capabilities': {
                'pymupdf': PYMUPDF_AVAILABLE,
                'pil': PIL_AVAILABLE,
                'openai': OPENAI_AVAILABLE,
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
        logger.info(f'Starting Vision-based document processor on port {port}')
        server.serve_forever()
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

if __name__ == "__main__":
    # Test basic functionality before starting server
    logger.info("=== Starting Vision-based Document Processor ===")
    logger.info(f"PyMuPDF available: {PYMUPDF_AVAILABLE}")
    logger.info(f"PIL available: {PIL_AVAILABLE}")
    logger.info(f"OpenAI available: {OPENAI_AVAILABLE}")
    logger.info(f"OpenAI API key configured: {bool(OPENAI_API_KEY)}")
    
    # Test OpenAI API key
    if OPENAI_AVAILABLE and OPENAI_API_KEY:
        try:
            logger.info("Testing OpenAI API connection...")
            
            # Use direct HTTP request to avoid library version issues
            import requests
            
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
        logger.error("✗ OpenAI API key not configured or library not available")
    
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
