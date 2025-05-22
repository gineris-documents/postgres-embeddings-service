import http.server
import socketserver
import os
import sys
import logging
import json
import psycopg2
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PORT = int(os.environ.get('PORT', 8080))

# PostgreSQL Configuration from environment variables
PG_HOST = os.environ.get("PG_HOST")
PG_DATABASE = os.environ.get("PG_DATABASE", "postgres")
PG_USER = os.environ.get("PG_USER")
PG_PASSWORD = os.environ.get("PG_PASSWORD")

logger.info(f"Starting server with Python {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"PostgreSQL configuration: host={PG_HOST}, database={PG_DATABASE}, user={PG_USER}")

# Global variable to store the model and load time
model_info = {
    "model": None,
    "load_time": None,
    "is_loaded": False,
    "error": None
}

def test_database():
    """Test database connection and verify tables."""
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            database=PG_DATABASE,
            user=PG_USER,
            password=PG_PASSWORD
        )
        
        cursor = conn.cursor()
        
        # Check tables
        tables = ['documents', 'document_embeddings', 'document_tracking']
        table_info = {}
        
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM ai_data.{table}")
            count = cursor.fetchone()[0]
            table_info[table] = {
                "exists": True,
                "row_count": count
            }
        
        cursor.close()
        conn.close()
        
        return True, table_info
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        return False, {"error": str(e)}

def load_model():
    """Load the sentence-transformers model."""
    try:
        start_time = time.time()
        logger.info("Loading sentence-transformers model...")
        
        # Import here to avoid loading issues on startup
        from sentence_transformers import SentenceTransformer
        
        # Use a smaller model for faster loading
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
        # Test encoding a simple sentence
        test_sentence = "This is a test sentence."
        embedding = model.encode(test_sentence)
        logger.info(f"Test encoding successful. Embedding shape: {embedding.shape}")
        
        model_info["model"] = model
        model_info["load_time"] = load_time
        model_info["is_loaded"] = True
        model_info["error"] = None
        
        return True, {
            "load_time": load_time,
            "embedding_dim": embedding.shape[0]
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error loading model: {error_msg}")
        model_info["error"] = error_msg
        return False, {"error": error_msg}

class ModelTestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        logger.info(f"Received GET request to {self.path} from {self.client_address}")
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        # Check database
        db_success, db_info = test_database()
        
        # We don't load the model on startup to avoid slowing down container initialization
        # Instead, we'll load it on the first request
        if not model_info["is_loaded"] and model_info["error"] is None:
            model_success, model_details = load_model()
        else:
            model_success = model_info["is_loaded"]
            model_details = {
                "load_time": model_info["load_time"],
                "error": model_info["error"]
            }
        
        response = {
            "service": "model-test-server",
            "path": self.path,
            "method": "GET",
            "database": {
                "success": db_success,
                "tables": db_info
            },
            "model": {
                "success": model_success,
                "details": model_details
            }
        }
        
        response_json = json.dumps(response)
        self.wfile.write(response_json.encode('utf-8'))
    
    def do_POST(self):
        logger.info(f"Received POST request to {self.path} from {self.client_address}")
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')
        logger.info(f"POST data received: {post_data}")
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        # Similar response as GET, but we could add more functionality here
        # for processing documents
        
        # Check database
        db_success, db_info = test_database()
        
        # Load model if not already loaded
        if not model_info["is_loaded"] and model_info["error"] is None:
            model_success, model_details = load_model()
        else:
            model_success = model_info["is_loaded"]
            model_details = {
                "load_time": model_info["load_time"],
                "error": model_info["error"]
            }
        
        response = {
            "service": "model-test-server",
            "path": self.path,
            "method": "POST",
            "database": {
                "success": db_success,
                "tables": db_info
            },
            "model": {
                "success": model_success,
                "details": model_details
            }
        }
        
        response_json = json.dumps(response)
        self.wfile.write(response_json.encode('utf-8'))

try:
    httpd = socketserver.TCPServer(("", PORT), ModelTestHandler)
    logger.info(f"Server listening on port {PORT}")
    httpd.serve_forever()
except Exception as e:
    logger.error(f"Error starting server: {str(e)}")
    sys.exit(1)
