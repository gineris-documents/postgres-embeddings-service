import http.server
import socketserver
import os
import sys
import logging
import json
import psycopg2

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
logger.info(f"Directory contents: {os.listdir('.')}")
logger.info(f"PostgreSQL configuration: host={PG_HOST}, database={PG_DATABASE}, user={PG_USER}")

def test_postgres_connection():
    """Test connection to PostgreSQL and return status."""
    try:
        logger.info(f"Attempting database connection to: {PG_HOST}/{PG_DATABASE} as {PG_USER}")
        
        conn = psycopg2.connect(
            host=PG_HOST,
            database=PG_DATABASE,
            user=PG_USER,
            password=PG_PASSWORD
        )
        logger.info("Database connection established successfully")
        
        cursor = conn.cursor()
        
        # Test general connectivity
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        logger.info(f"Basic query result: {result}")
        
        # Test schema existence
        cursor.execute("SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = 'ai_data')")
        schema_exists = cursor.fetchone()[0]
        logger.info(f"Schema 'ai_data' exists: {schema_exists}")
        
        # Test tables if schema exists
        if schema_exists:
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'ai_data'")
            tables = [row[0] for row in cursor.fetchall()]
            logger.info(f"Tables in 'ai_data' schema: {tables}")
        else:
            tables = []
        
        cursor.close()
        conn.close()
        
        return True, {
            "message": "Database connection successful",
            "schema_exists": schema_exists,
            "tables": tables
        }
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        return False, {
            "message": f"Database connection failed: {str(e)}"
        }

class DatabaseHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        logger.info(f"Received GET request to {self.path} from {self.client_address}")
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        # Test database connection
        success, details = test_postgres_connection()
        
        response = {
            "service": "database-server",
            "path": self.path,
            "method": "GET",
            "database_connection": {
                "success": success,
                "details": details
            },
            "environment": {
                "PG_HOST": PG_HOST,
                "PG_DATABASE": PG_DATABASE,
                "PG_USER": PG_USER,
                "PG_PASSWORD": "********" if PG_PASSWORD else None
            }
        }
        
        response_json = json.dumps(response)
        logger.info(f"Sending response: {response_json}")
        self.wfile.write(response_json.encode('utf-8'))
    
    def do_POST(self):
        logger.info(f"Received POST request to {self.path} from {self.client_address}")
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')
        logger.info(f"POST data received: {post_data}")
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        # Test database connection
        success, details = test_postgres_connection()
        
        response = {
            "service": "database-server",
            "path": self.path,
            "method": "POST",
            "post_data": post_data,
            "database_connection": {
                "success": success,
                "details": details
            }
        }
        
        response_json = json.dumps(response)
        logger.info(f"Sending response: {response_json}")
        self.wfile.write(response_json.encode('utf-8'))

try:
    httpd = socketserver.TCPServer(("", PORT), DatabaseHandler)
    logger.info(f"Serving at port {PORT}")
    httpd.serve_forever()
except Exception as e:
    logger.error(f"Error starting server: {str(e)}")
    sys.exit(1)
