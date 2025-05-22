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
logger.info(f"PostgreSQL configuration: host={PG_HOST}, database={PG_DATABASE}, user={PG_USER}")

def check_tables():
    """Check for specific tables in the ai_data schema."""
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
        
        # Define expected tables
        expected_tables = ['documents', 'document_embeddings', 'document_tracking']
        table_info = {}
        
        # Check each table
        for table in expected_tables:
            # Check if table exists
            cursor.execute(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'ai_data' 
                AND table_name = %s
            )
            """, (table,))
            exists = cursor.fetchone()[0]
            
            if exists:
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM ai_data.{table}")
                row_count = cursor.fetchone()[0]
                
                # Get column info
                cursor.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'ai_data' 
                AND table_name = %s
                """, (table,))
                columns = {row[0]: row[1] for row in cursor.fetchall()}
                
                table_info[table] = {
                    "exists": True,
                    "row_count": row_count,
                    "columns": columns
                }
            else:
                table_info[table] = {
                    "exists": False
                }
        
        cursor.close()
        conn.close()
        
        return True, table_info
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        return False, {"error": str(e)}

class TableCheckHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        logger.info(f"Received GET request to {self.path} from {self.client_address}")
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        # Check tables
        success, table_info = check_tables()
        
        response = {
            "service": "table-check-server",
            "path": self.path,
            "method": "GET",
            "database_status": {
                "success": success,
                "table_info": table_info
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
        
        # Check tables
        success, table_info = check_tables()
        
        response = {
            "service": "table-check-server",
            "path": self.path,
            "method": "POST",
            "database_status": {
                "success": success,
                "table_info": table_info
            }
        }
        
        response_json = json.dumps(response)
        logger.info(f"Sending response: {response_json}")
        self.wfile.write(response_json.encode('utf-8'))

try:
    httpd = socketserver.TCPServer(("", PORT), TableCheckHandler)
    logger.info(f"Serving at port {PORT}")
    httpd.serve_forever()
except Exception as e:
    logger.error(f"Error starting server: {str(e)}")
    sys.exit(1)
