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

def check_permissions():
    """Check database permissions in detail."""
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
        
        # Check database user
        cursor.execute("SELECT current_user, current_database(), session_user")
        current_user, current_db, session_user = cursor.fetchone()
        
        # Check schema existence directly with a query
        cursor.execute("SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = 'ai_data')")
        schema_exists_info = cursor.fetchone()[0]
        
        # Try to directly query the information_schema
        cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'ai_data'
        """)
        info_schema_tables = [row[0] for row in cursor.fetchall()]
        
        # Try a direct query to test each table
        direct_query_results = {}
        for table in ['documents', 'document_embeddings', 'document_tracking']:
            try:
                # First check if the table exists using pg_catalog
                cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_catalog.pg_tables 
                    WHERE schemaname = 'ai_data' AND tablename = %s
                )
                """, (table,))
                pg_catalog_exists = cursor.fetchone()[0]
                
                # Try to count rows
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM ai_data.{table}")
                    row_count = cursor.fetchone()[0]
                    can_query = True
                except Exception as e:
                    row_count = None
                    can_query = False
                    logger.error(f"Error querying {table}: {str(e)}")
                
                direct_query_results[table] = {
                    "pg_catalog_exists": pg_catalog_exists,
                    "can_query": can_query,
                    "row_count": row_count
                }
            except Exception as e:
                direct_query_results[table] = {
                    "error": str(e)
                }
        
        # Check user privileges
        cursor.execute("""
        SELECT table_schema, table_name, privilege_type
        FROM information_schema.table_privileges
        WHERE grantee = current_user
        AND table_schema = 'ai_data'
        """)
        privileges = []
        for row in cursor.fetchall():
            privileges.append({
                "schema": row[0],
                "table": row[1],
                "privilege": row[2]
            })
        
        cursor.close()
        conn.close()
        
        return True, {
            "current_user": current_user,
            "session_user": session_user,
            "current_database": current_db,
            "schema_exists_info": schema_exists_info,
            "info_schema_tables": info_schema_tables,
            "direct_query_results": direct_query_results,
            "privileges": privileges
        }
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        return False, {"error": str(e)}

class PermissionCheckHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        logger.info(f"Received GET request to {self.path} from {self.client_address}")
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        # Check permissions
        success, details = check_permissions()
        
        response = {
            "service": "permission-check-server",
            "path": self.path,
            "method": "GET",
            "database_permissions": {
                "success": success,
                "details": details
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
        
        # Check permissions
        success, details = check_permissions()
        
        response = {
            "service": "permission-check-server",
            "path": self.path,
            "method": "POST",
            "database_permissions": {
                "success": success,
                "details": details
            }
        }
        
        response_json = json.dumps(response)
        logger.info(f"Sending response: {response_json}")
        self.wfile.write(response_json.encode('utf-8'))

try:
    httpd = socketserver.TCPServer(("", PORT), PermissionCheckHandler)
    logger.info(f"Serving at port {PORT}")
    httpd.serve_forever()
except Exception as e:
    logger.error(f"Error starting server: {str(e)}")
    sys.exit(1)
