import os
import sys
import psycopg2
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

# Print Python version and path
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir('.')}")

# PostgreSQL Configuration from environment variables
PG_HOST = os.environ.get("PG_HOST")
PG_DATABASE = os.environ.get("PG_DATABASE", "postgres")
PG_USER = os.environ.get("PG_USER")
PG_PASSWORD = os.environ.get("PG_PASSWORD")

def test_postgres_connection():
    """Test connection to PostgreSQL and return status."""
    try:
        print(f"Attempting database connection to: {PG_HOST}/{PG_DATABASE} as {PG_USER}")
        
        conn = psycopg2.connect(
            host=PG_HOST,
            database=PG_DATABASE,
            user=PG_USER,
            password=PG_PASSWORD
        )
        print("Database connection established successfully")
        
        cursor = conn.cursor()
        
        # Test general connectivity
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        print(f"Basic query result: {result}")
        
        # Test schema existence
        cursor.execute("SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = 'ai_data')")
        schema_exists = cursor.fetchone()[0]
        print(f"Schema 'ai_data' exists: {schema_exists}")
        
        # Test tables if schema exists
        if schema_exists:
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'ai_data'")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"Tables in 'ai_data' schema: {tables}")
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
        print(f"Database connection error: {str(e)}")
        return False, {
            "message": f"Database connection failed: {str(e)}"
        }

class DebugHandler(BaseHTTPRequestHandler):
    def log_request(self, code='-', size='-'):
        print(f"Request: {self.command} {self.path} {code} {size}")
        BaseHTTPRequestHandler.log_request(self, code, size)

    def do_GET(self):
        print(f"Handling GET request to {self.path}")
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        # Test database connection
        success, details = test_postgres_connection()
        
        response = {
            "service": "debug-server",
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
        print(f"Sending response: {response_json}")
        self.wfile.write(response_json.encode('utf-8'))
    
    def do_POST(self):
        print(f"Handling POST request to {self.path}")
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')
        print(f"POST data received: {post_data}")
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        # Test database connection
        success, details = test_postgres_connection()
        
        response = {
            "service": "debug-server",
            "path": self.path,
            "method": "POST",
            "post_data": post_data,
            "database_connection": {
                "success": success,
                "details": details
            }
        }
        
        response_json = json.dumps(response)
        print(f"Sending response: {response_json}")
        self.wfile.write(response_json.encode('utf-8'))

def run():
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting debug server on port {port}")
    server_address = ('', port)
    httpd = HTTPServer(server_address, DebugHandler)
    print('Debug server is running and ready to accept requests')
    httpd.serve_forever()

if __name__ == '__main__':
    print('**** Starting debug application... ****')
    run()
