import os
import psycopg2
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

# PostgreSQL Configuration from environment variables
PG_HOST = os.environ.get("PG_HOST")
PG_DATABASE = os.environ.get("PG_DATABASE", "postgres")  # Default to 'postgres'
PG_USER = os.environ.get("PG_USER")
PG_PASSWORD = os.environ.get("PG_PASSWORD")

def test_postgres_connection():
    """Test connection to PostgreSQL and return status."""
    try:
        # Log connection parameters (for debugging)
        print(f"Connecting to: host={PG_HOST}, database={PG_DATABASE}, user={PG_USER}")
        
        conn = psycopg2.connect(
            host=PG_HOST,
            database=PG_DATABASE,
            user=PG_USER,
            password=PG_PASSWORD
        )
        cursor = conn.cursor()
        
        # Test general connectivity
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        
        # Test schema existence
        cursor.execute("SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = 'ai_data')")
        schema_exists = cursor.fetchone()[0]
        
        # Test tables if schema exists
        if schema_exists:
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'ai_data'")
            tables = [row[0] for row in cursor.fetchall()]
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
        return False, {
            "message": f"Database connection failed: {str(e)}"
        }

class DatabaseTestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        # Test database connection
        success, details = test_postgres_connection()
        
        response = {
            "service": "healthy",
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
        
        self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        # Test database connection
        success, details = test_postgres_connection()
        
        response = {
            "service": "healthy",
            "database_connection": {
                "success": success,
                "details": details
            }
        }
        
        self.wfile.write(json.dumps(response).encode('utf-8'))

def run():
    port = int(os.environ.get('PORT', 8080))
    print(f'Starting server on port {port}')
    server_address = ('', port)
    httpd = HTTPServer(server_address, DatabaseTestHandler)
    print('Server is running')
    httpd.serve_forever()

if __name__ == '__main__':
    print('Starting database test application...')
    run()
