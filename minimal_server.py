import http.server
import socketserver
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PORT = int(os.environ.get('PORT', 8080))

logger.info(f"Starting server with Python {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Directory contents: {os.listdir('.')}")
logger.info(f"Environment variables: {dict(os.environ)}")

class SimpleHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        logger.info(f"Received GET request to {self.path} from {self.client_address}")
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        message = f"Hello, World! You requested: {self.path}"
        self.wfile.write(message.encode('utf-8'))
        logger.info(f"Sent response: {message}")
    
    def do_POST(self):
        logger.info(f"Received POST request to {self.path} from {self.client_address}")
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')
        logger.info(f"POST data received: {post_data}")
        
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        message = f"Received POST request to {self.path}"
        self.wfile.write(message.encode('utf-8'))
        logger.info(f"Sent response: {message}")

try:
    httpd = socketserver.TCPServer(("", PORT), SimpleHandler)
    logger.info(f"Serving at port {PORT}")
    httpd.serve_forever()
except Exception as e:
    logger.error(f"Error starting server: {str(e)}")
    sys.exit(1)
