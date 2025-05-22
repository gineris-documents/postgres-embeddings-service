import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'Service is healthy!')
    
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'Received POST request')

def run():
    port = int(os.environ.get('PORT', 8080))
    print(f'Starting server on port {port}')
    server_address = ('', port)
    httpd = HTTPServer(server_address, SimpleHandler)
    print('Server is running')
    httpd.serve_forever()

if __name__ == '__main__':
    print('Starting application...')
    run()
