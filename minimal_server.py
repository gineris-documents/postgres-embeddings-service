import http.server
import socketserver
import os

PORT = int(os.environ.get('PORT', 8080))

class SimpleHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'Hello, World!')

httpd = socketserver.TCPServer(("", PORT), SimpleHandler)
print(f"Serving at port {PORT}")
httpd.serve_forever()
