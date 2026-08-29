#!/usr/bin/env python3
"""Static server for the XPBI demo with caching disabled (ES module dev)."""
import http.server, functools, sys

class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, must-revalidate')
        super().end_headers()

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8777
http.server.ThreadingHTTPServer(('127.0.0.1', port),
    functools.partial(NoCacheHandler, directory=__file__.rsplit('/', 1)[0])).serve_forever()
