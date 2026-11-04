

"""Local 250-page demo site for permitted, reproducible crawler benchmarks."""
from __future__ import annotations

import argparse
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar
from urllib.parse import urlsplit


class Handler(BaseHTTPRequestHandler):
    failures: ClassVar[set[str]] = set()
    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/robots.txt":
            content = b"User-agent: *\nAllow: /\n"
            mime = "text/plain"
        elif path.startswith("/page/"):
            try:
                n = int(path.split("/")[2])
            except (IndexError, ValueError):
                self.send_error(404)
                return
            if not 0 <= n < 250:
                self.send_error(404)
                return
            if n == 7 and path not in self.failures:
                self.failures.add(path)
                self.send_response(429)
                self.send_header("Retry-After", "0")
                self.end_headers()
                return
            # Model network waiting without stressing a third-party server.
            time.sleep(0.04)
            links = "".join(f'<a href="/page/{i}">link {i}</a>' for i in range(250))
            content = f"<html><head><title>Findex {n}</title></head><body><h1>Page {n}</h1><p>Async indexing document {n} uniqueword{n}</p>{links}</body></html>".encode()
            mime = "text/html"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mime + "; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print(f"Local permitted demo: http://127.0.0.1:{args.port}/page/0", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


