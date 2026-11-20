"""Password (token) protected reverse proxy in front of an Ollama server - standard library only.

Runs next to Ollama on the GPU machine (Google Colab, a rented GPU, an office server).
Only requests carrying ``Authorization: Bearer <OLLAMA_API_KEY>`` reach Ollama, so a public
tunnel URL alone is not enough to use the GPU. The kitchen generator sends that header
automatically when ``OLLAMA_API_KEY`` is set on the laptop (the ``ollama`` Python client reads it).

    OLLAMA_API_KEY=<secret> python ollama_auth_proxy.py <port>

No third-party packages: works on any Python 3.8+ (including Colab's) without pip installs.
"""
import hmac
import http.client
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM_HOST = os.environ.get("OLLAMA_UPSTREAM_HOST", "127.0.0.1")
UPSTREAM_PORT = int(os.environ.get("OLLAMA_UPSTREAM_PORT", "11434"))
TOKEN = os.environ["OLLAMA_API_KEY"]
DROP = {"host", "authorization", "content-length", "connection", "accept-encoding", "transfer-encoding"}


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _reply(self, status: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _forward(self) -> None:
        if not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {TOKEN}"):
            return self._reply(401, b'{"error":"unauthorized"}')
        length = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(length) if length else None
        headers = {k: v for k, v in self.headers.items() if k.lower() not in DROP}
        try:
            conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=None)
            conn.request(self.command, self.path, body=data, headers=headers)
            resp = conn.getresponse()
            body = resp.read()
            ctype = resp.getheader("Content-Type", "application/json")
            conn.close()
        except OSError as exc:
            return self._reply(502, json.dumps({"error": f"ollama not reachable: {exc}"}).encode())
        self._reply(resp.status, body, ctype)

    do_GET = do_POST = do_DELETE = do_HEAD = _forward

    def log_message(self, fmt, *args):  # one short line per request
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 11435
    print(f"Ollama password proxy on 127.0.0.1:{port} -> {UPSTREAM_HOST}:{UPSTREAM_PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Proxy).serve_forever()
