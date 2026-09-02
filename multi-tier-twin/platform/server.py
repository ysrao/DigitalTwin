#!/usr/bin/env python3
"""Standard-library browser server for the RAN decision-platform MVP."""

import json
import mimetypes
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

from engine import run_evaluation

WEB = Path(__file__).resolve().parent / "web"


def _twin_api():
    """Import the twin lazily so the slice MVP still serves if it fails."""
    from compare import SCENARIOS, optimize, preview, run_comparison
    return SCENARIOS, run_comparison, optimize, preview


class Handler(BaseHTTPRequestHandler):
    def send_json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            return self.send_json({"ok": True, "service": "RAN Slice Decision Platform MVP"})
        if path == "/api/twin/scenarios":
            try:
                scenarios, _, _, _ = _twin_api()
                return self.send_json({"scenarios": [
                    {"id": key, "label": value["label"],
                     "description": value["description"]}
                    for key, value in scenarios.items()]})
            except Exception as exc:
                return self.send_json({"error": str(exc)}, 500)
        rel = "index.html" if path == "/" else path.lstrip("/")
        target = (WEB / rel).resolve()
        if WEB.resolve() not in target.parents and target != WEB.resolve():
            return self.send_error(403)
        if not target.is_file():
            return self.send_error(404)
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)

    def do_POST(self):
        path = urlparse(self.path).path
        routes = {"/api/evaluate", "/api/twin/preview", "/api/twin/compare",
                  "/api/twin/optimize"}
        if path not in routes:
            return self.send_error(404)
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
            if path == "/api/evaluate":
                return self.send_json(run_evaluation(payload))
            _, compare, optimize, preview = _twin_api()
            handler = {"/api/twin/preview": preview,
                       "/api/twin/compare": compare,
                       "/api/twin/optimize": optimize}[path]
            self.send_json(handler(payload))
        except Exception as exc:
            self.send_json({"error": str(exc)}, 400)


if __name__ == "__main__":
    address = ("127.0.0.1", 8765)
    print(f"RAN Slice Decision Platform: http://{address[0]}:{address[1]}")
    ThreadingHTTPServer(address, Handler).serve_forever()
