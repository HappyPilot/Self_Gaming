#!/usr/bin/env python3
"""Simple stub LLM endpoint returning canned reasoning/action pairs."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import time


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            payload = {}
        user_messages = [msg.get("content", "") for msg in payload.get("messages", []) if msg.get("role") == "user"]
        prompt = "\\n".join(user_messages).lower()
        if "reasoning paragraph" in prompt:
            content = "The UI shows a search area; we should likely type a query."
        else:
            content = "Click the search box and type example"
        response = {
            "id": f"stub-{time.time():.0f}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model", "local-stub"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
        data = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    host = os.getenv("LLM_STUB_HOST", "127.0.0.1")
    port = int(os.getenv("LLM_STUB_PORT", "8000"))
    server = HTTPServer((host, port), Handler)
    print(f"Stub LLM server listening on {host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
