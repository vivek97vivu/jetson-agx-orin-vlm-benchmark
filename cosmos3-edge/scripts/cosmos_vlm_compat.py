#!/usr/bin/env python3
"""Bridge Live VLM WebUI image_url frames to TensorRT Edge-LLM image paths."""

import base64
import binascii
import json
import os
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

UPSTREAM = os.environ.get("COSMOS_UPSTREAM", "http://127.0.0.1:8000")
MEDIA_DIR = Path(os.environ.get(
    "COSMOS_MEDIA_DIR", "/home/algosium/cosmos/cosmos3-edge/media"
))
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.rstrip("/") != "/v1/models":
            self.send_error(404)
            return
        self.forward("GET", f"{UPSTREAM}/v1/models", None)

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_error(404)
            return
        saved = []
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            for message in payload.get("messages", []):
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                converted = []
                for item in content:
                    if item.get("type") == "image_url":
                        image_url = item.get("image_url", {}).get("url", "")
                        if not image_url.startswith("data:image/") or "," not in image_url:
                            raise ValueError("Only inline camera frames are supported")
                        header, encoded = image_url.split(",", 1)
                        extension = ".png" if "png" in header.lower() else ".jpg"
                        path = MEDIA_DIR / f"webcam-{uuid.uuid4().hex}{extension}"
                        path.write_bytes(base64.b64decode(encoded, validate=True))
                        saved.append(path)
                        converted.append({"type": "image", "image": str(path)})
                    else:
                        converted.append(item)
                message["content"] = converted
            self.forward("POST", f"{UPSTREAM}/v1/chat/completions", json.dumps(payload).encode())
        except (ValueError, KeyError, TypeError, binascii.Error) as exc:
            self.send_json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
        except Exception as exc:
            self.send_json(502, {"error": {"message": str(exc), "type": "proxy_error"}})
        finally:
            for path in saved:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def forward(self, method, url, body):
        headers = {"Authorization": self.headers.get("Authorization", "Bearer EMPTY")}
        if body is not None:
            headers["Content-Type"] = "application/json"
        try:
            upstream = urlopen(Request(url, data=body, headers=headers, method=method), timeout=120)
            response_body = upstream.read()
            self.send_response(upstream.status)
            self.send_header("Content-Type", upstream.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        except Exception as exc:
            self.send_json(502, {"error": {"message": str(exc), "type": "proxy_error"}})

    def send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8001), Handler).serve_forever()
