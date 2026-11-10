"""Small standard-library HTTP layer shared by independent app processes."""

import json
import mimetypes
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent
MAX_REQUEST_BYTES = 5 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    page = ""

    def do_GET(self):
        path = urlsplit(self.path).path
        try:
            if path == "/":
                self.file(ROOT / "templates" / self.page, "text/html; charset=utf-8")
            elif path == "/static/style.css":
                self.file(ROOT / "static" / "style.css", "text/css; charset=utf-8")
            else:
                self.get_api(path)
        except ValueError as error:
            self.json({"error": str(error)}, 400)
        except FileNotFoundError:
            self.json({"error": "ファイルが見つかりません。"}, 404)

    def do_POST(self):
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_REQUEST_BYTES:
                raise ValueError("リクエストが空か、大きすぎます（上限5MB）。")
            body = self.rfile.read(size)
            self.post_api(urlsplit(self.path).path, body)
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as error:
            self.json({"error": str(error)}, 400)
        except FileNotFoundError:
            self.json({"error": "ファイルが見つかりません。"}, 404)

    def get_api(self, path: str):
        self.json({"error": "ページが見つかりません。"}, 404)

    def post_api(self, path: str, body: bytes):
        self.json({"error": "ページが見つかりません。"}, 404)

    def json(self, value: object, status: int = 200):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def file(self, path: Path, content_type: str | None = None, attachment: str | None = None):
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        if attachment:
            self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def parse_json(body: bytes) -> dict:
        value = json.loads(body.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSONの最上位はオブジェクトにしてください。")
        return value

    def upload_json(self, body: bytes) -> dict:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data;"):
            raise ValueError("JSONファイルを選択してください。")
        message = BytesParser(policy=default).parsebytes(
            b"Content-Type: " + content_type.encode("ascii") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
        for part in message.iter_parts():
            if part.get_param("name", header="content-disposition") == "file":
                return self.parse_json(part.get_payload(decode=True))
        raise ValueError("JSONファイルを選択してください。")


def serve(handler: type[Handler], port: int):
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        print(f"http://127.0.0.1:{port}")
        server.serve_forever()
