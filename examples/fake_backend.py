"""A tiny fake OpenAI-compatible endpoint for the lexvault quickstart.

Echoes the last user message back as a non-streaming chat completion, so you
can SEE that lexvault masked the codename to a ``[LEX-...]`` placeholder on the
way UP, and that the client sees the original RESTORED on the way back down.

Intentionally dependency-free (stdlib only) so it builds in seconds and has no
network egress. This replaces the non-existent ``ghcr.io/berriai/fake-openai-endpoint``
image that previously made the Docker quickstart fail to boot (CS-D2).

Run directly::

    python3 examples/fake_backend.py    # listens on :8000

Or via the Dockerfile.fake-backend in docker-compose.yml.
"""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# A [LEX-...] placeholder (lexvault's default namespace) for the echo display.
_PLACEHOLDER = re.compile(r"\[LEX-[A-Z2-7]+\](?:-\d+)?")


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - stdlib requires this name
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = {}

        # Last user message, or the raw prompt.
        messages = body.get("messages") or []
        user_msg = ""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                user_msg = str(m.get("content", ""))
                break

        # The upstream "model" sees whatever lexvault masked — surface the
        # placeholder count so the demo makes the mask visible.
        placeholders = _PLACEHOLDER.findall(user_msg)
        echo = (
            f"echo: {user_msg}"
            if not placeholders
            else f"echo: {user_msg}  [upstream saw {len(placeholders)} placeholder(s)]"
        )

        # Non-streaming chat completion. (Streaming is covered by the unit +
        # integration suites; the quickstart only needs the round-trip here.)
        resp = {
            "id": "chatcmpl-fake-lexvault",
            "object": "chat.completion",
            "model": body.get("model", "fake"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": echo},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        payload = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: A003 - stdlib name
        # Quieter logs; the echo itself is the signal.
        print(f"[fake-backend] {fmt % args}", flush=True)


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", 8000), _FakeOpenAIHandler)
    print("[fake-backend] listening on :8000 (echoes chat completions)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
