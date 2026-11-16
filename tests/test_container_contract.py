"""Native packaging/health-command checks; these do not claim a Docker build."""
from __future__ import annotations

import http.server
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import unittest


ROOT = Path(__file__).resolve().parents[1]


def docker_lines():
    return [line.strip() for line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def health_command():
    line = next(line for line in docker_lines() if line.startswith("HEALTHCHECK "))
    return json.loads(line.split(" CMD ", 1)[1])


class ContainerContract(unittest.TestCase):
    def test_copy_inputs_exclude_secrets_and_corpus_artifacts(self):
        allowed = {"requirements.txt", "*.py", "static"}
        inputs = set()
        for line in docker_lines():
            if line.startswith("COPY "):
                words = [word for word in shlex.split(line[5:]) if not word.startswith("--")]
                inputs.update(words[:-1])
        self.assertEqual(inputs, allowed)
        rules = [line.strip() for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        self.assertEqual(rules[0], "*")
        self.assertEqual({rule for rule in rules if rule.startswith("!")},
                         {"!*.py", "!requirements.txt", "!static/", "!static/**"})
        self.assertIn(".env", rules)
        self.assertIn(".env.*", rules)

    def test_compose_requires_existing_read_only_archive_and_loopback(self):
        compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
        for line in ("type: bind", "source: ./knowledge-base.zip", "target: /data/knowledge-base.zip",
                     "read_only: true", "create_host_path: false", "env_file: .env"):
            self.assertIn(line, compose)
        self.assertIn('"127.0.0.1:8000:8000"', compose)
        self.assertNotIn('"8000:8000"', compose)
        self.assertNotIn("0.0.0.0:8000", compose)
        users = [line.split(None, 1)[1] for line in docker_lines() if line.startswith("USER ")]
        self.assertEqual(users[-1], "10001:10001")

    def check_health(self, status: int):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.server.paths.append(self.path)
                self.send_response(status)
                self.end_headers()
                self.wfile.write(b"fixture")

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.paths = []
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            command = health_command()
            self.assertEqual(command[:2], ["python", "-c"])
            self.assertIn("http://127.0.0.1:8000/", command[2])
            code = command[2].replace("http://127.0.0.1:8000/", f"http://127.0.0.1:{server.server_port}/")
            # Run the actual health code against a local HTTP fixture, never the API provider.
            env = {**os.environ, "NO_PROXY": "127.0.0.1,localhost"}
            result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                                    text=True, timeout=10)
            self.assertEqual(server.paths, ["/"])
            return result.returncode
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_health_succeeds_on_http_200_without_model_calls(self):
        self.assertEqual(self.check_health(200), 0)

    def test_health_fails_on_http_503(self):
        self.assertNotEqual(self.check_health(503), 0)


if __name__ == "__main__":
    unittest.main()
