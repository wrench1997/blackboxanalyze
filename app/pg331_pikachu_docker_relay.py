"""Loopback relay for a disposable, network-none Pikachu container.

The relay is intentionally narrower than a web crawler.  It starts one pinned
image, keeps the target on Docker's ``none`` network, and forwards only
origin-relative GET/POST requests through a PHP process running *inside* the
target namespace.  The host-side listener binds to ``127.0.0.1`` on an
ephemeral port.  Response bytes are returned to the caller in memory; this
module never writes a response body, payload, cookie, or URL to disk.

The image's fixed entrypoint needs a small capability allowlist to create its
MySQL log and PHP-FPM state.  We therefore drop all capabilities and add only
``DAC_OVERRIDE``, ``CHOWN``, ``FOWNER``, ``SETUID`` and ``SETGID``; the target
still uses ``no-new-privileges``, no host mounts and no external network.

This is transport infrastructure for PG-331A source-row collection.  It does
not decide whether a vulnerability exists and it does not provide a free-form
network primitive to the model.
"""

from __future__ import annotations

import base64
import json
import os
import socketserver
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
IMAGE = "sift/pikachu-fixed@sha256:cca4288b6b701725e7a771f47ce7fcafd6cea9bd7622fa34ef2ed0b440f472c6"
INTERNAL_PORT = 8090
MAX_REQUEST_BODY = 1 * 1024 * 1024
MAX_RESPONSE_BODY = 2 * 1024 * 1024


def _docker(*args: str, timeout: float = 60.0) -> str:
    result = subprocess.run(
        ["docker", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout.strip()


def _exists(name: str) -> bool:
    return bool(_docker("ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.Names}}"))


def _database_health(name: str) -> bool:
    """Check the pinned image's own database without retaining query data."""

    code = "$db=@new mysqli('127.0.0.1','root','root','pikachu',3306); exit($db->connect_errno ? 1 : 0);"
    try:
        result = subprocess.run(
            ["docker", "exec", name, "php", "-r", code],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


_PHP_BRIDGE = r'''$line = fopen("php://stdin", "r");
while (($encoded = fgets($line)) !== false) {
    $input = json_decode(base64_decode(trim($encoded)), true);
    if (!is_array($input)) { echo json_encode(["error" => "bad_json"]) . "\n"; continue; }
    $method = strtoupper((string)($input["method"] ?? "GET"));
    $path = (string)($input["path"] ?? "/");
    if ($path === "" || $path[0] !== "/" || strpos($path, "//") === 0 || strpos($path, "://") !== false || strlen($path) > 8192) {
        echo json_encode(["error" => "unsafe_path"]) . "\n"; continue;
    }
    $headers = [];
    foreach ((array)($input["headers"] ?? []) as $key => $value) {
        $key = strtolower((string)$key);
        if (in_array($key, ["accept", "content-type", "user-agent"], true)) {
            $headers[] = $key . ": " . (string)$value;
        }
    }
    $body = base64_decode((string)($input["body"] ?? ""), true);
    if ($body === false) { $body = ""; }
    $options = ["http" => [
        "method" => $method,
        "header" => implode("\r\n", $headers),
        "content" => $body,
        "ignore_errors" => true,
        "timeout" => 15,
        "protocol_version" => 1.1,
    ]];
    $context = stream_context_create($options);
    $response = @file_get_contents("http://127.0.0.1:8090" . $path, false, $context);
    $responseHeaders = [];
    $status = 0;
    foreach ((array)($http_response_header ?? []) as $header) {
        if (preg_match('/^HTTP\/[^ ]+ ([0-9]{3})/', (string)$header, $match)) { $status = (int)$match[1]; }
        elseif (strpos((string)$header, ":") !== false) {
            [$key, $value] = explode(":", (string)$header, 2);
            $responseHeaders[strtolower(trim($key))] = trim($value);
        }
    }
    if ($response === false && $status === 0) { echo json_encode(["error" => "target_request_failed"]) . "\n"; continue; }
    if ($response === false) { $response = ""; }
    if (strlen($response) > 2097152) { $response = substr($response, 0, 2097152); }
    echo json_encode(["status" => $status, "headers" => $responseHeaders, "body" => base64_encode($response)], JSON_UNESCAPED_SLASHES) . "\n";
    fflush(STDOUT);
}
'''


class PhpDockerBridge:
    """Serialize bounded HTTP requests through ``docker exec`` and PHP."""

    def __init__(self, container_name: str) -> None:
        self.container_name = str(container_name)
        self._lock = threading.Lock()
        self.process = subprocess.Popen(
            ["docker", "exec", "-i", self.container_name, "php", "-r", _PHP_BRIDGE],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def request(self, method: str, path: str, *, body: bytes = b"", headers: Mapping[str, str] | None = None) -> dict[str, Any]:
        if len(body) > MAX_REQUEST_BODY:
            raise ValueError("PG-331 relay request body is too large")
        method = str(method).upper()
        if method not in {"GET", "POST", "HEAD"}:
            raise ValueError("PG-331 relay only permits GET/POST/HEAD")
        if not str(path).startswith("/") or str(path).startswith("//") or "://" in str(path) or len(str(path)) > 8192:
            raise ValueError("PG-331 relay requires an origin-relative path")
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("PG-331 PHP bridge pipes unavailable")
        payload = {
            "method": method,
            "path": str(path),
            "headers": {str(k).casefold(): str(v) for k, v in dict(headers or {}).items() if str(k).casefold() in {"accept", "content-type", "user-agent"}},
            "body": base64.b64encode(body).decode("ascii") if body else "",
        }
        encoded = base64.b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("ascii")
        with self._lock:
            if self.process.poll() is not None:
                raise RuntimeError(f"PG-331 PHP bridge exited ({self.process.returncode})")
            self.process.stdin.write(encoded + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("PG-331 PHP bridge returned no response")
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise RuntimeError("PG-331 PHP bridge returned malformed JSON")
        if value.get("error"):
            raise RuntimeError(f"PG-331 target request failed: {value.get('error')}")
        result = dict(value)
        try:
            result["status"] = int(result.get("status", 0) or 0)
            result["body"] = base64.b64decode(str(result.get("body", "")), validate=True)
        except (TypeError, ValueError) as error:
            raise RuntimeError("PG-331 PHP bridge returned invalid response encoding") from error
        if len(result["body"]) > MAX_RESPONSE_BODY:
            result["body"] = result["body"][:MAX_RESPONSE_BODY]
        result["headers"] = {str(k).casefold(): str(v) for k, v in dict(result.get("headers") or {}).items()}
        return result

    def close(self) -> None:
        process = self.process
        if process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except OSError:
                pass
            try:
                process.terminate()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                except OSError:
                    pass


class _RelayHandler(BaseHTTPRequestHandler):
    server: "LoopbackRelayServer"

    def log_message(self, *_: Any) -> None:
        return

    def _forward(self) -> None:
        path = str(self.path)
        if not path.startswith("/") or path.startswith("//") or "://" in path or len(path) > 8192:
            self.send_error(400)
            return
        try:
            length = int(self.headers.get("content-length", "0") or 0)
        except ValueError:
            self.send_error(400)
            return
        if length < 0 or length > MAX_REQUEST_BODY:
            self.send_error(413)
            return
        body = self.rfile.read(length) if length else b""
        try:
            result = self.server.bridge.request(
                self.command,
                path,
                body=body,
                headers={"accept": self.headers.get("accept", "*/*"), "content-type": self.headers.get("content-type", ""), "user-agent": "pg331-loopback-relay/1"},
            )
        except Exception:
            self.send_error(502)
            return
        status = int(result.get("status", 0) or 0)
        if status <= 0:
            self.send_error(502)
            return
        response_body = bytes(result.get("body") or b"")
        response_headers = dict(result.get("headers") or {})
        self.send_response(status)
        for key in ("content-type", "location", "set-cookie", "cache-control", "etag"):
            if key in response_headers:
                location = str(response_headers[key])
                if key == "location" and "://" in location:
                    # The target image must not redirect the observer to an
                    # external origin.  Preserve only relative locations.
                    if not location.startswith("/"):
                        continue
                self.send_header(key, location)
        self.send_header("content-length", str(len(response_body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(response_body)

    do_GET = _forward
    do_POST = _forward
    do_HEAD = _forward


class LoopbackRelayServer(HTTPServer):
    """A loopback-only host listener forwarding to one target bridge."""

    allow_reuse_address = False

    def __init__(self, bridge: PhpDockerBridge) -> None:
        self.bridge = bridge
        super().__init__(("127.0.0.1", 0), _RelayHandler)


class LoopbackRelay:
    def __init__(self, bridge: PhpDockerBridge) -> None:
        self.httpd = LoopbackRelayServer(bridge)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


class DisposablePikachu:
    """Start and attest one fresh pinned target, then expose a relay."""

    def __init__(self, name: str, *, seed: int, index: int) -> None:
        self.name = str(name)
        self.seed = int(seed)
        self.index = int(index)
        self.container_id = ""
        self.bridge: PhpDockerBridge | None = None
        self.relay: LoopbackRelay | None = None
        self.reset: dict[str, Any] = {}

    def start(self, *, timeout: float = 150.0) -> dict[str, Any]:
        if _exists(self.name):
            raise RuntimeError(f"PG-331 refuses to reuse target {self.name}")
        _docker(
            "run", "--detach", "--rm", "--pull=never", "--name", self.name,
            "--label", "sift.pg331=true", "--label", f"sift.pg331.reset_epoch={self.seed}-{self.index}",
            "--network", "none", "--cap-drop", "ALL",
            "--cap-add", "DAC_OVERRIDE", "--cap-add", "CHOWN", "--cap-add", "FOWNER",
            "--cap-add", "SETUID", "--cap-add", "SETGID",
            "--security-opt", "no-new-privileges",
            "--pids-limit", "256", "--memory", "1g", "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m",
            IMAGE,
        )
        self.bridge = PhpDockerBridge(self.name)
        deadline = time.monotonic() + float(timeout)
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                response = self.bridge.request("GET", "/")
                status = int(response.get("status", 0) or 0)
                if 100 <= status < 600 and _database_health(self.name):
                    self.container_id = _docker("inspect", "--format", "{{.Id}}", self.name)
                    mounts = json.loads(_docker("inspect", "--format", "{{json .Mounts}}", self.name) or "[]")
                    image_ref = _docker("inspect", "--format", "{{.Config.Image}}", self.name)
                    network_mode = _docker("inspect", "--format", "{{.HostConfig.NetworkMode}}", self.name)
                    if image_ref != IMAGE or network_mode != "none" or mounts:
                        raise RuntimeError("PG-331 target attestation failed: image/network/mount")
                    self.relay = LoopbackRelay(self.bridge)
                    self.reset = {
                        "reset_id": f"pg331-reset-{self.seed}-{self.index}",
                        "fresh_reset": True,
                        "target_instance_digest": __import__("hashlib").sha256(self.container_id.encode("utf-8")).hexdigest(),
                        "network_mode": "none",
                        "external_network": False,
                        "loopback_only": True,
                        "state_clean": True,
                        "database_health_gate": "mysqli_root_pikachu_ok",
                    }
                    return dict(self.reset)
            except Exception as error:  # target startup is retried, not hidden
                last_error = error
            time.sleep(1.0)
        self.stop()
        raise RuntimeError(f"PG-331 target {self.name} did not become ready: {last_error}")

    @property
    def origin(self) -> str:
        if self.relay is None:
            raise RuntimeError("PG-331 relay is not started")
        return f"http://127.0.0.1:{self.relay.port}"

    def stop(self) -> None:
        if self.relay is not None:
            self.relay.close()
            self.relay = None
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None
        try:
            if _exists(self.name):
                _docker("stop", "--timeout", "5", self.name)
        except (subprocess.SubprocessError, OSError):
            pass


__all__ = ["DisposablePikachu", "IMAGE", "LoopbackRelay", "PhpDockerBridge"]
