"""Disposable, network-none transport for the reviewed PG-332 DVWA lane.

The module is deliberately narrower than a crawler.  A fresh pinned DVWA
container is started with no published ports and no bind/volume/tmpfs mounts.
The only request path is a short-lived ``docker exec`` PHP bridge to the
container's own ``127.0.0.1:80``.  Response bytes are returned to the caller in
memory and are never written by this module.  The caller must still keep raw
values on the evaluator side and must bind a typed candidate/reference/negative
sidecar before a row can be considered anything more than diagnostic/ASK.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from scripts.plan_pg332_dvwa_source_rows import IMAGE


ROOT = Path(__file__).resolve().parents[1]
INTERNAL_PORT = 80
MAX_REQUEST_BODY = 1 * 1024 * 1024
MAX_RESPONSE_BODY = 2 * 1024 * 1024
ALLOWED_METHODS = frozenset({"GET", "POST"})


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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _database_health(name: str) -> bool:
    """Check only connectivity to the disposable target's own DB."""

    # These are the fixed credentials shipped by the pinned DVWA image.  They
    # remain inside the evaluator-side health check and are never projected to
    # the model or written to a dataset.
    code = "$db=@new mysqli('127.0.0.1','app','vulnerables','dvwa',3306); exit($db->connect_errno ? 1 : 0);"
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


# The PHP process is intentionally persistent so its cookie jar stays inside
# the evaluator process.  It accepts only origin-relative paths and bounded
# GET/POST requests.  It emits status/header shape plus an in-memory body;
# callers are responsible for applying the raw-context firewall.
_PHP_BRIDGE = r'''$line = fopen("php://stdin", "r");
$cookies = [];
while (($encoded = fgets($line)) !== false) {
    $input = json_decode(base64_decode(trim($encoded)), true);
    if (!is_array($input)) { echo json_encode(["error" => "bad_json"]) . "\n"; continue; }
    $method = strtoupper((string)(isset($input["method"]) ? $input["method"] : "GET"));
    $path = (string)(isset($input["path"]) ? $input["path"] : "/");
    if (!in_array($method, ["GET", "POST"], true) || $path === "" || $path[0] !== "/" || strpos($path, "//") === 0 || strpos($path, "://") !== false || strlen($path) > 8192) {
        echo json_encode(["error" => "unsafe_request"]) . "\n"; fflush(STDOUT); continue;
    }
    $headers = [];
    foreach ((array)(isset($input["headers"]) ? $input["headers"] : []) as $key => $value) {
        $key = strtolower((string)$key);
        if (in_array($key, ["accept", "content-type", "user-agent"], true)) {
            $headers[] = $key . ": " . (string)$value;
        }
    }
    if (count($cookies) > 0) { $headers[] = "Cookie: " . implode("; ", $cookies); }
    $body = base64_decode((string)(isset($input["body"]) ? $input["body"] : ""), true);
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
    $http_response_header = [];
    $response = @file_get_contents("http://127.0.0.1:80" . $path, false, $context);
    $responseHeaders = [];
    $status = 0;
    foreach ((array)(isset($http_response_header) ? $http_response_header : []) as $header) {
        if (preg_match('/^HTTP\/[^ ]+ ([0-9]{3})/', (string)$header, $match)) { $status = (int)$match[1]; }
        elseif (strpos((string)$header, ":") !== false) {
            list($key, $value) = explode(":", (string)$header, 2);
            $key = strtolower(trim($key)); $value = trim($value);
            $responseHeaders[$key] = $value;
            if ($key === "set-cookie" && preg_match('/^([^=;]+=[^;]*)/', $value, $cookie)) {
                $cookieName = explode("=", $cookie[1], 2)[0];
                $cookies = array_values(array_filter($cookies, function($item) use ($cookieName) { return strpos($item, $cookieName . "=") !== 0; }));
                $cookies[] = $cookie[1];
            }
        }
    }
    if ($response === false && $status === 0) { echo json_encode(["error" => "target_request_failed"]) . "\n"; fflush(STDOUT); continue; }
    if ($response === false) { $response = ""; }
    if (strlen($response) > 2097152) { $response = substr($response, 0, 2097152); }
    echo json_encode(["status" => $status, "headers" => $responseHeaders, "body" => base64_encode($response)], JSON_UNESCAPED_SLASHES) . "\n";
    fflush(STDOUT);
}
'''


class PhpDockerBridge:
    """Serialize bounded requests through ``docker exec`` and PHP."""

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
        method = str(method).upper()
        if method not in ALLOWED_METHODS:
            raise ValueError("PG-332 relay only permits GET/POST")
        if len(body) > MAX_REQUEST_BODY:
            raise ValueError("PG-332 request body is too large")
        path = str(path)
        if not path.startswith("/") or path.startswith("//") or "://" in path or len(path) > 8192:
            raise ValueError("PG-332 relay requires an origin-relative path")
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("PG-332 PHP bridge pipes unavailable")
        payload = {
            "method": method,
            "path": path,
            "headers": {str(k).casefold(): str(v) for k, v in dict(headers or {}).items() if str(k).casefold() in {"accept", "content-type", "user-agent"}},
            "body": base64.b64encode(body).decode("ascii") if body else "",
        }
        encoded = base64.b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("ascii")
        with self._lock:
            if self.process.poll() is not None:
                raise RuntimeError("PG-332 PHP bridge exited")
            self.process.stdin.write(encoded + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("PG-332 PHP bridge returned no response")
        value = json.loads(line)
        if not isinstance(value, Mapping) or value.get("error"):
            raise RuntimeError("PG-332 target request failed")
        try:
            body_value = base64.b64decode(str(value.get("body", "")), validate=True)
            status = int(value.get("status", 0) or 0)
        except (TypeError, ValueError) as error:
            raise RuntimeError("PG-332 PHP bridge returned invalid response") from error
        if len(body_value) > MAX_RESPONSE_BODY:
            body_value = body_value[:MAX_RESPONSE_BODY]
        return {"status": status, "headers": {str(k).casefold(): str(v) for k, v in dict(value.get("headers") or {}).items()}, "body": body_value}

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                if self.process.stdin is not None:
                    self.process.stdin.close()
            except OSError:
                pass
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    self.process.kill()
                except OSError:
                    pass


def _attest(name: str, container_id: str) -> dict[str, Any]:
    mounts = json.loads(_docker("inspect", "--format", "{{json .Mounts}}", name) or "[]")
    ports = json.loads(_docker("inspect", "--format", "{{json .NetworkSettings.Ports}}", name) or "{}")
    image_ref = _docker("inspect", "--format", "{{.Config.Image}}", name)
    network_mode = _docker("inspect", "--format", "{{.HostConfig.NetworkMode}}", name)
    if image_ref != IMAGE or network_mode != "none" or mounts or ports not in ({}, None):
        raise RuntimeError("PG-332 target attestation failed")
    return {
        "reset_id": f"pg332-reset-{_sha256_text(container_id)[:16]}",
        "fresh_reset": True,
        "target_instance_digest": _sha256_text(container_id),
        "network_mode": "none",
        "external_network": False,
        "loopback_only": True,
        "state_clean": True,
        "database_health_gate": "database_health_ok",
        "database_clean_attestation": True,
        "container_name": name,
        "image_digest": IMAGE.split("@sha256:", 1)[1],
        "published_port_count": 0,
        "bind_or_volume_mount_count": 0,
        "teardown_required": True,
    }


class DisposableDvwa:
    """One fresh role target with an evaluator-side cookie/session jar."""

    def __init__(self, *, name: str, seed: int, index: int, command: tuple[str, ...]) -> None:
        self.name = str(name)
        self.seed = int(seed)
        self.index = int(index)
        self.command = tuple(command)
        self.container_id = ""
        self.bridge: PhpDockerBridge | None = None
        self.reset: dict[str, Any] = {}

    def start(self, *, timeout: float = 150.0) -> dict[str, Any]:
        if _exists(self.name):
            raise RuntimeError("PG-332 refuses target reuse")
        if self.command[-1] != IMAGE or "--network" not in self.command or self.command[self.command.index("--network") + 1] != "none":
            raise RuntimeError("PG-332 command is not fixed network-none")
        _docker(*self.command[1:])
        self.bridge = PhpDockerBridge(self.name)
        deadline = time.monotonic() + float(timeout)
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                response = self.bridge.request("GET", "/login.php")
                if int(response.get("status", 0)) == 200 and _database_health(self.name):
                    self.container_id = _docker("inspect", "--format", "{{.Id}}", self.name)
                    self.reset = _attest(self.name, self.container_id)
                    self._initialize_database()
                    # Setup may rotate the session token; always fetch a
                    # fresh login form before posting credentials.
                    login_page = self.bridge.request("GET", "/login.php")
                    self._login(login_page["body"])
                    self._assert_authenticated()
                    self.reset.update({
                        "application_database_initialized": True,
                        "application_health_gate": "authenticated",
                    })
                    return dict(self.reset)
            except Exception as error:
                last_error = error
            time.sleep(1.0)
        self.stop()
        raise RuntimeError(f"PG-332 DVWA target failed health gate: {type(last_error).__name__ if last_error else 'unknown'}")

    def _initialize_database(self) -> None:
        """Initialize only the fresh container's disposable application DB.

        The pinned DVWA image starts with a reachable MariaDB but without the
        DVWA schema.  Setup is therefore an evaluator-side readiness step,
        not a model action or a training label.  A fresh container is required
        before this method is called, so the database write cannot escape the
        disposable target.
        """

        if self.bridge is None:
            raise RuntimeError("PG-332 bridge is not started")
        setup = self.bridge.request("GET", "/setup.php")
        body = bytes(setup.get("body") or b"")
        token_match = re.search(rb'name=[\"\']user_token[\"\'] value=[\"\']([a-f0-9]+)', body)
        if token_match is None or b"create_db" not in body:
            # A pre-initialized image is acceptable only if its authenticated
            # health gate below succeeds; no setup request is guessed.
            return
        token = token_match.group(1).decode("ascii")
        form = urlencode({
            "create_db": "Create / Reset Database",
            "user_token": token,
        }).encode("ascii")
        result = self.bridge.request(
            "POST",
            "/setup.php",
            body=form,
            headers={"content-type": "application/x-www-form-urlencoded"},
        )
        if int(result.get("status", 0)) not in {200, 302, 303}:
            raise RuntimeError("PG-332 DVWA database initialization failed")

    def _assert_authenticated(self) -> None:
        """Require a real session before allowing any allowlisted route."""

        if self.bridge is None:
            raise RuntimeError("PG-332 bridge is not started")
        response = self.bridge.request("GET", "/")
        body = bytes(response.get("body") or b"").lower()
        login_form = b"name=\"username\"" in body or b"name='username'" in body
        logout_marker = b"logout" in body
        if int(response.get("status", 0)) != 200 or login_form or not logout_marker:
            raise RuntimeError("PG-332 DVWA authentication health gate failed")

    def _login(self, body: bytes) -> None:
        if self.bridge is None:
            raise RuntimeError("PG-332 bridge is not started")
        token_match = re.search(rb"name=['\"]user_token['\"] value=['\"]([a-f0-9]+)['\"]", body)
        token = token_match.group(1).decode("ascii") if token_match else ""
        form = f"username=admin&password=password&user_token={token}&Login=Login".encode("ascii")
        response = self.bridge.request("POST", "/login.php", body=form, headers={"content-type": "application/x-www-form-urlencoded"})
        if int(response.get("status", 0)) not in {200, 302, 303}:
            raise RuntimeError("PG-332 DVWA login failed")

    def request(self, method: str, path: str, *, body: bytes = b"", content_type: str = "application/x-www-form-urlencoded") -> dict[str, Any]:
        if self.bridge is None:
            raise RuntimeError("PG-332 bridge is not started")
        return self.bridge.request(method, path, body=body, headers={"content-type": content_type, "accept": "text/html"})

    def stop(self) -> None:
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None
        try:
            if _exists(self.name):
                _docker("stop", "--timeout", "10", self.name)
        except (OSError, subprocess.SubprocessError):
            pass


__all__ = ["DisposableDvwa", "IMAGE", "PhpDockerBridge", "_attest"]
