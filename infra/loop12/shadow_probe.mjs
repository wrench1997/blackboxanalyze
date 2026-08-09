import http from "node:http";

const path = process.argv[2];
if (!path || !path.startsWith("/") || path.startsWith("//")) {
  process.stdout.write(JSON.stringify({ error: "origin-relative path required" }));
  process.exit(2);
}

const request = http.request({ host: "127.0.0.1", port: 3000, method: "GET", path }, (response) => {
  let bodyLength = 0;
  let preview = "";
  let finished = false;
  const headers = Object.fromEntries(Object.entries(response.headers).filter(([key]) => ["content-type", "content-length", "location"].includes(key)));
  const contentType = String(response.headers["content-type"] ?? "").toLowerCase();
  const declaredLength = Number.parseInt(String(response.headers["content-length"] ?? ""), 10);

  const emit = (transportError = null) => {
    if (finished) return;
    finished = true;
    const lowered = preview.toLowerCase();
    // Prefer bytes actually observed.  Juice Shop's static fallback can
    // advertise a stale Content-Length and close early; that is still a
    // useful response-shape observation, not a reason to discard the probe.
    const effectiveLength = bodyLength > 0 ? bodyLength : (Number.isFinite(declaredLength) ? declaredLength : 0);
    let bodyShape = "unknown";
    if (contentType.includes("text/plain; version=0.0.4") || lowered.includes("# help ") || lowered.includes("# type ")) bodyShape = "prometheus";
    else if (lowered.includes("listing directory") || (lowered.includes("<pre") && lowered.includes("href=") && effectiveLength >= 11000)) bodyShape = "directory_listing";
    else if (lowered.includes("sourcemappingurl") || lowered.includes('"sources"') || lowered.includes('"mappings"')) bodyShape = "source_map";
    else if (lowered.includes("stack trace") || lowered.includes("error:    at ") || lowered.includes("traceback")) bodyShape = "traceback";
    else if (contentType.includes("json") && ["debug", "diagnostic", '"stack"', '"error"'].some((token) => lowered.includes(token))) bodyShape = "diagnostic";
    else if (lowered.includes("user-agent:") || lowered.includes("disallow:")) bodyShape = "robots_text";
    else if (lowered.includes("security.txt") || lowered.includes("contact:") || lowered.includes("policy")) bodyShape = "security_policy_text";
    else if (contentType.includes("html") && (lowered.includes("<app-root") || lowered.includes("owasp juice shop") || (effectiveLength >= 8000 && effectiveLength <= 20000))) bodyShape = "spa_shell";
    else if (contentType.includes("json")) bodyShape = effectiveLength < 4096 ? "ordinary_json" : "diagnostic";
    const payload = {
      status_code: response.statusCode ?? 0,
      headers,
      body_length: bodyLength,
      body_shape: bodyShape,
    };
    if (transportError) payload.transport_error = transportError;
    process.stdout.write(JSON.stringify(payload));
  };

  response.on("data", (chunk) => {
    bodyLength += chunk.length;
    if (preview.length < 8192) preview += chunk.toString("utf8").slice(0, 8192 - preview.length);
  });
  response.on("end", () => emit());
  // Some static responses in the pinned image close before their declared
  // length.  Capture the bounded partial projection instead of exiting with
  // an unhandled response error and losing the whole matrix row.
  response.on("aborted", () => emit("response_aborted"));
  response.on("error", (error) => emit(`response_error:${String(error)}`));
  response.on("close", () => emit("response_closed_before_end"));
});
request.setTimeout(15000, () => {
  request.destroy(new Error("request_timeout"));
});
request.on("error", (error) => {
  process.stdout.write(JSON.stringify({ error: String(error) }));
  process.exitCode = 1;
});
request.end();
