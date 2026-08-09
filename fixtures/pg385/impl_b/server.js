#!/usr/bin/env node
"use strict";

// Independent PG-385 local filter fixture B.  It is intentionally tiny and
// disposable: no files, database, credentials, external network, or business
// state.  Only bounded abstract projections leave this process.

const http = require("node:http");
const { URL } = require("node:url");
const crypto = require("node:crypto");

const HOST = "127.0.0.1";
const PORT = Number.parseInt(process.env.PORT || "0", 10);
const ROUTE = "/pg385b/filter";
const FIELD = "value";
const IMPLEMENTATION_ID = "pg385_filter_impl_b_node";
const SCHEMA_VERSION = "pg385-filter-node-fixture-v1";
let resetGeneration = 0;

function digest(value) {
  return crypto.createHash("sha256").update(JSON.stringify(value), "utf8").digest("hex");
}

function projection(fields) {
  const result = {
    schema_version: SCHEMA_VERSION,
    status_class: fields.status_class,
    response_shape: "bounded_json_projection",
    filter_state: fields.filter_state,
    filter_class: fields.filter_class,
    failure_shape: fields.failure_shape,
    effect_class: fields.effect_class,
    typed_effect_confirmed: Boolean(fields.typed_effect_confirmed),
    encoding_acceptance: fields.encoding_acceptance,
    implementation_id: IMPLEMENTATION_ID,
    loopback_only: true,
    external_network: false,
    raw_response_stored: false,
  };
  result.evidence_sha256 = digest(result);
  return result;
}

function classify(raw) {
  if (typeof raw !== "string" || raw.length === 0) {
    return projection({ status_class: "4xx", filter_state: "parser_error", filter_class: "missing_parameter", failure_shape: "field_not_observed", effect_class: "none", typed_effect_confirmed: false, encoding_acceptance: "not_observed" });
  }
  // B uses a different implementation and field name, but exposes the same
  // abstract canonicalization feedback contract.
  if (raw.includes(":") || (/%3A/i.test(raw) && !/%25/i.test(raw))) {
    return projection({ status_class: "4xx", filter_state: "filtered", filter_class: "encoding_filter", failure_shape: "raw_delimiter_blocked", effect_class: "none", typed_effect_confirmed: false, encoding_acceptance: "encoded_variant_required" });
  }
  let decoded = raw;
  try {
    for (let index = 0; index < 3; index += 1) {
      const next = decodeURIComponent(decoded.replace(/\+/g, " "));
      if (next === decoded) break;
      decoded = next;
    }
  } catch (_error) {
    return projection({ status_class: "4xx", filter_state: "parser_error", filter_class: "encoding_parse_error", failure_shape: "encoding_parse_error", effect_class: "none", typed_effect_confirmed: false, encoding_acceptance: "parser_error" });
  }
  if (!decoded.includes(":")) {
    return projection({ status_class: "2xx", filter_state: "no_effect", filter_class: "none", failure_shape: "marker_not_reached", effect_class: "none", typed_effect_confirmed: false, encoding_acceptance: "no_delimiter" });
  }
  if (decoded.includes("_NEG_")) {
    return projection({ status_class: "2xx", filter_state: "no_effect", filter_class: "matched_negative", failure_shape: "negative_control", effect_class: "none", typed_effect_confirmed: false, encoding_acceptance: "encoded_variant" });
  }
  return projection({ status_class: "2xx", filter_state: "typed_effect", filter_class: "none", failure_shape: "none", effect_class: "bounded_marker_reflection", typed_effect_confirmed: true, encoding_acceptance: "encoded_variant" });
}

function bodyProjection(res, value) {
  const body = Buffer.from(JSON.stringify(value), "utf8");
  res.writeHead(value.status_class === "4xx" ? 400 : 200, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": body.length,
    "Cache-Control": "no-store",
  });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let chunks = [];
    let size = 0;
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size <= 8192) chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", () => reject(new Error("body_error")));
  });
}

function parseField(raw, method, contentType) {
  if (method === "GET") {
    try { return new URLSearchParams(raw).get(FIELD) || ""; } catch (_error) { return ""; }
  }
  if (contentType.includes("application/x-www-form-urlencoded")) {
    try { return new URLSearchParams(raw).get(FIELD) || ""; } catch (_error) { return ""; }
  }
  return "";
}

function handler(req, res) {
  let parsed;
  try { parsed = new URL(req.url || "/", `http://${HOST}`); } catch (_error) { bodyProjection(res, projection({ status_class: "4xx", filter_state: "parser_error", filter_class: "invalid_request", failure_shape: "parse_error", effect_class: "none", typed_effect_confirmed: false, encoding_acceptance: "parser_error" })); return; }
  if (parsed.pathname === "/__health" && req.method === "GET") {
    bodyProjection(res, { schema_version: SCHEMA_VERSION, implementation_id: IMPLEMENTATION_ID, loopback_only: true, external_network: false, fresh_reset: true, reset_generation: resetGeneration, target_instance_digest: digest({ implementation: IMPLEMENTATION_ID, reset: resetGeneration }) });
    return;
  }
  if (parsed.pathname === "/__reset" && req.method === "POST") {
    resetGeneration += 1;
    bodyProjection(res, { schema_version: SCHEMA_VERSION, fresh_reset: true, state_clean: true, reset_generation: resetGeneration, target_instance_digest: digest({ implementation: IMPLEMENTATION_ID, reset: resetGeneration }) });
    return;
  }
  if (parsed.pathname !== ROUTE || !["GET", "POST"].includes(req.method)) { bodyProjection(res, { status_class: "4xx", response_shape: "route_not_found" }); return; }
  if (req.method === "GET") { bodyProjection(res, classify(parsed.search)); return; }
  readBody(req).then((body) => bodyProjection(res, classify(parseField(body, "POST", String(req.headers["content-type"] || ""))))).catch(() => bodyProjection(res, projection({ status_class: "4xx", filter_state: "parser_error", filter_class: "body_error", failure_shape: "body_error", effect_class: "none", typed_effect_confirmed: false, encoding_acceptance: "parser_error" })));
}

const server = http.createServer(handler);
server.listen(PORT, HOST, () => {
  const address = server.address();
  console.log(JSON.stringify({ event: "ready", schema_version: SCHEMA_VERSION, implementation_id: IMPLEMENTATION_ID, host: HOST, port: address && typeof address === "object" ? address.port : PORT, route: ROUTE, method: "GET+POST", external_network: false }));
});
function shutdown() { server.close(() => process.exit(0)); }
process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);
