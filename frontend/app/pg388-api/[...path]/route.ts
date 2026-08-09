import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const ALLOWED_PATHS = new Set([
  "health",
  "api/manifest",
  "api/cases",
  "api/supplemental-cases",
  "api/reset",
  "api/observe",
  "api/episode",
  "api/canary",
]);
const ALLOWED_HOSTS = new Set(["127.0.0.1", "localhost", "::1", "pg388-backend"]);
const MAX_REQUEST_BYTES = 4096;
const MAX_RESPONSE_BYTES = 64 * 1024;

function backendBaseUrl(): URL | null {
  const configured = process.env.PG388_API_URL || "http://127.0.0.1:8088";
  try {
    const parsed = new URL(configured);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return null;
    if (!ALLOWED_HOSTS.has(parsed.hostname)) return null;
    parsed.pathname = parsed.pathname.replace(/\/+$/, "");
    parsed.search = "";
    parsed.hash = "";
    return parsed;
  } catch {
    return null;
  }
}

async function resolvePath(context: { params: Promise<{ path?: string[] }> | { path?: string[] } }) {
  const params = await context.params;
  const path = (params.path || []).join("/");
  return ALLOWED_PATHS.has(path) ? path : null;
}

async function proxy(request: Request, context: { params: Promise<{ path?: string[] }> | { path?: string[] } }) {
  const path = await resolvePath(context);
  const base = backendBaseUrl();
  if (!path || !base) {
    return NextResponse.json({ status: "blocked_backend_route" }, { status: 404 });
  }

  const target = new URL(`${base.toString().replace(/\/$/, "")}/${path}`);
  const headers = new Headers({ accept: "application/json" });
  let body: ArrayBuffer | undefined;
  if (request.method !== "GET" && request.method !== "HEAD") {
    body = await request.arrayBuffer();
    if (body.byteLength > MAX_REQUEST_BYTES) {
      return NextResponse.json({ status: "request_too_large" }, { status: 413 });
    }
    const contentType = request.headers.get("content-type");
    if (contentType && (contentType.startsWith("application/json") || contentType.startsWith("application/x-www-form-urlencoded"))) {
      headers.set("content-type", contentType);
    }
  }

  let response: Response;
  try {
    response = await fetch(target, {
      method: request.method,
      headers,
      body,
      cache: "no-store",
    });
  } catch {
    return NextResponse.json({ status: "backend_fetch_unavailable" }, { status: 502 });
  }
  try {
    const bytes = await response.arrayBuffer();
    if (bytes.byteLength > MAX_RESPONSE_BYTES) {
      return NextResponse.json({ status: "response_too_large" }, { status: 502 });
    }
    const responseHeaders = new Headers();
    const contentType = response.headers.get("content-type");
    if (contentType) responseHeaders.set("content-type", contentType);
    return new NextResponse(bytes, { status: response.status, headers: responseHeaders });
  } catch {
    return NextResponse.json({ status: "backend_response_unavailable" }, { status: 502 });
  }
}

export async function GET(request: Request, context: { params: Promise<{ path?: string[] }> | { path?: string[] } }) {
  return proxy(request, context);
}

export async function POST(request: Request, context: { params: Promise<{ path?: string[] }> | { path?: string[] } }) {
  return proxy(request, context);
}
