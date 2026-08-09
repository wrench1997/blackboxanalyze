import http from "node:http";
import { resolve4 } from "node:dns/promises";

const upstreamHost = "sift-loop12-juice-v20";
const upstreamPort = 3000;
const listenPort = 3000;
const maximumBodyBytes = 1_048_576;
const hopByHop = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
]);

function filteredHeaders(headers) {
  return Object.fromEntries(
    Object.entries(headers).filter(([name]) => !hopByHop.has(name.toLowerCase())),
  );
}

const server = http.createServer(async (request, response) => {
  if (!request.url?.startsWith("/")) {
    response.writeHead(400, { "content-type": "text/plain" });
    response.end("origin-form request target required");
    return;
  }

  let upstreamAddress;
  try {
    [upstreamAddress] = await resolve4(upstreamHost);
  } catch {
    response.writeHead(502, { "content-type": "text/plain" });
    response.end("local target unavailable");
    return;
  }

  let received = 0;
  const upstream = http.request({
    hostname: upstreamAddress,
    port: upstreamPort,
    method: request.method,
    path: request.url,
    headers: { ...filteredHeaders(request.headers), host: `${upstreamHost}:${upstreamPort}` },
  }, (upstreamResponse) => {
    response.writeHead(upstreamResponse.statusCode ?? 502, filteredHeaders(upstreamResponse.headers));
    upstreamResponse.pipe(response);
  });

  upstream.on("error", () => {
    if (!response.headersSent) response.writeHead(502, { "content-type": "text/plain" });
    response.end("local target unavailable");
  });

  request.on("data", (chunk) => {
    received += chunk.length;
    if (received > maximumBodyBytes) {
      upstream.destroy();
      if (!response.headersSent) response.writeHead(413, { "content-type": "text/plain" });
      response.end("request body too large");
      request.destroy();
    }
  });
  request.pipe(upstream);
});

server.listen(listenPort, "0.0.0.0");
