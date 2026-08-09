export type DomOracleEvidence = {
  oracle: "controlled_detached_browser_dom_v1";
  sink: string;
  transforms: string[];
  source_to_sink: { source: string; transforms: string[]; sink: string };
  source_sha256: string;
  transformed_sha256: string;
  before_element_count: number;
  after_element_count: number;
  marker_hits: number;
  tag_shape: string[];
  script_like_present: boolean;
  browser_sink_observed: boolean;
  dom_change: boolean;
  candidate_signal: boolean;
  script_execution: false;
  network_access: false;
  navigation: false;
  detached: true;
  evidence_hash: string;
  evidence_hash_algorithm: "sha256-canonical-json" | "non-cryptographic-fallback";
};

const ALLOWED_SINKS = new Set(["innerHTML", "template.innerHTML"]);
const ALLOWED_TRANSFORMS = new Set(["identity", "html_entity_decode", "casefold"]);

function transformValue(input: string, transforms: string[]): string {
  return transforms.reduce((value, transform) => {
    if (!ALLOWED_TRANSFORMS.has(transform)) throw new Error(`unsupported DOM transform: ${transform}`);
    if (transform === "html_entity_decode") {
      const textarea = document.createElement("textarea");
      textarea.innerHTML = value;
      return textarea.value;
    }
    return transform === "casefold" ? value.toLocaleLowerCase() : value;
  }, input);
}

function tagsIn(root: ParentNode): string[] {
  return Array.from(root.querySelectorAll("*")).slice(0, 32).map((element) => element.tagName.toLowerCase());
}

function markerHits(root: ParentNode, marker: string): number {
  return Array.from(root.querySelectorAll("[data-sift-marker]"))
    .filter((element) => element.getAttribute("data-sift-marker") === marker).length;
}

function fallbackDigest(input: string): string {
  let hash = 2166136261;
  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return `fnv1a-${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

async function digest(input: string): Promise<{ value: string; algorithm: DomOracleEvidence["evidence_hash_algorithm"] }> {
  if (globalThis.crypto?.subtle) {
    const bytes = await globalThis.crypto.subtle.digest("SHA-256", new TextEncoder().encode(input));
    return {
      value: Array.from(new Uint8Array(bytes), (item) => item.toString(16).padStart(2, "0")).join(""),
      algorithm: "sha256-canonical-json",
    };
  }
  return { value: fallbackDigest(input), algorithm: "non-cryptographic-fallback" };
}

export async function runControlledDomOracle(
  input: string,
  options: { sink?: string; transforms?: string[]; marker?: string } = {},
): Promise<DomOracleEvidence> {
  const sink = options.sink ?? "innerHTML";
  const transforms = [...(options.transforms ?? [])];
  const marker = options.marker ?? "sift-marker";
  if (!ALLOWED_SINKS.has(sink)) throw new Error(`unsupported DOM sink: ${sink}`);
  const transformed = transformValue(input, transforms);

  // Never append this node to document.body.  Detached nodes do not navigate,
  // load external resources, or execute script elements in this oracle.
  const host = document.createElement("div");
  host.setAttribute("aria-hidden", "true");
  const beforeElementCount = host.childElementCount;
  let root: ParentNode = host;
  if (sink === "template.innerHTML") {
    const template = document.createElement("template");
    template.innerHTML = transformed;
    root = template.content;
  } else {
    host.innerHTML = transformed;
  }
  const tagShape = tagsIn(root);
  const afterElementCount = tagShape.length;
  const hits = markerHits(root, marker);
  const canonical = JSON.stringify({ sink, transforms, input, transformed, tagShape, hits });
  const sourceHash = await digest(input);
  const transformedHash = await digest(transformed);
  const evidenceHash = await digest(canonical);
  return {
    oracle: "controlled_detached_browser_dom_v1",
    sink,
    transforms,
    source_to_sink: { source: "untrusted_text", transforms, sink },
    source_sha256: sourceHash.value,
    transformed_sha256: transformedHash.value,
    before_element_count: beforeElementCount,
    after_element_count: afterElementCount,
    marker_hits: hits,
    tag_shape: tagShape,
    script_like_present: tagShape.some((tag) => ["script", "iframe", "object", "embed"].includes(tag)),
    browser_sink_observed: true,
    dom_change: afterElementCount !== beforeElementCount,
    candidate_signal: afterElementCount > 0 || hits > 0,
    script_execution: false,
    network_access: false,
    navigation: false,
    detached: true,
    evidence_hash: evidenceHash.value,
    evidence_hash_algorithm: evidenceHash.algorithm,
  };
}

