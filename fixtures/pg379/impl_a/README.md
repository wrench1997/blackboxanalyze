# PG-379 independent implementation A

This directory is a disposable, dependency-free Python WSGI implementation
for the PG-379 source/implementation-disjoint collection plan. It has six
abstract GET route classes and six abstract POST route classes. Responses are
bounded shape projections: input is reduced to presence, length bucket, and a
safe `reflected_bounded`/`filtered_bounded` differential. The original probe is
never logged, persisted, or returned; there is no business-data write and no
external network call.

The route table and `manifest_v1.json` carry a source SHA-256 and a distinct
route hash for every route. `manifest_v1.json` is fixture metadata, not PG-331
source rows. It remains `fixture_only_live_unbound`, with training and all
promotion flags false.

## Static deterministic checks

From the repository root:

```text
python -m pytest -q tests/test_pg379_impl_a_contract.py
python fixtures/pg379/impl_a/build_manifest.py --output fixtures/pg379/impl_a/manifest_v1.json
```

The manifest builder hashes only local `app.py` bytes and writes canonical
JSON. It does not start a server, Docker, GPU, or network operation.

## Optional disposable image (not run by the contract)

The Dockerfile requires an operator-reviewed Python base-image digest; no
default digest is provided. A future, explicitly authorised build must supply
`PYTHON_IMAGE_DIGEST=sha256:<reviewed-digest>`, use `--network none`, no bind or
volume mounts, and a disposable fresh container. The resulting image digest,
runtime/process boundary, source digest, and authorization must be added to a
separate evaluator-side attestation before any live gate can be reconsidered.
The current PG-379 collector intentionally has no code path that performs this
binding or starts the image.

```text
docker build --network=none --build-arg PYTHON_IMAGE_DIGEST=sha256:<reviewed-digest> -t pg379-impl-a:reviewed fixtures/pg379/impl_a
docker run --rm --network none --read-only --tmpfs /tmp:rw,noexec,nosuid,size=16m pg379-impl-a:reviewed
```

These commands are documentation only; CI/static tests must not invoke them.
