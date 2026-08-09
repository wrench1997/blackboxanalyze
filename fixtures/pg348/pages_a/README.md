# PG-348 pages A

This directory is a deterministic, static localhost fixture.  It crosses ten
visibly different HTML layouts with twelve safe surface variants, yielding 120
challenge identities.  The pages contain only bounded sample labels and local
relative links/forms.  They do not contact a server, use credentials or
callbacks, execute a real probe, or write persistent state.  The one
`POST`-shaped variant has a `type=button` control and is explicitly marked as a
disabled static preview.

`manifest_v1.json` is generated from the rendered UTF-8 bytes.  It records the
template/variant identity, abstract transport and shape fields, content hash,
raw-source policy, and duplicate counts.  All promotion flags are false; this
fixture is evaluator-only and is not a vulnerability, payload, or model-memory
corpus.

Regenerate (or verify without writing):

```text
python generate_pages_a.py
python generate_pages_a.py --check
```

The generator accepts `--output-dir` for a disposable copy, and never writes
outside that directory.

