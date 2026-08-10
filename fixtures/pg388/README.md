# PG-388 logic lab

This is a dynamic, in-memory business-logic state simulator for the `/pg388`
frontend. It exposes abstract case/role/feedback enums and returns bounded
Rule-IR projections. It does not implement real accounts, payments, cookies,
verification codes, credentials, external callbacks, persistent writes, or
arbitrary request values.

Run only after an operator has reviewed an immutable Python base digest:

```powershell
docker build --network=none --pull=false --build-arg PYTHON_IMAGE_DIGEST=<reviewed-digest> -f fixtures/pg388/Dockerfile .
docker run --rm --network=none --read-only --tmpfs /tmp:rw,noexec,nosuid,size=16m -p 127.0.0.1:8088:8088 <reviewed-image>
```

The container is disposable and the service binds loopback only. The model
boundary remains abstract (`safe_to_send=false`); typed fields and evidence
hashes are evaluator-side display data.

For the two-container display (frontend proxy + private backend), use
`docker-compose.pg388.yml` with both immutable base-image variables set. The
compose network is Docker `internal: true` and publishes only the frontend to
`127.0.0.1:3000`; it is a display deployment, not a replacement for the
network-none evaluator contract.

Implementation B is an optional source/implementation holdout. It uses a
separate transition table and `fixtures/pg388/Dockerfile.b`; enable it only
with `docker compose --profile holdout -f docker-compose.pg388.yml up` after
reviewing `PG388_PYTHON_IMAGE_DIGEST_B`. It exposes only the private `8089`
container port, has no persistent storage, and must remain candidate-only
until its own fresh reset, typed candidate/reference/negative/replay evidence
and source-row audit are complete.

The frozen core catalog contains 56 abstract contracts.  The separate
`/api/supplemental-cases` endpoint exposes 10 candidate-only taxonomy-gap
contracts (OAuth/activation/CSRF second-factor, CAPTCHA detail cases, and
Session guessing/forgery/leakage).  They accept the same enum-only episode
contract and never turn a raw value into a request.  The frontend catalog
shows all 24 presentation cases while preserving `training_eligible=false`
and all promotion flags false.

The local canary endpoint `/api/canary` is a bounded evaluator demonstration
for 28 abstract state machines covering installation, purchase boundaries,
including a transaction-concurrency/version-binding canary,
identity/recovery, 2FA (including OAuth/activation/CSRF order), CAPTCHA,
sessions (including entropy/integrity/exposure), authorization, identifier entropy,
execution order, and response projection. It accepts only `case_ref`, `role`,
and `phase` enums and emits state buckets/deltas; it does not accept
identifiers, prices, coupons, tokens, credentials, or arbitrary request values.
A `vulnerable_effect=true` result is only a typed result from this disposable
simulator, not a claim about any real application. Reset before each sequence
and keep `safe_to_send=false`.
