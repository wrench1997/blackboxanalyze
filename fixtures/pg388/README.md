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

The frozen core catalog contains 56 abstract contracts.  The separate
`/api/supplemental-cases` endpoint exposes 10 candidate-only taxonomy-gap
contracts (OAuth/activation/CSRF second-factor, CAPTCHA detail cases, and
Session guessing/forgery/leakage).  They accept the same enum-only episode
contract and never turn a raw value into a request.  The frontend catalog
shows all 24 presentation cases while preserving `training_eligible=false`
and all promotion flags false.
