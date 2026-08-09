# Juice Shop External-Validity Loop 12 — Preregistration

This protocol was frozen before inspecting the locally deployed Juice Shop challenge catalog.

## Research question

Does the fixed 908,546-parameter Loop 11 system transfer from synthetic rule episodes to an unseen interactive web application, and which failure layer limits that transfer?

## Safety and environment

Only a version-pinned, disposable local Docker instance may be tested. It binds to `127.0.0.1` and runs on an internal Docker network without public ingress or container internet egress. Denial of service, resource exhaustion, command execution, persistent shells, external callbacks and destructive bulk deletion are excluded.

## Information boundary

The black-box agent may see only the application UI and the same-origin observations created by its own actions. Challenge titles, descriptions, hints, difficulty, solutions, solved state, source snippets, CWE labels, target family and evaluator Rule IR remain hidden until an episode ends.

## Split rule

The unit of separation is a complete normalized vulnerability family. Families are deterministically ordered by `SHA256(split seed + family)`: the first eligible family is training, the second validation, and every remaining family is hidden test. Requests or individual challenges from the same family may not cross splits.

## Frozen baseline

Before any Juice Shop training, four paths are evaluated under the same probe budget: random probes; frozen Loop 11 neural without Juice Shop memory; the same neural model with only pre-existing synthetic memory; and the separate C5 executable rule path. C5 is never counted as neural learning.

## Metrics and intervention gate

The experiment records solved-and-evidenced episode rate, probe cost, Counterexample Top-1, canonical Rule IR accuracy, negative-control false positives and clean-reset replay. Any later intervention must improve an untouched hidden family by at least 10 percentage points over frozen neural, keep old synthetic-family regression within 2 points, pass a same-component ablation, and repeat on a fresh environment seed. Thresholds cannot be relaxed after observation.

The resulting claim is limited to this pinned local target range and is not a production security claim.
