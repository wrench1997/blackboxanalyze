# Frontend structured-semantics loop 08

Status: **preregistered_pilot_passed**. Scope: research pilot; not a production vulnerability detector claim.

| Metric | Structured mean | Raw mean | Structured min |
|---|---:|---:|---:|
| postmessage_origin | 100.00% | 100.00% | 100.00% |
| dom_sink_injection | 100.00% | 60.48% | 100.00% |
| Counterexample Top-1 | 100.00% | 83.78% | 100.00% |

The preregistered pilot passes all frozen checks across three independent seeds. Counterexample@10 itself is saturated by the random baseline, so the stronger Top-1 result is reported alongside it.

## Root-cause result

The successful architecture is a small learned decoder plus an abstract, executable episode memory. URL transfer comes from cross-trace variable binding; DOM transfer comes from parser-derived markup structure. A protocol escaping bug (`<` inside serialized rules) was also identified and removed by using language-neutral Rule IR operators.

## Engineering boundary

This is mature enough for engineering-scale validation, not for a production security claim. The next stage must add languages, browser engines, noisy black boxes, throughput telemetry, and explicit experiment-vs-engineering failure triage.
