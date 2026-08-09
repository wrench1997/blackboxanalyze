# URL Root-Cause Loop 10b

Status: **confirmed**.

C5 was frozen before generating two fresh datasets. Across all six data-seed × model-seed evaluations, URL agreement moved from a 44.36% legacy mean to 100.00%; the minimum C5 score was 100.00%. The neural path remained 48.94%, so this is an architecture/rule-abstraction repair, not neural generalization.

| Data seed | Model seed | Legacy | C5 | Neural | Worst regression |
|---:|---:|---:|---:|---:|---:|
| 20261123 | 20261001 | 42.17% | 100.00% | 43.83% | -0.33% |
| 20261123 | 20261019 | 42.33% | 100.00% | 49.50% | +0.00% |
| 20261123 | 20261107 | 50.67% | 100.00% | 55.33% | -0.67% |
| 20261211 | 20261001 | 39.33% | 100.00% | 41.83% | -0.33% |
| 20261211 | 20261019 | 41.67% | 100.00% | 47.83% | +0.00% |
| 20261211 | 20261107 | 50.00% | 100.00% | 55.33% | -0.33% |

Worst regression across all six evaluations: -0.67%. The C5 rule head uses only black-box observations, requires both label classes, chooses maximum empirical fit, and abstains when equally fitting rules disagree.
