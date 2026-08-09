# Neural URL Generalization Loop 11

Status: **confirmed**. The target family remained completely absent from training.

| Fresh seed | Candidate neural | Same checkpoint, no set head | Frozen neural | Gain vs frozen | Set-head gain | Worst old regression |
|---:|---:|---:|---:|---:|---:|---:|
| 20261603 | 88.83% | 60.00% | 63.83% | +25.00% | +28.83% | -0.83% |
| 20261621 | 88.00% | 58.83% | 62.67% | +25.33% | +29.17% | +0.00% |

Candidate neural mean: 88.42%; frozen native neural mean: 63.25%. Counterexample Top-1 averaged 88.00%, versus 37.50% random.

The result is attributed to a fixed-budget learned set-comparison architecture plus canonical URL slots and episode-consistent meta-label training. C5 remained a separate executable rule path and is not included in the neural result.
