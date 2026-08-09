# Neural URL Generalization Loop 11

Status: **failed confirmation**. The target family remained completely absent from training.

| Fresh seed | Candidate neural | Same checkpoint, no set head | Frozen neural | Gain vs frozen | Set-head gain | Worst old regression |
|---:|---:|---:|---:|---:|---:|---:|
| 20261329 | 75.83% | 58.17% | 60.17% | +15.67% | +17.67% | -2.33% |
| 20261417 | 78.17% | 60.33% | 61.50% | +16.67% | +17.83% | -3.83% |

Candidate neural mean: 77.00%; frozen native neural mean: 60.83%. Counterexample Top-1 averaged 75.33%, versus 37.50% random.

The result is attributed to a fixed-budget learned set-comparison architecture plus canonical URL slots and episode-consistent meta-label training. C5 remained a separate executable rule path and is not included in the neural result.
