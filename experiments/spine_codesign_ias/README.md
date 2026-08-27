# IAS Silver Badger spine co-design runs

These launch files reproduce the Silver Badger bunker-training jobs submitted
on 27 August 2026. Both treatments use seed 0 and an immutable source checkout
from the `spine_codesign` branch of `Gwidi/RL-X`.

The isolated IAS installation is rooted at
`/home/bohlinger/rlx_spine_codesign_cluster`. Its dedicated Conda environment
contains Python 3.11.4, JAX 0.7.2, Flax 0.12.0, and MuJoCo 3.7.0. The explicit
package manifest is stored remotely as
`configs/conda-explicit-abfbf8a.txt`.

The initial jobs were:

- `137440`: randomized-spine scheduler smoke, one RTX 2080, 30 minutes.
- `137443`: fixed default spine, one RTX 3090 or A5000, three days.
- `137444`: full spine-design randomization, one RTX 3090 or A5000, three days.

The two initial production jobs had the Slurm dependency `afterok:137440`.
The smoke completed successfully in 7:33, after which both production jobs
started but failed before training because the IAS W&B key for
`nico-bohlinger` did not have permission to write to Gwidon's entity.

The corrected W&B destination is
`nico-bohlinger/spine_gym_bunker_ruins`. CPU preflight job `139268` completed
successfully and synced W&B run `89bo8wnw` before the corrected production
jobs were submitted:

- `139270`: fixed default spine, one RTX 3090 or A5000, three days; W&B
  run `ifu116fe`.
- `139271`: full spine-design randomization, one RTX 3090 or A5000, three
  days; W&B run `j7k8pfhw`.

Jobs `139270` and `139271` were cancelled before their first PPO update after
the hinge-axis design was changed from a redundant normalized three-vector to a
two-coordinate tangent-plane exponential map. No results from those jobs are
valid for the final design space.

The replacement source commit is
`275b9490b7937dbc81882ec1bb1f1a6020a0e256`. It includes the tangent-axis
parameterization, stable zero-angle gradients, nominal state-bank collection,
and nominal-referenced single-critic design search. Replacement production
jobs are gated on a fresh 4096-environment randomized-design smoke test.

The replacement jobs are:

- `139324`: randomized tangent-axis scheduler smoke, one RTX 2080, 30 minutes.
- `139325`: fixed nominal spine, one 24 GB GPU, three days; dependency
  `afterok:139324`.
- `139326`: full 9-dimensional spine randomization, one 24 GB GPU, three days;
  dependency `afterok:139324`.

The W&B destination for the replacement production jobs is
`nico-bohlinger/spine_gym_bunker_ruins`.

Remote SHA-256 hashes:

```text
dacf61815d4378271d6ffe403d38e1594e37527dafa5b01ea68bd4afdb484085  ias_spine_bunker_275b949.sbatch
c7fc4f09150e314eaede10f916613f165cded990c7495568d8d52741877bfc08  spine_bunker_smoke.args
6431adbb127f3d22fcefa957941a59e96f31a75c4a4410ce80a80804e4680a72  spine_bunker_fixed.args
90c430fd3d60b0a5690d000139f0af1c8560a206367ecd549eff00a0f2aea291  spine_bunker_randomized.args
```
