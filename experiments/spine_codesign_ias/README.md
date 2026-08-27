# IAS Silver Badger spine co-design runs

These launch files reproduce the two bunker-training jobs submitted on 27
August 2026. Both treatments use seed 0 and the immutable source commit
`abfbf8a171d8567b8d4c243308ef834d3e62ebf8` from the `spine_codesign`
branch of `Gwidi/RL-X`.

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

The successful smoke was not rerun. Slurm had already purged job `137440` from
its live dependency table, so the corrected jobs were submitted directly after
its successful accounting state and completion marker were rechecked.

The W&B destination for both corrected production jobs is
`nico-bohlinger/spine_gym_bunker_ruins`.

Remote SHA-256 hashes:

```text
3911987e62b35e54daebd3e56438f1ab2a3ddad512062876b48a42a20df2e644  ias_spine_bunker.sbatch
ba742318b4b1f40de93074da37b9adbf73891b3e7f42fdea8d3d997d1c28f7c4  spine_bunker_fixed.args
40d4e8435460018c79f097497b6205ba9c4bd43cd3da036755899d291bc11f42  spine_bunker_randomized.args
e1d7c1c1729f038badf8244cec1be920ff0a0792d17f6faa57eaf37dcd4aa9d8  spine_bunker_smoke.args
```
