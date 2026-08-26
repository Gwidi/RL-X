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

The jobs are:

- `137440`: randomized-spine scheduler smoke, one RTX 2080, 30 minutes.
- `137443`: fixed default spine, one RTX 3090 or A5000, three days.
- `137444`: full spine-design randomization, one RTX 3090 or A5000, three days.

The two production jobs have the Slurm dependency `afterok:137440`, so they
cannot begin unless the 4096-environment PPO smoke completes successfully.
At submission all three jobs were pending; the smoke was waiting for priority
and the production jobs were waiting for the smoke.

The W&B destination for both production jobs is
`gwidon-szczepankiewicz-poznan-university-of-technology/spine_gym_bunker_ruins`.
Write access from IAS was checked before submission.

Remote SHA-256 hashes:

```text
3911987e62b35e54daebd3e56438f1ab2a3ddad512062876b48a42a20df2e644  ias_spine_bunker.sbatch
00527e43d38b72e244a45f979b95f91a54261183364fdd4a6f1785c64b81c110  spine_bunker_fixed.args
f9e6439c130213e49167b74485450058ceb1783919e05337f5e879702c299882  spine_bunker_randomized.args
e1d7c1c1729f038badf8244cec1be920ff0a0792d17f6faa57eaf37dcd4aa9d8  spine_bunker_smoke.args
```
