# Offline spine design search

The workflow deliberately runs after policy training:

1. Train one PPO policy and one critic with
   `spine_design_randomization_enabled=True`.
2. Collect a state bank with that checkpoint but the nominal morphology:

   ```bash
   python experiments/spine_codesign/collect_state_bank.py \
     --runner.load_model=/path/to/latest.model \
     --algorithm.spine_state_bank_path=/path/to/nominal_state_bank.npz \
     --algorithm.spine_state_bank_size=4096 \
     --environment.nr_envs=64 \
     --environment.terrain.type=hfield_bunker_ruins \
     --environment.critic_exteroceptive_observation_type=height_samples
   ```

3. Search from the nominal design with the same checkpoint and the single PPO
   critic:

   ```bash
   python experiments/spine_codesign/search_design.py \
     --runner.load_model=/path/to/latest.model \
     --algorithm.spine_state_bank_path=/path/to/nominal_state_bank.npz \
     --algorithm.spine_search_output_path=/path/to/best_spine_design.npz \
     --environment.terrain.type=hfield_bunker_ruins \
     --environment.critic_exteroceptive_observation_type=height_samples
   ```

The search variable is a normalized physical-design delta whose zero is the
nominal Silver Badger. At every step, the candidate is decoded into the ten
design observations expected by the trained critic and substituted into every
saved state. The objective is the mean of the single critic minus a configurable
L2 penalty to the nominal design. The tangent-axis pair is projected onto its
hemisphere disk after every optimizer step.

All observation-architecture flags must match training when loading a
checkpoint. A repeatable end-to-end check with an untrained policy and critic is
available as:

```bash
python experiments/spine_codesign/smoke_untrained.py
```
