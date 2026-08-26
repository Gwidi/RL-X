# Configurable Silver Badger

This robot option ports the 10-dimensional Silver Badger spine design from the
`loco_mjx` `co_design` branch while leaving `silver_badger` unchanged.

Select it with:

```bash
--environment.train_robot=silver_badger_codesign
```

Enable independent per-environment sampling over the complete design space:

```bash
--environment.train_robot=silver_badger_codesign \
--environment.spine_design_randomization_enabled=True
```

With randomization disabled, the configurable model uses the default Silver
Badger spine parameters. The defaults and all bounds live in `robot_config.py`.

The sampled spine design has precedence over seen/unseen robot and MuJoCo
domain randomization for the same model fields. Joint dropout still applies to
the legs, but deliberately excludes the configurable spine joint.
