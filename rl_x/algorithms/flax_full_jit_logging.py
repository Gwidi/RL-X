import jax.numpy as jnp
import tree


CURRICULUM_METRIC_PREFIXES = ("env_curriculum/", "terrain_curriculum/")
CURRICULUM_PERCENTILES = (5, 25, 75, 95)


def aggregate_metrics(infos, optimization_metrics):
    """Aggregate full-JIT metrics and retain curriculum distribution details."""
    combined_metrics = {**infos, **optimization_metrics}

    for metric_name, values in infos.items():
        if metric_name.startswith(CURRICULUM_METRIC_PREFIXES):
            percentile_values = jnp.percentile(
                values,
                jnp.asarray(CURRICULUM_PERCENTILES),
            )
            for index, percentile in enumerate(CURRICULUM_PERCENTILES):
                combined_metrics[f"{metric_name}_p{percentile}"] = percentile_values[index]

    return tree.map_structure(jnp.mean, combined_metrics)
