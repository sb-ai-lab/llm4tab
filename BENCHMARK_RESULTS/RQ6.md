
# RQ6: How does dataset complexity influence LLM performance on tabular tasks?

LLMs successfully capture predictive patterns only at low complexity levels. Performance gains from additional shots diminish rapidly as complexity increases, with no statistically significant improvement beyond moderate complexity (mlp2 and above). This capability scales with model size: smaller models (1.7B) fail even at the simplest level, while larger models (14B) extract signal up to moderate complexity. Real-world datasets likely fall within the low-to-moderate complexity range where LLMs remain effective.

## Model performance on MLP-synthetic datasets across shots

| Dataset | GPT-4o-mini 0-shot | GPT-4o-mini 4-shot | GPT-4o-mini 16-shot | Qwen3-1.7B 0-shot | Qwen3-1.7B 4-shot | Qwen3-1.7B 16-shot | Qwen3-8B 0-shot | Qwen3-8B 4-shot | Qwen3-8B 16-shot | Qwen3-14B 0-shot | Qwen3-14B 4-shot | Qwen3-14B 16-shot |
|---------|-------------------|-------------------|---------------------|------------------|------------------|-------------------|-----------------|-----------------|------------------|------------------|------------------|-------------------|
| MLP0 | 0.608 ± 0.004 | 0.645 ± 0.065 | 0.708 ± 0.055 | 0.570 ± 0.004 | 0.641 ± 0.052 | 0.662 ± 0.039 | 0.564 ± 0.001 | 0.639 ± 0.070 | 0.713 ± 0.059 | 0.602 ± 0.000 | 0.688 ± 0.088 | 0.748 ± 0.059 |
| MLP1 | 0.596 ± 0.007 | 0.593 ± 0.023 | 0.578 ± 0.033 | 0.564 ± 0.002 | 0.597 ± 0.019 | 0.596 ± 0.013 | 0.562 ± 0.004 | 0.587 ± 0.032 | 0.568 ± 0.051 | 0.562 ± 0.000 | 0.584 ± 0.032 | 0.579 ± 0.053 |
| MLP2 | 0.563 ± 0.010 | 0.558 ± 0.023 | 0.566 ± 0.020 | 0.534 ± 0.001 | 0.555 ± 0.016 | 0.556 ± 0.012 | 0.547 ± 0.004 | 0.564 ± 0.022 | 0.580 ± 0.030 | 0.535 ± 0.000 | 0.564 ± 0.029 | 0.577 ± 0.028 |
| MLP3 | 0.603 ± 0.008 | 0.569 ± 0.035 | 0.581 ± 0.026 | 0.540 ± 0.002 | 0.580 ± 0.022 | 0.577 ± 0.012 | 0.566 ± 0.011 | 0.576 ± 0.027 | 0.573 ± 0.027 | 0.583 ± 0.001 | 0.587 ± 0.041 | 0.587 ± 0.031 |
| MLP5 | 0.546 ± 0.006 | 0.542 ± 0.014 | 0.541 ± 0.013 | 0.532 ± 0.001 | 0.518 ± 0.013 | 0.520 ± 0.017 | 0.527 ± 0.007 | 0.536 ± 0.016 | 0.538 ± 0.018 | 0.541 ± 0.000 | 0.537 ± 0.015 | 0.543 ± 0.026 |
| MLP7 | 0.533 ± 0.005 | 0.538 ± 0.019 | 0.541 ± 0.012 | 0.583 ± 0.002 | 0.532 ± 0.017 | 0.545 ± 0.015 | 0.523 ± 0.006 | 0.534 ± 0.018 | 0.535 ± 0.018 | 0.537 ± 0.000 | 0.534 ± 0.017 | 0.540 ± 0.019 |
| MLP9 | 0.526 ± 0.007 | 0.526 ± 0.012 | 0.525 ± 0.009 | 0.523 ± 0.008 | 0.516 ± 0.012 | 0.518 ± 0.011 | 0.526 ± 0.009 | 0.525 ± 0.016 | 0.529 ± 0.014 | 0.521 ± 0.000 | 0.528 ± 0.012 | 0.523 ± 0.011 |