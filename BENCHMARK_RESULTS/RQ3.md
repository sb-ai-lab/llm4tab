# RQ3: Do LLMs effectively leverage few-shot demonstrations for tabular classification?

Yes, LLMs effectively leverage few-shot examples, with statistically significant improvements over zero-shot baselines across all shot configurations. However, gains are domain-dependent and non-monotonic: performance peaks at 8-16 shots and plateaus thereafter. Larger models show smaller marginal gains from additional shots despite higher absolute performance, suggesting they rely more on pretrained knowledge while smaller models depend more heavily on explicit demonstrations.

## Models vs Shots on Old Datasets (Cols-Context, Real Data)

| Model | 0-shot | 4-shot | 8-shot | 16-shot | 32-shot | 64-shot |
|-------|--------|--------|--------|---------|---------|---------|
| GPT-4o-mini | 0.728 ± 0.011 | 0.756 ± 0.040 | 0.770 ± 0.044 | 0.774 ± 0.042 | 0.786 ± 0.030 | 0.793 ± 0.022 |
| Qwen3-1.7B | 0.640 ± 0.007 | 0.675 ± 0.056 | 0.672 ± 0.077 | 0.690 ± 0.068 | 0.703 ± 0.060 | 0.715 ± 0.057 |
| Qwen3-8B | 0.711 ± 0.000 | 0.746 ± 0.051 | 0.766 ± 0.049 | 0.778 ± 0.039 | 0.789 ± 0.033 | 0.793 ± 0.024 |
| Qwen3-14B | 0.733 ± 0.000 | 0.754 ± 0.048 | 0.774 ± 0.043 | 0.784 ± 0.034 | 0.792 ± 0.025 | 0.801 ± 0.022 |