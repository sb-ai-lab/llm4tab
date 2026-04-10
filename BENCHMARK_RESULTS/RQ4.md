# RQ4: How does removing in-context information affect few-shot performance across shot regimes?

The contribution of in-context information diminishes as shot count increases. The performance gap between high-context and no-context configurations shrinks substantially (from Δ = 0.142 at 0-shot to Δ = 0.033 at 64-shot for GPT-4o-mini). This indicates that in-context information and few-shot examples act as partial substitutes: prior knowledge bootstraps performance when examples are scarce, but becomes redundant when sufficient demonstrations are available. Notably, at high shot counts, removing all contextual cues can outperform partial-context configurations, suggesting interference between pretrained priors and task-specific examples.

## Delta (Cols-Context - NoCols-NoContext) across shots on Old Datasets (Real Data)

| Model | 0-shot | 4-shot | 8-shot | 16-shot | 32-shot | 64-shot |
|-------|--------|--------|--------|---------|---------|---------|
| GPT-4o-mini | +0.103 ± 0.017 | +0.058 ± 0.066 | +0.057 ± 0.069 | +0.045 ± 0.069 | +0.041 ± 0.050 | +0.031 ± 0.038 |
| Qwen3-1.7B | +0.029 ± 0.007 | +0.015 ± 0.081 | +0.007 ± 0.101 | +0.023 ± 0.093 | +0.003 ± 0.077 | +0.001 ± 0.077 |
| Qwen3-8B | +0.132 ± 0.006 | +0.062 ± 0.075 | +0.062 ± 0.074 | +0.043 ± 0.070 | +0.040 ± 0.051 | +0.026 ± 0.038 |
| Qwen3-14B | +0.137 ± 0.002 | +0.060 ± 0.071 | +0.061 ± 0.069 | +0.036 ± 0.064 | +0.033 ± 0.045 | +0.026 ± 0.036 |