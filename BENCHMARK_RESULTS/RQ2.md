# RQ2: To what extent does zero-shot performance depend on in-context information versus LLM prior knowledge?

Zero-shot performance depends on both prior knowledge and in-context information, with a clear scaling effect: larger models (>8B parameters) benefit more from added context, while the smallest model (1.7B) performs better without it. This suggests that larger models integrate task descriptions and schema cues more effectively, whereas smaller models may rely primarily on prior knowledge encoded during pretraining.

## Context configurations comparison across models on Old Datasets (0-shot, Real Data)

| Configuration | GPT-4o-mini | Qwen3-1.7B | Qwen3-8B | Qwen3-14B | Random Model |
|---------------|-------------|------------|----------|-----------|--------------|
| NoCols-NoContext | 0.625 ± 0.013 | 0.610 ± 0.003 | 0.579 ± 0.006 | 0.596 ± 0.002 | 0.593 ± 0.117 |
| Cols-NoContext | 0.682 ± 0.013 | 0.645 ± 0.002 | 0.637 ± 0.008 | 0.678 ± 0.000 | 0.593 ± 0.117 |
| NoCols-Context | 0.641 ± 0.013 | 0.608 ± 0.010 | 0.624 ± 0.001 | 0.660 ± 0.000 | 0.593 ± 0.117 |
| Cols-Context | 0.728 ± 0.011 | 0.640 ± 0.007 | 0.711 ± 0.000 | 0.733 ± 0.000 | 0.593 ± 0.117 |