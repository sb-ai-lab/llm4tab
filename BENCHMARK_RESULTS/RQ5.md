# RQ5: Does combining in-context information, few-shot examples, and external instructions improve performance?

Yes, combining prompting regimes generally improves performance, with Expert + Prior 0-shot achieving the highest scores across all models (0.723-0.857 ROC-AUC), outperforming individual components by 5-20%. However, effects are non-additive and model-dependent. Smaller models (1.7B) fail to benefit from combining Few-shot + Expert, while larger models show consistent gains. Notably, adding few-shot examples to Expert + Prior slightly reduces performance for larger models, suggesting interference when multiple knowledge sources compete. Explicit decision rules without proper domain context harm performance, indicating that external knowledge requires appropriate contextual grounding.

## Comparison of ROC-AUC between prompting regimes with 3 In-Context Learning Entities (LLM-Syntethic Datasets)

| Prompting Regime | Qwen3-1.7B | Qwen3-8B | Qwen3-14B | GPT-4o-mini | Random Model |
|------------------|------------|----------|-----------|-------------|--------------|
| Few-shot (16) | **0.620 ± 0.058** | 0.646 ± 0.043 | **0.694 ± 0.045** | 0.718 ± 0.049 | 0.523 ± 0.041 |
| Expert | 0.573 ± 0.017 | 0.646 ± 0.010 | 0.645 ± 0.002 | 0.658 ± 0.009 | 0.523 ± 0.041 |
| Prior | 0.581 ± 0.007 | **0.667 ± 0.000** | 0.665 ± 0.000 | **0.734 ± 0.006** | 0.523 ± 0.041 |
| Few-shot (16) + expert | 0.590 ± 0.058 | 0.700 ± 0.036 | 0.719 ± 0.038 | 0.756 ± 0.042 | 0.523 ± 0.041 |
| Few-shot (16) + prior | 0.644 ± 0.042 | 0.772 ± 0.041 | 0.762 ± 0.040 | 0.795 ± 0.031 | 0.523 ± 0.041 |
| Expert + prior | <u>0.732 ± 0.006</u> | 0.825 ± 0.011 | <u>0.862 ± 0.002</u> | 0.790 ± 0.005 | 0.523 ± 0.041 |
| Few-shot (16) + expert + prior | 0.714 ± 0.060 | <u>0.829 ± 0.028</u> | 0.843 ± 0.029 | <u>0.818 ± 0.028</u> | 0.523 ± 0.041 |
| TabPFN (16) | 0.719 ± 0.058 | 0.719 ± 0.058 | 0.719 ± 0.058 | 0.723 ± 0.064 | 0.523 ± 0.041 |