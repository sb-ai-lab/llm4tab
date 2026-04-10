
# RQ1: How competitive are LLMs in zero-shot tabular classification?

Zero-shot LLMs achieve classification performance comparable to 16-shot TabPFN across all datasets and outperform TabPFN trained on the full training set in the healthcare domain. Moderate performance drops on post-cutoff datasets suggest that data leakage is not the primary driver of these results.

## Zero-shot performance by dataset type (Real Data)

| Dataset Group | GPT-4o-mini (gen) | Qwen3-1.7B (gen) | Qwen3-8B (gen) | Qwen3-14B (gen) | TabPFN (16-shot) | Random Model |
|--------------|-------------------|------------------|----------------|-----------------|------------------|----------------|
| All (Classic + New) | 0.712 ± 0.010 | 0.626 ± 0.006 | 0.698 ± 0.001 | 0.714 ± 0.000 | 0.749 ± 0.063 | 0.564 ± 0.073 |
| Classic | 0.728 ± 0.011 | 0.640 ± 0.007 | 0.711 ± 0.000 | 0.733 ± 0.000 | 0.767 ± 0.059 | 0.593 ± 0.117 |
| New | 0.664 ± 0.008 | 0.587 ± 0.005 | 0.657 ± 0.001 | 0.657 ± 0.000 | 0.693 ± 0.073 | 0.535 ± 0.028 |