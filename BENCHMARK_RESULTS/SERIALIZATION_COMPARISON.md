# Serialization comparison for Qwen3 model family

This table ranks different serialization methods (e.g., feat_val, html, markdown) based on their internal comparison of ROC AUC scores.

Thus, a method like feat_val is considered best overall because it has the lowest average rank (2.38) and highest first-place rate (37.5%), despite not always winning.

| Serialization | Avg. Rank | Std. Rank | First Place (%) |
|---------------|-----------|-----------|----------------|
| feat_val      | 2.38      | 0.94      | 37.5           |
| html          | 2.64      | 0.84      | 21.9           |
| markdown      | 2.93      | 0.85      | 18.8           |
| markdown_masked | 3.28   | 1.06      | 12.5           |
| feat_val_masked | 3.77    | 0.88      | 3.1            |