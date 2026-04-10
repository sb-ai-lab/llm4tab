DATASETS=(
  bank_credit_scoring
  callcenter
  crimes_arrest
  extrovert
  machine
  postpartum
  reading
  stars
)

for dataset in "${DATASETS[@]}"; do
    for i in 0 1 2 3 4; do
    python -m llm_trees.cli generate \
        --dataset "$dataset" \
        --method gpt-4o-mini \
        --num_trees 1 \
        --regenerating_invalid_trees False \
        --iter $i \
        --seed $i \
        --num_retry_llm 3

    python -m llm_trees.cli eval_induction \
        --dataset "$dataset" \
        --method gpt-4o-mini \
        --iter $i \
        --seed $i \
        --generate_tree_if_missing False \
        --num_retry_llm 3
    done
done
