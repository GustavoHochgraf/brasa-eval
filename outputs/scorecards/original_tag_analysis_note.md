# Original PoETa Tag Analysis — Methodological Note

## Purpose

This analysis complements the paper's manual diagnostic grouping (6 categories)
with a breakdown using the **original PoETa v2 subcategory tags**.

## Key differences

| Aspect | Manual paper grouping | Original PoETa tags |
|--------|----------------------|---------------------|
| Cardinality | 6 categories | 11 tags |
| Label type | Single-label per task | Multi-label per task |
| Design goal | Interpretive diagnostic buckets for the paper | Benchmark-native capability taxonomy |
| Granularity | Coarser, oriented to training/evaluation decisions | Finer, oriented to capability coverage |

## Multi-label handling

Because original PoETa tags are multi-label (e.g., BB Mathematical Induction
has both "reasoning" and "math"), each task contributes to the mean score of
**every tag it belongs to**.  This means:

- Tag-level means are **not mutually exclusive**
- They should **not be summed** or averaged across tags
- The total "task count" across all tags exceeds 40

This is by design: it preserves the original benchmark's capability taxonomy
without forcing an artificial single-label reduction.

## Relationship to the paper

The paper uses the manual grouping for its main results tables and discussion.
The original-tag analysis is presented as a complementary view that:

1. Shows the benchmark's native structure
2. Demonstrates that even finer-grained breakdowns reveal training effects
3. Validates that the manual grouping captures the dominant patterns

## Files

- `data/original_poeta_tag_expanded.csv` — one row per (task, tag) pair
- `data/original_poeta_tag_summary.csv` — tag counts and task lists
- `data/manual_vs_original_taxonomy_map.csv` — both taxonomies side by side
- `outputs/scorecards/original_tag_breakdown.csv` — tidy scores per (checkpoint, tag)
- `outputs/scorecards/original_tag_breakdown_pivot.csv` — pivoted comparison
- `outputs/figures/original_tag_breakdown_heatmap.{png,pdf}` — heatmap
- `outputs/figures/original_tag_breakdown_by_checkpoint.{png,pdf}` — bar chart
