# ENEM / BLUEX Subarea Analysis — Summary

Supplementary drill-down within the Brazil / Exams category.
Baseline: Qwen3-1.7B-Base. Deltas in percentage points (pp).

## TuQwen (GigaVerbo)

**Largest gains vs baseline:**
- BLUEX / Biology: 0.69 -> 0.73 (+4.8 pp)
- ENEM 2022 / Mathematics: 0.14 -> 0.18 (+4.5 pp)
- ENEM 2022 / Natural Sciences: 0.58 -> 0.62 (+3.9 pp)

**Largest losses vs baseline:**
- ENEM 2022 / Languages: 0.76 -> 0.70 (-6.1 pp)
- BLUEX / English: 0.66 -> 0.61 (-4.3 pp)
- BLUEX / Chemistry: 0.44 -> 0.40 (-3.9 pp)

## QwenRolina (Carolina)

**Largest gains vs baseline:**
- ENEM 2022 / Natural Sciences: 0.58 -> 0.69 (+11.5 pp)
- BLUEX / Biology: 0.69 -> 0.78 (+9.6 pp)
- ENEM 2022 / Mathematics: 0.14 -> 0.23 (+9.1 pp)

**Largest losses vs baseline:**
- BLUEX / Physics: 0.29 -> 0.24 (-5.1 pp)
- BLUEX / Philosophy: 0.64 -> 0.59 (-4.5 pp)
- ENEM 2022 / Languages: 0.76 -> 0.73 (-3.0 pp)

## Key findings

- **ENEM 2022 Mathematics** sees the largest absolute gains from continued pretraining (+4.5 pp TuQwen, +9.1 pp QwenRolina), though from a very low baseline (0.14).
- **BLUEX Biology** improves with both corpora (+4.8 pp TuQwen, +9.6 pp QwenRolina), the largest BLUEX gain for QwenRolina.
- **BLUEX Physics** declines for at least one model (-1.3 pp TuQwen, -5.1 pp QwenRolina), suggesting continued pretraining in Portuguese does not help STEM problem-solving.
- **ENEM 2022 Natural Sciences** shows a strong QwenRolina gain (+11.5 pp), consistent with the Carolina corpus containing Brazilian educational content.
- Gains from continued pretraining are concentrated in humanities and life sciences; exact sciences (mathematics, physics, chemistry) show mixed or negative effects, reinforcing that linguistic pretraining benefits language-dependent reasoning more than formal/symbolic reasoning.
