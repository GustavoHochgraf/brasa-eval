# ENEM / BLUEX Subarea Analysis — Summary

Supplementary drill-down within the Brazil / Exams category.
Baseline: Qwen 1.7B Base. Deltas in percentage points (pp).

## Gigaverbo adapted

**Largest gains vs baseline:**
- BLUEX / Biology: 0.69 -> 0.73 (+4.8 pp)
- ENEM 2022 / Mathematics: 0.14 -> 0.18 (+4.5 pp)
- ENEM 2022 / Natural Sciences: 0.58 -> 0.62 (+3.9 pp)

**Largest losses vs baseline:**
- ENEM 2022 / Languages: 0.76 -> 0.70 (-6.1 pp)
- BLUEX / English: 0.66 -> 0.61 (-4.3 pp)
- BLUEX / Chemistry: 0.44 -> 0.40 (-3.9 pp)

## Carolina adapted

**Largest gains vs baseline:**
- ENEM 2022 / Natural Sciences: 0.58 -> 0.69 (+11.5 pp)
- BLUEX / Biology: 0.69 -> 0.78 (+9.6 pp)
- ENEM 2022 / Mathematics: 0.14 -> 0.23 (+9.1 pp)

**Largest losses vs baseline:**
- BLUEX / Physics: 0.29 -> 0.24 (-5.1 pp)
- BLUEX / Philosophy: 0.64 -> 0.59 (-4.5 pp)
- ENEM 2022 / Languages: 0.76 -> 0.73 (-3.0 pp)

## Key findings

- **ENEM 2022 Mathematics** sees the largest absolute gains from continued pretraining (+4.5 pp Gigaverbo adapted, +9.1 pp Carolina adapted), though from a very low baseline (0.14).
- **BLUEX Biology** improves with both corpora (+4.8 pp Gigaverbo adapted, +9.6 pp Carolina adapted), the largest BLUEX gain for Carolina adapted.
- **BLUEX Physics** declines for at least one model (-1.3 pp Gigaverbo adapted, -5.1 pp Carolina adapted), suggesting continued pretraining in Portuguese does not help STEM problem-solving.
- **ENEM 2022 Natural Sciences** shows a strong Carolina adapted gain (+11.5 pp), consistent with the Carolina corpus containing Brazilian educational content.
- Gains from continued pretraining are concentrated in humanities and life sciences; exact sciences (mathematics, physics, chemistry) show mixed or negative effects, reinforcing that linguistic pretraining benefits language-dependent reasoning more than formal/symbolic reasoning.
