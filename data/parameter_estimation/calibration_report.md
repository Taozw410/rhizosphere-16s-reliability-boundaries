# Paper 4 phase 1 calibration

- Simulator: template_multinomial_v1
- Selected sample pseudocount: 1 reads
- Final seeds: 5
- Passed summaries: 8/8

| Statistic | Real | Synthetic mean | Distance | Tolerance | Pass |
|---|---|---|---:|---:|---|
| prevalence_distribution | 0.0033557, 0.0075503, 0.027685, 0.099832, 0.2505, 0.38104, 0.66366 | 0.003255; 0.0075503; 0.026846; 0.099161; 0.24869; 0.37956; 0.66398 | 0.0077 | 0.120 | yes |
| rank_abundance | -6.7251, -6.0119, -5.3552, -4.4975, -3.6995, -3.0402, -2.1407 | -6.7983; -6.0191; -5.3528; -4.498; -3.6989; -3.0401; -2.1409 | 0.0080 | 0.120 | yes |
| sample_depth | 8.4554, 9.3763, 9.9719, 10.354, 10.691 | 8.4554; 9.3763; 9.9719; 10.354; 10.691 | 0.0000 | 0.050 | yes |
| within_between_variance_ratio | 0.867451 | 0.868136 [0.865662, 0.873351] | 0.0029 | 0.300 | yes |
| bray_curtis_distribution | 0.52624, 0.6734, 0.77711, 0.86789, 0.96008 | 0.52749; 0.67424; 0.77765; 0.86824; 0.96012 | 0.0021 | 0.100 | yes |
| aitchison_distribution | 46.559, 58.987, 66.94, 73.788, 82.13 | 46.461; 58.87; 66.818; 73.655; 82.007 | 0.0051 | 0.100 | yes |
| zero_fraction | 0.80388, 0.88296, 0.92395, 0.95407, 0.98617 | 0.80566; 0.88383; 0.9239; 0.95383; 0.98577 | 0.0089 | 0.080 | yes |
| network_topology | 0.30961, 0.44136, 0.65357 | 0.29892; 0.43542; 0.65192 | 0.1273 | 0.200 | yes |

## Interpretation

The selected MVP is a sample-template multinomial parametric bootstrap. It preserves the observed project design, sample-level co-variation and zero structure while adding read sampling noise. It is suitable as a calibrated null/template simulator. It is not yet an independently generated hierarchical community model.

The independent project-mean + presence-mask generator was tested and rejected at this stage: it overestimated zeros and underestimated network co-occurrence. It remains a Phase-2 task.