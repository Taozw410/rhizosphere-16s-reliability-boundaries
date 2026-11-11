# Manuscript integration draft

## Contribution statement

We introduce the Transportability-Aware Reliability Envelope (TARE), a
project-held-out model that converts study design, sequencing characteristics,
observed community structure and effect severity into a method-specific
performance interval and a recommendation decision. TARE distinguishes three
objects that the original benchmark kept separate only qualitatively:
within-study reproducibility, accuracy against simulator-defined truth, and
transportability to a new project.

## Methods paragraph

TARE-DA was trained on 90 project-specific known-effect datasets generated
from 45 rhizosphere projects. For each project, we generated one null and one
strong sparse-effect dataset on the frozen 120-feature panel. Conditions were
balanced within a project, and the realized compositional effect after
renormalization was used as truth. ALDEx2, ANCOM-BC2, LinDA and MaAsLin2 were
run under a common fixed-effect condition model; project was held constant and
therefore could not enter as a random factor. Method utility was defined as F1
under the sparse-effect scenario and one minus the false-positive rate under
the null scenario. Reliability required FDP ≤ 0.10 and F1 ≥ 0.50 for the
sparse-effect scenario, or false-positive rate ≤ 0.10 for the null scenario.

TARE-DA combined study descriptors, target-project prevalence, zero fraction,
within-to-between variance ratio, Bray–Curtis and Aitchison summaries, network
topology, method identity and scenario severity. A shallow histogram
gradient-boosted regression tree predicted utility. Four-fold inner
project-group cross-fitting generated conformal residuals; their 95th
percentile defined the interval radius. A separate shallow classifier
estimated the probability of meeting the reliability criterion. A method was
allowed only when that probability was at least 0.80 and the lower conformal
utility bound was at least 0.50.

## Results paragraph

Across ten repeated stratified five-fold project holdouts, TARE-DA reduced
utility MAE from 0.0948 under a study-agnostic method-by-scenario mean to
0.0790, a 16.7% reduction. The improvement occurred in every partition. The
95% interval covered 95.4% of held-out utilities. In the heterogeneous
strong-effect stratum, where project specificity mattered most, MAE decreased
from 0.1640 to 0.1271 and reliability discrimination reached AUROC 0.883.
TARE-DA issued 1,786 ALLOW decisions among 3,600 predictions, with precision
0.983; all but 14 were in the null scenario. Under the frozen criterion, only
4.4%–6.7% of strong-effect project-method combinations were reliable, so TARE
withheld recommendation for the sparse-effect scenario despite useful ranking
of risk.

In a fully independent six-project external panel (two maize, two rice and two
wheat projects; 426 samples), the serialized model was applied without
retraining. Utility MAE was 0.1414 versus 0.1853 for the development-only
method-by-scenario prior, interval coverage was 0.854 and strong-effect
reliability AUROC was 0.837. All strong-effect cases were withheld, whereas all
24 null cases received correct ALLOW decisions. Every external project fell
outside the 95th percentile of the development support distance, primarily
because four projects contained 54–177 samples versus 4–35 in development.
The panel therefore demonstrates directional robustness, calibration caution
and conservative abstention, but not in-distribution generalization.

## Discussion paragraph

The gain is not a claim that simulation rankings transfer to every unseen
rhizosphere study. It shows that a calibrated envelope can improve conditional
performance prediction and can withhold recommendations when evidence is
insufficient. TARE-DA therefore converts the benchmark's negative external
validation result into an operational boundary: simulation-derived method
recommendations are permitted only within a learned calibration envelope, and
the absence of support is reported rather than averaged into a universal
ranking.

## Figure legend draft

**Figure 6 | Project-held-out TARE reliability envelopes.** **a**, Utility MAE
for TARE-DA and the study-agnostic method-by-scenario mean under null and strong
sparse-effect scenarios. **b**, Empirical coverage of 95% conformal intervals.
**c**, AUROC for identification of reliable method outcomes in the
strong-effect scenario; null-scenario AUROC is omitted because almost all
outcomes are reliable and the endpoint is highly imbalanced. **d**, ALLOW,
CONDITIONAL and DENY decisions across 3,600 held-out predictions. TARE-DA uses
project-held-out training and simulator-defined truth; it does not estimate
biological truth on unlabelled real projects.
