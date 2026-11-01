# Paper 4 taxonomy-classification benchmark specification v1.0

Date frozen: 2026-10-28

## Reference and labels

- Reference extract: SILVA SSU r138.2 V4 region, 515F/806R.
- Classification rank: genus.
- Eligible genera require at least 10 full V4 reference sequences after
  length and character filtering.
- The benchmark uses the 24 genera with the largest eligible reference sets.
- Four genera are withheld completely to simulate unseen taxa.
- The remaining 20 genera define the closed-set label space.

## Data split

For each genus and seed:

- up to six reference sequences for training;
- two for method validation;
- two for testing;
- training sequences are augmented threefold with 0.3% substitutions;
- validation and test queries receive 0.5% substitutions.

The seed controls genus holdout and sequence assignment.

## Methods

- 4-mer/5-mer nearest-neighbor classifier;
- multinomial naive Bayes;
- linear SGD classifier with log loss;
- small one-dimensional CNN;
- lightweight k-mer Transformer trained from scratch.

The k-mer Transformer is not a pretrained DNA-BERT model. Results from this
method must not be described as DNA-BERT performance.

## Scenarios and endpoints

- `closed_set`: query sequences belong to genera represented in training;
- `unseen_genus`: all query genera are absent from training.

Primary endpoints:

- closed-set accuracy and macro F1;
- unseen-genus rejection at a threshold calibrated to retain 95% of known
  validation samples;
- known-taxon retention.

## Assessment boundary

This benchmark tests sequence error and unseen-genus shift within one SILVA V4
extract. It does not test cross-database taxonomy conflicts, primer-region
shift, chimeras, or a pretrained genomic language model.

