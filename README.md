# Privacy-Aware Federated Learning for Heart-Disease Prediction

This educational Python project compares centralized and simulated federated
logistic-regression models using the UCI Cleveland Heart Disease dataset and
the separate cardio cardiovascular dataset. The Cleveland experiment also
examines client-level differentially private FedAvg, client counts, and client
dropout. It is an experiment, not a medical diagnostic system.

## Run

From the project directory, install `requirements.txt` and run:

```powershell
python main.py
```

The included `heart+disease.zip` is used as the input dataset. Every run writes
its reports, graphs, and models to a new timestamped directory under `results/`;
existing result files are not overwritten.

## Dataset and preprocessing

The dataset archive contains the UCI Cleveland `processed.cleveland.data` file.
The experiment uses 303 rows, 13 clinical/categorical features, and maps the
original target to `0` (no disease) or `1` (disease present). Rows without a
valid target are removed. A stratified, fixed-seed 75/25 train/test split is
made before preprocessing. Missing feature values are imputed using training
medians, and `StandardScaler` is fitted on training data only. The imputation
medians and scaling values are saved with each model and reused at prediction
time.

Feature order for raw prediction is:

`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`

```python
import main

record = [
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal
]
probability = main.predict_proba_from_raw(record)
prediction = main.predict_from_raw(
    record
)
```

The default prediction helper loads the latest standard Cleveland FedAvg model
from `results/run_*/`. To use another model, pass its path as `model_path`; DP
models are saved separately. Cardio models require the cardio feature order
documented below. Missing raw feature values (`NaN`) are imputed with the saved
training medians.

## System and federated methodology

`main.py` performs the full experiment:

1. Split off a held-out test set and fit preprocessing on training data only.
2. Partition the training rows across four simulated clients with non-IID
   class proportions. Every training row is assigned exactly once.
3. Train local logistic-regression models with minibatch gradient descent.
4. Aggregate each client's model using sample-count-weighted FedAvg.
5. Compare Local-only, Centralized, FedAvg, and DP-FedAvg on the same test set.
6. Run noise, client-count, and dropout experiments and save metrics, plots,
   and model artifacts.

The standard four-client experiment drops client 3 in round 4. Aggregation
weights are recalculated using only clients that participated in that round.
For scalability, client counts are 2, 4, 6, and 8; those runs disable dropout
to isolate client-count effects. The dropout comparison uses four clients and
compares the no-dropout run against the standard round-4 dropout schedule.

## Differential privacy

DP-FedAvg computes each participating client's full model delta (weights and
bias), clips its L2 norm to `DP_CLIPPING_NORM`, computes the sample-weighted
mean of clipped deltas, and adds Gaussian noise before updating the global
model. Each full update vector is clipped as one vector, so the bias is also
included in its L2 norm. The default experiment uses 10 rounds, clipping norm `1.0`, noise
multiplier `2.0`, and delta `1e-5`. The privacy/utility sweep uses noise
multipliers `0.5`, `1.0`, `2.0`, and `3.0`, with three independent noise
repetitions per multiplier.

The RDP accountant uses replace-one-client adjacency and assumes the client
roster, public client sample counts, and data-independent participation/dropout
schedule are fixed. Per round, the weighted clipped update sensitivity is
bounded by `2 * clipping_norm * max(active client sample weight)`. Gaussian
RDP is composed over rounds without subsampling amplification; epsilon is the
maximum across clients. A client contributes no update cost in rounds when it
does not participate.

For each participant and round the accountant uses Gaussian RDP
`alpha * (q_i / q_max)^2 / (2 * noise_multiplier^2)`, where `q_i` is its
active-round sample weight and `q_max` is the maximum active client weight.
The reported epsilon uses the minimum RDP-to-`(epsilon, delta)` conversion
over the configured orders from 1.25 through 256. This is a conservative
finite-order upper bound under the stated assumptions, not an exact epsilon.

The per-model epsilon reported for a noise setting covers one DP-FedAvg
training run. Since the sweep also releases utility measurements from three
runs at each of four noise settings, `privacy_accounting.csv` includes a
`joint_sweep` composition row for those 12 releases. In this run the joint
maximum-client bound is epsilon `143.6509` at delta `1e-5`, substantially
larger than the per-model epsilon. The result directory also includes
non-private FedAvg and centralized comparators; therefore the directory as a
whole is not differentially private.

The stated accounting is conditional on the fixed, already-preprocessed
client shards. Imputation/scaler statistics are fit centrally and are not
privatized, so this is not end-to-end privacy for raw patient records unless
those statistics are public or separately protected. NumPy's
pseudorandom-number generator is used for Gaussian noise, not a production
cryptographic randomness source. Data partitioning and local training use
fixed seeds; DP noise is intentionally not seeded so the reported noise
mechanism is not made reproducible by a public seed. Consequently, DP metrics
vary across runs.

Therefore, interpret epsilon/delta as theoretical accountant bounds for an
ideal Gaussian mechanism applied to fixed preprocessed client shards. This
code is not a formally certified or production privacy implementation; do not
claim an end-to-end or cryptographically enforced privacy guarantee.

Secure aggregation is not implemented. Clients are simulated in one Python
process, and this small dataset/test split is not sufficient to support
clinical claims.

## Experimental results

The results below are the actual output of the run directory
`results/run_20261101_192937/`. The same held-out test set (76 rows) is used
throughout. Precision, recall, and F1 use class 1 as the positive class.
DP sweep utility results are means across three DP-noise trials; the default
noise-2.0 model is also reported separately because it is a single saved
artifact. Epsilon/delta for non-DP methods are not applicable (they have no
finite bound under this accounting).

### Comprehensive method comparison

| Approach | Accuracy | Precision | Recall | F1 | Epsilon | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Local-only baseline | 0.8586 | 0.8469 | 0.8714 | 0.8497 | — | — |
| Centralized | 0.8816 | 0.8611 | 0.8857 | 0.8732 | — | — |
| FedAvg | 0.8553 | 0.8333 | 0.8571 | 0.8451 | — | — |
| DP-FedAvg, noise 2.0 (saved single run) | 0.6842 | 0.6667 | 0.6286 | 0.6471 | 8.8376 | 1e-5 |
| DP-FedAvg, noise 0.5 (3-run mean) | 0.6667 | 0.6369 | 0.7143 | 0.6673 | 50.3506 | 1e-5 |
| DP-FedAvg, noise 1.0 (3-run mean) | 0.7807 | 0.7607 | 0.7905 | 0.7611 | 20.7565 | 1e-5 |
| DP-FedAvg, noise 2.0 (3-run mean) | 0.6579 | 0.6145 | 0.6667 | 0.6369 | 8.8376 | 1e-5 |
| DP-FedAvg, noise 3.0 (3-run mean) | 0.5702 | 0.5124 | 0.5524 | 0.5271 | 5.6359 | 1e-5 |

### Noise multiplier and privacy/utility

The DP utility values below are means across three DP-noise runs; accuracy
standard deviation illustrates run-to-run variation. For each fixed noise
multiplier the per-model epsilon is unchanged across repeats.

| Noise multiplier | Epsilon (`delta=1e-5`) | Mean accuracy | Accuracy SD |
|---:|---:|---:|---:|
| 0.5 | 50.3506 | 0.6667 | 0.1712 |
| 1.0 | 20.7565 | 0.7807 | 0.0331 |
| 2.0 | 8.8376 | 0.6579 | 0.1335 |
| 3.0 | 5.6359 | 0.5702 | 0.1692 |

The non-private FedAvg comparison has accuracy `0.8553`; it has no finite DP
epsilon under this experiment. Utility is noisy and does not have to decrease
monotonically in this small, three-repeat experiment.

Increasing noise decreases the accountant's epsilon for this fixed number of
rounds. Measured mean accuracy was nonmonotonic across the four settings in
this run, illustrating the variability of only three noisy repetitions rather
than a stable utility trend. The noise-2.0 single saved model also need not
equal its three-run mean.

The per-model epsilon values (5.64 to 50.35) and joint-sweep epsilon (143.65)
are high; these settings illustrate accounting and utility behavior but do
not demonstrate a strong practical privacy budget.

### Client count and dropout

| Clients (no dropout) | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 2 | 0.8553 | 0.8333 | 0.8571 | 0.8451 |
| 4 | 0.8553 | 0.8333 | 0.8571 | 0.8451 |
| 6 | 0.8553 | 0.8333 | 0.8571 | 0.8451 |
| 8 | 0.8684 | 0.8571 | 0.8571 | 0.8571 |

For four clients, the no-dropout and client-3-dropout runs both scored
accuracy `0.8553`, precision `0.8333`, recall `0.8571`, and F1 `0.8451`.
Thus, this particular single-round dropout did not change the final metrics
at the displayed precision.

These are single partition/training configurations and one fixed test split.
The test set was not used for fitting or preprocessing, but repeated reporting
on the same small test set can make comparisons optimistic; independent
replicated splits and confidence intervals are future work. The Local-only
baseline trains each model on one shard but currently shares the train-fitted
imputation/scaling statistics, so it is not an end-to-end isolated-client
privacy baseline.

### Graph/CSV correspondence

The six PNGs are generated directly from the CSV-backed metric frames:

- `model_comparison.png` uses `model_comparison.csv` (all four approaches and
  all four classification metrics).
- `performance.png` uses `federated_metrics.csv` (standard FedAvg metrics by
  round; dropout is reflected in active-client counts).
- `privacy_epsilon_vs_noise.png` uses per-model epsilon in
  `privacy_utility_metrics.csv`.
- `privacy_accuracy_vs_noise.png` uses DP accuracy means and the no-DP
  FedAvg reference in `privacy_utility_metrics.csv`.
- `scalability_comparison.png` uses `scalability_metrics.csv`.
- `dropout_comparison.png` uses `dropout_metrics.csv`.

The source audit confirmed that each plotted series is built from the
corresponding CSV frame; PNG files were also checked as readable image
artifacts.

## Run artifacts

Each timestamped results directory contains:

- CSV: `model_comparison.csv`, `federated_metrics.csv`,
  `dp_federated_metrics.csv`, `privacy_accounting.csv`,
  `privacy_utility_metrics.csv`, `privacy_utility_runs.csv`,
  `scalability_metrics.csv`, and `dropout_metrics.csv`.
- PNG: `model_comparison.png`, `performance.png`,
  `privacy_epsilon_vs_noise.png`, `privacy_accuracy_vs_noise.png`,
  `scalability_comparison.png`, and `dropout_comparison.png`.
- Models: `global_model.npz`, `centralized_model.npz`,
  `dp_global_model.npz`, and one `dp_global_model_noise_*.npz` per sweep
  multiplier. Each artifact includes feature order and preprocessing values.

The older CSV/model/plot files directly under `results/` are legacy outputs
from an earlier experiment schema (for example, their model comparison has
only three methods). The timestamped run directory is the authoritative
complete result set.

## Interactive demonstration dashboard

Install the project dependencies and launch the Streamlit app from the project
directory:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The sidebar first offers exactly two dataset types: UCI Cleveland Heart Disease
and Cardiovascular Disease (`cardio_train.csv`). It identifies compatible runs
from the saved model's exact ordered feature schema; when multiple compatible
runs exist, a separate training-run selector is shown. The selected dataset,
record counts, feature count, and run name are displayed in the sidebar. A
dataset without a compatible saved model produces an error rather than falling
back to another dataset's model. The dashboard reads metrics and model
artifacts from the selected run; it does not retrain models or overwrite
results. Prediction inputs follow that run's model feature order and use its
saved imputation and scaling parameters. Privacy, scalability, and dropout
results are shown only when the selected run includes those artifacts.
Predictions are for educational demonstration only, not medical diagnosis.

## Training on the cardio dataset

The project also supports the 70,000-row `cardio_train.csv` dataset, which has a
different schema from Cleveland. Train a separate centralized and federated
model bundle by passing the ZIP that contains `cardio_train.csv`:

```powershell
python train_cardio.py --dataset-zip "C:\path\to\archive.zip"
```

The script validates the archive schema, excludes rows with implausible height,
weight, or blood-pressure combinations, makes a stratified 75/25 split, and
fits imputation and scaling on training rows only. Each run is saved to a new
`results/cardio_run_*` directory, which the dashboard can select without
replacing the Cleveland results. The cardio model uses these 11 features in
order: `age_years, gender, height_cm, weight_kg, systolic_bp, diastolic_bp,
cholesterol, glucose, smoke, alcohol, active`. Class 1 is cardiovascular
disease. Cardio runs include local-only, centralized, FedAvg, and DP-FedAvg
comparisons, a four-noise privacy/utility sweep, and client-count and dropout
experiments. DP accounting is conditional on the project's documented
assumptions; preprocessing statistics are not privatized, and these results
are not a certified end-to-end privacy guarantee.

## IID versus Non-IID FedAvg experiment

Run the dedicated comparison against the same cardio dataset and the shared,
fixed stratified train/test split:

```powershell
python run_iid_noniid.py --dataset-zip "C:\path\to\archive.zip"
```

By default, the experiment uses four simulated clients, the existing FedAvg
model and round count, three independent training/partition seeds (`42`, `43`,
and `44`), and Dirichlet alpha values `0.1`, `0.5`, `1.0`, and `10.0`. Configure
the repetition, alpha, client, or round counts with `--seeds`, `--alphas`,
`--clients`, and `--rounds`. At least two unique seeds are required. IID uses
random shuffled, balanced splits; the Non-IID partitioner draws per-class
client proportions from a Dirichlet distribution and redistributes overflow
to maintain balanced sample counts. Every training row is assigned once.

Smaller alpha generally yields more heterogeneous client label distributions;
larger alpha generally yields more similar distributions. This is a statistical
tendency, not a guarantee of monotonic measured model performance. The same
held-out test records and training-only imputation/scaling are reused for every
condition. The IID/Non-IID trials train independently from scratch with matching
model settings and no dropout or differential-privacy noise. Client-local model
scores are evaluated on that same held-out test set. The analysis studies
heterogeneity and does not provide additional privacy protection.

Each run creates a new `results/iid_noniid_run_*` directory without replacing
previous results. It contains per-seed metrics, mean/standard-deviation
summaries, round histories, client-wise distribution and local-model metrics,
a partition audit with the shared test-set hash, and five plots:
`iid_vs_noniid_performance.png`, `client_class_distribution.png`,
`accuracy_across_rounds.png`, `f1_across_rounds.png`, and
`performance_by_alpha.png`. The dashboard's **IID vs Non-IID Analysis** page
reads these saved results; it does not start training.

## Limitations and future work

This remains a simulation on one historical dataset. The standard baseline
bundle uses one seed; the IID/Non-IID study repeats partitioning and training
across three seeds but intentionally holds its split fixed for a controlled
comparison. Further work should include larger independent datasets, repeated
train/test splits, confidence intervals, client-level validation, secure
aggregation, production-grade private randomness, end-to-end DP treatment of
preprocessing, and strict per-client preprocessing for the local-only baseline.
Secure aggregation is a future direction, not an implemented feature. Do not
use these results for clinical decision-making.
