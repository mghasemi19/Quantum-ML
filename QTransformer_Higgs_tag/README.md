# H -> bb vs QCD: hybrid quantum-attention Transformer encoder

<img width="1195" height="896" alt="QTransformer_img" src="https://github.com/user-attachments/assets/c15847a1-03d1-4774-8670-f390de344787" />


An end-to-end **educational, runnable** jet-constituent classifier inspired by
Smaldone et al., *A Hybrid Transformer Architecture with a Quantized Self-Attention
Mechanism Applied to Molecular Generation* (arXiv:2502.19214v2).


## 1. Quick start

Python 3.10+; from this folder (install `requirements-notebook.txt` instead
of `requirements.txt` if you need to install JupyterLab too):

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m hbbqt.demo_hadamard
python -m pytest -q
python -m hbbqt.train --smoke --output runs/smoke
```

Larger toy experiment, train both independent models:

```bash
python -m hbbqt.train --n-jets 1200 --max-constituents 16 --epochs 12 \
    --model both --output runs/toy_1200
```

Use CUDA if available by default; force CPU with `--device cpu`.
The CLI defaults to four PyTorch CPU threads (`--cpu-threads 4`) to prevent
oversubscription for small state-vector operations. For finite-shot
**evaluation only**: `--eval-shots 2048`. Default is exact state-vector
expectation, including during training. For a shorter run use `--smoke`.

## 2. The model and exact relationship to the paper

```
per-jet particles [L,5] -> sort by descending pT -> padding mask
                           |                       |
              feature-to-angle layer       learned pT-rank position angles
                   3 Ry + CNOT                3 Ry + CNOT
                        |                           |
                    |e_i>          tensor          |p_i>
                             |z_i> (6 qubits, 64 real amplitudes)
                                    |
                         +----------+----------+
                         |                     |
                      U_q (Ry+CNOT)        U_k (Ry+CNOT)
                         |                     |
                       |q_i>                 |k_j>
                         +----------+----------+
                                    |
             real <q_i|k_j> = ancilla <Z> of modified Hadamard test
                                    |
                    ALL valid i,j; multiply by sqrt(64)
                                    |
                key padding mask -> row softmax -> A
                                    |
                         A @ V_classical
                                    |
                  output proj + residual + norm
                         FFN + residual + norm
                                    |
                  masked mean over valid particles
                                    |
                      MLP -> Higgs logit
```

Preserved from paper: independent token and positional registers (3+3 qubits),
trainable Ry and CNOT ansatzes, tensor-product input, separate learned query and
key unitaries, their **real inner product** as quantum attention score, paper's
`sqrt(d)` score multiplication, classical values, and downstream classical FFN.

Changed for jets: particles have continuous five-vectors, so a trainable
`Linear(5,3)` plus sigmoid maps each constituent's features to token angles in
(0,pi). Learned positional angles (initialized to zero) encode **descending pT
rank** rather than word positions. The five input observables include physical
eta/phi offsets so angular information is not lost. Since this is a classifier,
no causal mask, autoregressive decoding, or next-token loss. Instead, use
bidirectional attention, masked mean pooling, and BCE-with-logits binary loss.
PADDED keys and queries do not contribute; there is NO CLS token.

The classical baseline uses a standard single-head, one-layer PyTorch
MultiheadAttention encoder with the same d_model, FFN, pooling and classifier.
**It is not parameter-count matched**; do not interpret a single-run difference
as evidence of quantum advantage.

### Hadamard test fidelity and limitations

The separate `python -m hbbqt.demo_hadamard` example explicitly implements the
**seven-qubit construction**: six working qubits, plus an ancilla represented as
two amplitude branches. It applies the inverse query and inverse embedding
circuits only on ancilla=1, then the key preparation on that branch, then the
final Hadamard. Unit tests verify the resulting `P(0)-P(1)` equals `q dot k`.

The batched training path directly uses the **mathematically identical** exact
real-state overlap `q @ k.T` instead of executing O(L^2) controlled quantum
circuits. `hadamard_probabilities` computes explicit `(q+k)/2` and `(q-k)/2`
branches. This shortcut supplies conventional PyTorch autograd through the
state-vector simulator; the paper's CUDA-Q and SPSA training are NOT reproduced.
Optional `--eval-shots` samples Bernoulli ancilla measurements for a separate
post-training evaluation; it does NOT model gates, noise, errors, QPU runtime,
or quantum-hardware gradients. Real circuit sampling costs grow with precision.
No quantum speedup or practical advantage is demonstrated by this code.

The paper's positional token order is intrinsic to SMILES. Jets are sets, so
assigning ranks introduces an inductive bias; sorting by pT makes results
invariant to reordering the original input rows barring ties. It does NOT make
the architecture invariant to arbitrary changes in physical constituents.

## 3. Toy constituent data

Labels: `y=1`: illustrative boosted Higgs -> bb-like two-prong jets;
`y=0`: QCD-like one-core jets with occasional secondary radiation.
No truth-level b quarks, QCD matrix elements, proper parton showers, calorimeter
response, b tagging, pileup, or jet mass selection. The name Hbb is only a toy
class label. Constituents are sorted by pT; each jet has up to L particles.

Five features, precisely in this order:

1. `log_pt_fraction = log(pT_i / sum_j pT_j)`
2. `delta_eta = eta_i - eta_jet`
3. `delta_phi = wrap(phi_i - phi_jet)` in radians
4. `log_energy_fraction = log(E_i / sum_j E_j)`
5. `charge`, e.g. -1, 0, +1

Feature normalization mean and std are fitted ONLY to valid training
constituents, saved in `preprocessing.npz`, applied to validation/test and
padded rows set to zero. Train/val/test are disjoint 70/15/15% stratified splits
in toy mode and without groups. Random seed is controlled.

## 4. Replace toy data with real jets

Supply one file, `jets.npz`:

| Key | Shape / meaning |
|---|---|
| `x` | float `[N,L,5]`, UNSTANDARDIZED feature order above |
| `mask` | boolean `[N,L]`, True for valid particles |
| `y` | integer `[N]`, 0=QCD, 1=H->bb |
| `group_id` | optional integer/string `[N]`, event identifier to group jets in splits |

String `group_id` is supported as fixed-width NumPy strings, not object arrays.
This avoids identical-event jets in separate splits. All jet arrays must
contain both labels in every split. Input rows are re-sorted using column 0
(log pT fraction); only the highest-pT `--max-constituents` are kept.

```bash
python -m hbbqt.train --data-npz jets.npz --model both \
    --max-constituents 16 --epochs 15 --output runs/real_jets
```

For upstream per-particle arrays `pt, eta, phi, energy, charge, mask` of shape
`[N,L]`, `y[N]`, and optional `group_id[N]`, use:

```bash
python examples/from_particle_arrays.py raw_particles.npz jets.npz
```

**Physics checklist before publishing:** use truth-identified Higgs->bb jets
and a defined QCD mixture with proper pT, eta and mass selection; avoid sample
origin/data-taking period leakage; split by event or generator source as needed;
check label and feature provenance; compare spectra and use matched or reweighted
kinematics; handle weights and class priors consistently; report mass sculpting,
rejection versus H efficiency, independent systematic and statistical
uncertainties and independent seeds. This teaching project does not implement
sample reweighting, detector calibration, b-tag baselines, uncertainties or
mass decorrelation. It is therefore **not a publishable benchmark out of box**.

## 5. Outputs

`runs/<name>/` contains:

- `config.json`: parameters and explicit data provenance warning.
- `preprocessing.npz`: normalization and held-out split indices.
- `quantum/` and/or `classical/`: `best_model.pt`, `history.json`,
  `test_metrics.json`, `test_predictions.npz`, `loss.png`, `attention_example.png`.
- `summary.json`, `test_roc.png`.
- Optional `quantum/shot_metrics.json` with finite-shot **post-training** metrics.

Models are selected using validation AUC; the held-out test set is evaluated
after checkpoint selection. `test_metrics.json` includes ROC AUC, accuracy,
confusion matrix, QCD efficiency/rejection at at least 50% or 80% Higgs
efficiency where finite-sample ROC supports those values. A zero observed QCD
efficiency gives a `null` rejection instead of pretending infinite evidence.

## 6. Files

```
hbbqt/data.py            generator, NPZ import, event-safe splits, scaler
hbbqt/quantum.py         differentiable R_y/CNOT state vector + attention
hbbqt/demo_hadamard.py   explicit inverse / reprepare Hadamard-test demonstration
hbbqt/model.py           hybrid encoder and classical baseline
hbbqt/metrics.py         binary discrimination metrics
hbbqt/train.py           train, validation selection, held-out test, plots
examples/from_particle_arrays.py   raw physics arrays -> model features
tests/                   numerical and end-to-end correctness checks
notebooks/Walkthrough.ipynb        interactive mini-tutorial
```

## 7. Cost and methodological caveat

The six-qubit state vector has 64 amplitudes per constituent; standard
state-vector simulation materializes these arrays and calculates all pairs on
a CPU/GPU. Exact matrix products are used for speed. This is a faithful
mathematical *attention score*, NOT faithful quantum-device cost accounting.
The study in the provided paper considers O(n^2 log d) **under assumptions of
state preparation and ideal overlap evaluation**, while its classical V
multiplication remains O(n^2 d) and shot-based estimation has additional
precision overhead. Neither their complexity bound nor hardware acceleration
may be claimed from this PyTorch demonstration.

## 8. Apply the trained classifier to new jets

For labeled or unlabeled NPZ files with `x` and `mask` (and optional `y`):

```bash
python -m hbbqt.predict --run runs/smoke --model quantum \
    --data-npz new_jets.npz --output runs/new_predictions.csv
```

Inference loads the checkpoint's saved **training-only normalization** and
feature dimensions. It does not retrain or use new labels to choose a threshold.
A fixed default probability threshold of 0.5 is exported alongside `p_hbb`;
for a physics working point, pick thresholds on independent validation data.
