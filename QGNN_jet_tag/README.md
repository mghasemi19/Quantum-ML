# Paper-inspired boosted-jet quantum graph neural network
<img width="1448" height="1086" alt="project_img" src="https://github.com/user-attachments/assets/338f36ef-154c-4cd6-8bfb-ba045599809c" />


**Status:** An implementation scaffold and independently checkable architecture,
**not** the authors' implementation or a verified reproduction of their AUCs.
The original user notebook uses 4-node cycle/path graphs and RZZ/RY layers.
We reuse its graph-to-qubit mapping, entangling-gate idea, and mean-Z readout,
but replace the architecture to follow Kangaziankangazi et al., arXiv:2605.18416v1.

## Requirements
Python 3.10+, CPU with sufficient RAM (10-qubit complex64 statevectors).

```bash
pip install -r requirements.txt
pytest -q tests
python qgnn.py --data /path/to/jets.npz --output results --samples 8000 --sinkhorn
```

The last command can be **very slow**. A 10-qubit differentiable simulation
rebuilds a circuit for each event. For a smoke test:

```bash
python make_demo_data.py
python qgnn.py --data demo_only.npz --samples 20 --pretrain-epochs 1 --quantum-epochs 1 --joint-epochs 1 --batch-size 2 --output demo_results
```

`demo_only.npz` contains random labels and random feature vectors: tests software
only and cannot establish a jet-tagging performance. No actual dataset accompanies
the uploaded files.

## Real data contract
Provide `.npz` with `jets: float32[J,P,16]`, `labels: int[J]`, Z=1,
gluon=0, **or** already-preprocessed `x: float32[J,10,160]` and labels.
The preprocessing code assumes energy feature index 3, eta index 6 and phi
index 7, **placeholders requiring verification against the real dataset's feature
schema**. All particles must be real (no padding); if padded, adapt the validity
mask and nearest-neighbor selection before running. It selects 10 highest-energy
particles and 10 nearest *other* particles by wrapped delta-phi and delta-eta,
then concatenates absolute 16-feature differences.

Training-only per-feature min/max and clipping map data to [0,1]. The paper says
features are normalized to [0,1] but does not specify all normalization details.
All train/validation/test partitions are stratified and disjoint; with 8000 jets
they have 4000/2000/2000 events, respectively. `--samples` must be divisible by four and
sufficient balanced events must exist.

## Architecture and differences from paper
- Per-particle encoder 160->128->64->4 and symmetric decoder.
- Noise generator: uniform 64D plus binary label -> 128 -> 40 -> (10,4).
  Its **exact conditioning and architecture are not specified in the paper**;
  ours is an explicit approximation.
- Reconstruction MSE + 0.001 debiased entropic OT divergence at batch level;
  Sinkhorn epsilon=0.2, iterations=15 are **ours** (unspecified in paper).
- Ten nodes and ten qubits, 3-nearest-neighbor graph in latent Euclidean space;
  union of directed edges, distance weight inverse with epsilon=1e-3.
  The paper does not fully disambiguate directed-edge symmetrization or zero
  distance handling. Discrete kNN neighbor selection is not differentiable,
  but selected edge distances and their inverse weights remain differentiable;
  gradients also flow through RX encoding angles into the encoder.
- Four rounds of RX(x) data upload, RY then RZ trainable rotations, chained CRZ.
  Paper's Eq.(3) product ordering is ambiguous; this implementation follows
  RY followed by RZ as stated in its prose.
- Edge evolution uses exp(-i*t*w*Xi*Xj) on unique undirected edges via RXX(2tw).
  First-order Trotter, single step. The paper does not give the Trotter count.
- Final trainable RY, mean Z expectation, signal score is negative expectation.
- Separate encoder pretraining and QGNN training then optional joint fine-tuning.
  BCE-with-logits is an explicit choice; paper does not specify classifier loss.
- No ParticleNet baseline, architecture search, five-fold CV, GPU optimization,
  experimental error model, or exact reproduction of paper results is included.
  The paper's Fig. 2 and Fig. 6 compare to ParticleNet, so a full reproduction
  requires separately implementing its correct network and data protocol.

### Source results (comparison targets, not measured here)
Standalone latent QGNN AUC 0.683 +/- 0.013; latent ParticleNet 0.738 +/- 0.024;
joint hybrid without Sinkhorn 0.825 +/- 0.019; joint Sinkhorn hybrid 0.842 +/- 0.007;
full-feature ParticleNet 0.875 +/- 0.006. They report five-fold CV uncertainty.
Do **not** present this code's single-seed AUC as a reproduced paper figure.

## Outputs
`separate.pt`, `joint.pt`, `scaler.npz`, `test_roc.npz`, `metrics.json`.
Only the joint model is tested after training; validation metrics are provided
for the separate and joint stages. The run does not select between these based
on validation, but validation can guide later experiments without examining test.

## Important research limitations
The paper cites 20,000 simulated jets but uses a subset of 8,000 for QGNN;
its precise dataset branches and class indices are not supplied. Check source
and license before using the linked dataset. Energy/eta/phi column ordering,
missing-particle masks, and feature definitions are mandatory to resolve.
The published text duplicates a sentence in Section 5 and uses `k` twice in
Eq. (3); these do not specify implementation choices. The paper also describes
conditional noise in general terms without complete implementation details.
Our differentiable Torch simulator is validated against a Qiskit circuit through
`validate_qiskit`. Your original notebook's parameter-shift-of-MSE-loss formula
is not generally correct, so this project uses autograd through a statevector.
