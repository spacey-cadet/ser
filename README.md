# ser

Speech emotion recognition service — WavLM backbone + a small
trainable classifier head, built for use inside a chat app: decode →
VAD → embed → calibrate → cascade → session-aware emotion state.

Two deployment targets exist for this repo, sharing the same model and
serving code:

| Branch | Target | Cost | Status |
|---|---|---|---|
| `main` | HF Spaces / local docker (self-hostable) | free | original target |
| `aws-serverless` | AWS Lambda + S3 + DynamoDB, Kaggle-orchestrated retraining | ≤ $5/month | build in progress |

Only `infra/`-level deployment config and a handful of pluggable
storage backends differ between them — model code, calibration,
cascade logic, and consent handling are identical on both branches by
design.

## Where to start

- **Understand the inference pipeline itself** (VAD, calibration,
  cascade, drift, champion/challenger, review queue, consent):
  [`PIPELINE_README.md`](./PIPELINE_README.md). Read this first — it's
  the source of truth for what each `app/` module actually does.
- **Training recipe and numerics** (how the checkpoint is produced,
  what each hyperparameter does, precision/stability notes): the
  [Training](#training-offline-stage-1) and
  [Numerical computing notes](#numerical-computing-notes) sections below.
- **Deploying to AWS**: start with
  [`infra/aws-serverless/README.md`](./infra/aws-serverless/README.md),
  then [`infra/aws-serverless/SETUP_ORDER.md`](./infra/aws-serverless/SETUP_ORDER.md)
  for the actual dependency-ordered setup sequence. The full design
  rationale (why each AWS service was picked, cost breakdown, what was
  deliberately left out) is in
  [`docs/decisions/project3-ser-aws-execution-plan.md`](./docs/decisions/project3-ser-aws-execution-plan.md).
- **Retraining loop internals** (policy-gated triggering, Kaggle
  orchestration, the champion/challenger promotion gate): see
  `scripts/check_retrain_policy.py`, `scripts/promotion_gate.py`, and
  `kaggle/train_head_kernel.py` — each has a docstring explaining its
  role in the loop, not just its arguments.

## Architecture, one level down

```
audio in ─▶ decode ─▶ VAD ─▶ WavLM ─▶ stats pooling ─▶ MLP head ─▶ calibration
                                                                       │
                                                                 cascade + session state
                                                                       │
                                                                 prediction out
```

At **serving time** the backbone is frozen (`eval()`, `no_grad`), and in the
**routine retraining loop** only the MLP head is trained. That is why routine
retraining doesn't need a GPU — see the AWS README for why that matters for
cost. The *initial* checkpoint, however, is produced by a one-off GPU run that
also adapts the backbone (Stage 1 below). Both stages are documented
explicitly so nobody assumes the backbone weights are the stock
`microsoft/wavlm-base-plus` ones.

| Stage | What trains | Hardware | Frequency | Entry point |
|---|---|---|---|---|
| 1. Backbone + head fine-tune | WavLM transformer (CNN feature encoder frozen) + MLP head | Kaggle T4 | once / on major data change | `notebooks/speechbrain-wavlm-ser.ipynb` (`train.py`, `train.yaml`) |
| 2. Head-only retrain | MLP head on cached, frozen embeddings | CPU | routine, policy-gated | `kaggle/train_head_kernel.py` |

## Training (offline, Stage 1)

### Data

| Item | Value |
|---|---|
| Corpora | RAVDESS, CREMA-D, IEMOCAP, SAVEE, EMO-DB, TESS (flattened to `dataset_emotion_number.wav`) |
| Usable clips | 17,854 after filtering (0.1 s ≤ duration ≤ 30 s) |
| Classes (8) | angry, calm, disgust, fearful, happy, neutral, sad, surprised |
| Label remapping | IEMOCAP `fru`→angry, `exc`→happy, `bor`/`oth`→neutral, `xxx`→dropped |
| Split | 80 / 10 / 10, stratified by emotion, `random_state=42` → 14,283 / 1,785 / 1,786 |
| Sample rate | 16 kHz mono |

**Known limitation — speaker leakage.** The split is stratified by *emotion*,
not by *speaker*. The same actors appear in train, valid and test, so the
reported validation/test accuracy is optimistic relative to unseen speakers.
Treat it as an in-distribution number. A speaker-independent or
leave-one-corpus-out split is the honest generalisation estimate and is on the
roadmap (it requires carrying a speaker id through `prepare_data.py`; the
current filename scheme discards it).

### Model

| Component | Config | Params |
|---|---|---|
| Backbone | `microsoft/wavlm-base-plus`, 12 layers, 768-d, last hidden state, output layer-norm on | ≈ 94.4 M (≈ 90.2 M trainable; CNN feature encoder frozen) |
| Pooling | `StatisticsPooling` → concat(mean, std) over time → 1536-d | 0 |
| Head | Linear(1536→256) → ReLU → Dropout(0.3) → Linear(256→8) | 395,528 |
| Total (SpeechBrain log) | 94.8 M params, 90.6 M trainable | |

Frame rate is 50 Hz (320-sample stride), so a 3 s clip is ≈ 149 frames.
Self-attention cost is O(T²) in frames, which is what bounds batch size on a
16 GB T4.

### Objective

Class-weighted negative log-likelihood on `log_softmax(logits)`:

```
w_c = N / (K · n_c)          N = #train clips, K = 8, n_c = #train clips of class c
L   = Σ_i w_{y_i} · (−log p_{y_i}) / Σ_i w_{y_i}      (PyTorch weighted-mean reduction)
```

Resulting weights: angry 0.48, happy 0.68, neutral 0.75, sad 0.82, fearful
1.32, disgust 1.35, surprised 4.54, calm 5.81. `calm` (RAVDESS only) and
`surprised` are rare, hence the large weights. Because of the weighted-mean
reduction, the *scale* of the weights does not change the effective step size,
only their ratios do.

### Optimisation

| Setting | Value | Notes |
|---|---|---|
| Optimiser | AdamW, weight decay 0.05 | |
| LR | 5e-6 (single group) | See "Learning rate" below |
| LR schedule | NewBob: `lr ← 0.8·lr` when relative val-error improvement < 0.0025 (patience 2) | `lr_final` in the YAML is currently unused by NewBob |
| Batch size | 4 | Limited by memory with uncapped clip length |
| Epoch cap | 40 | |
| Early stopping | patience 5 epochs, min delta 0.001 error rate | |
| Precision | fp16 autocast + dynamic `GradScaler` | |
| Grad clipping | SpeechBrain default, global norm 5.0 | |
| Model selection | lowest validation error rate | checkpoint pushed to the HF Hub each epoch |
| Seed | 1986 | `torch.manual_seed` only; see reproducibility note |

**Learning rate.** 5e-6 is a sensible rate for fine-tuning pretrained
transformer weights but is very small for a *randomly initialised* head. If
convergence of the head is slow, use two parameter groups (backbone ~1e-5,
head ~1e-3) instead of a single rate.

### Augmentation (train split only)

- **Speed/pitch perturbation:** factor ∈ {0.9, 1.0, 1.1} drawn per batch,
  implemented as `resample(orig=16000·f, new=16000)`. Factor 0.9 stretches the
  clip by 1/0.9 ≈ 1.11×; 1.1 shortens it to ≈ 0.91×. Relative lengths
  (`wav_lens`) are unaffected.
- **Additive Gaussian noise:** σ = 0.005 with p = 0.5. This is an *absolute*
  amplitude, so effective SNR varies with clip loudness. An SNR-targeted noise
  level is preferable.

### Observed training behaviour

| Quantity | Value |
|---|---|
| Throughput | ≈ 4.9 it/s at batch 4 (≈ 20 clips/s), 3,571 steps ≈ 12 min / epoch on a T4 |
| Validation pass | 447 steps ≈ 1 min |
| Checkpoint upload | ≈ 1.1 GB (of which optimizer state ≈ 725 MB), ≈ 15 s |
| Epoch 1 | train loss 0.407, valid loss 1.03, valid accuracy 0.7826 (error 0.2174) |
| Full 40-epoch run | ≈ 9 h, i.e. near or over typical Kaggle session limits → rely on resume from the HF checkpoint |

Epoch-1 numbers are from a single run and are **in-distribution** (see the
speaker-leakage note). Final test metrics should be reported as macro-F1 and
UAR (unweighted average recall), not accuracy alone — see
[Evaluation philosophy](#evaluation-philosophy).

### Resuming

Each epoch the best checkpoint (`wavlm.ckpt`, `classifier.ckpt`,
`normalizer.ckpt`, `optimizer.ckpt`, `counter.ckpt`, …) is pushed to the HF
repo and restored at the top of the notebook. A checkpoint whose classifier
state-dict does not match the current head (e.g. the old single `Linear`
`w.weight/w.bias` vs. the current MLP `net.*`) is detected on load and
training restarts from epoch 1.

## Numerical computing notes

- **Mixed precision.** The forward pass runs under fp16 autocast on the T4.
  fp16 has ≈ 3 decimal digits of precision and a max of 65,504, so the loss
  scaler starts at 2¹⁶ and backs off on overflow (skipped steps are normal
  early in training). Keep reductions that are sensitive to range in fp32:
  `log_softmax`, and ideally the statistics pooling (cast features with
  `.float()` before computing mean/std).
- **Statistics pooling.** Mean and (unbiased) std over time. The std is
  undefined for a single frame (`T = 1` → NaN) and its gradient blows up near
  zero variance, so use `std = sqrt(var + eps)` with eps ≈ 1e-5 and mask padded
  frames using the relative lengths in a padded batch. At batch size 1 (the
  serving case) there is no padding to mask.
- **Output normalisation.** `output_norm: True` applies a layer-norm over the
  whole `[T, H]` slab of each utterance. This is part of the *model* and must
  be reproduced at inference. Skipping it silently shifts the pooled-feature
  scale and degrades accuracy without raising any error.
- **Input normalisation.** WavLM-base-plus is trained without per-utterance
  waveform normalisation; keep the same `Wav2Vec2FeatureExtractor`/SpeechBrain
  setting at train and serve time.
- **Padding and batch composition.** Clips of 0.1–30 s in one batch waste
  compute on padding (cost is quadratic in the longest clip). Cap the
  training crop length (e.g. 6–8 s) and bucket by duration to raise the usable
  batch size, then use gradient accumulation to reach a larger effective batch.
- **Resampling ratios.** Speed perturbation is a rational resample; 16000/14400
  and 16000/17600 reduce to small integer ratios, so the polyphase filter is
  cheap and exact. Arbitrary non-rational factors fall back to slower paths.
- **Loss reduction.** With class weights, PyTorch normalises by Σ w_{y_i}, not by
  the batch size. Per-batch loss therefore fluctuates with which classes land
  in the batch; compare epochs using the validation loss on the *same* fixed
  set, never the raw training-batch loss.
- **Reproducibility.** Only `torch.manual_seed` is set. Python `random` (speed
  factor, noise gate), NumPy, and cuDNN nondeterminism are not controlled.
  Use `speechbrain.utils.seed.seed_everything(seed)` and log the seed with each
  run if bit-level repeatability matters; otherwise treat ±0.5–1 pt variation
  between runs as noise.
- **Calibration.** Class weighting and label-noisy corpora push the softmax
  away from calibrated probabilities. Fit temperature scaling on the validation
  set (`scripts/calibration_fit.py`) and report ECE alongside macro-F1.

## Human-in-the-loop data collection

The model's training data is otherwise static. Low-confidence,
consented predictions get queued for a human to label
(`app/logging_pipeline/review_queue.py`, `consent.py`); labeled batches
periodically retrain the classifier head (Stage 2), gated by an explicit
promotion policy rather than auto-deployed. This closes the loop
without needing to re-collect a dataset from scratch — see
`docs/decisions/project3-ser-aws-execution-plan.md` for the full
data-flow diagram on the AWS branch specifically.

## Evaluation philosophy

Aggregate accuracy flatters easy classes and hides the confusions that
actually matter in a chat context (e.g. Sad/Neutral). `scripts/eval_report.py`
reports PR-AUC and per-class F1/recall alongside accuracy — that's what
to read in any eval, not the headline accuracy number. It's a pure
reporting step by design; `scripts/promotion_gate.py` is the separate
piece that turns two of its reports (champion vs. challenger, scored
against an identical pinned held-out set) into an actual promote/reject
decision, currently gated on macro F1 with a 0.01 minimum improvement
and a per-class recall floor.

Recommended headline metrics for the offline model: **UAR** (mean per-class
recall), **macro-F1**, per-class recall, and a confusion matrix; accuracy only
as a secondary number. Evaluate with the *same* code path used in serving
(`app/model/loader.py`) so train/serve parity bugs (e.g. missing output
normalisation) cannot hide in a separate evaluation script.

## Repo layout

```
app/                  inference pipeline — shared across both branches
  model/               loader.py, classifier.py
  logging_pipeline/     feature_log.py, review_queue.py, consent.py
  session/              state.py
  routing/               canary.py (champion/challenger routing)
scripts/               eval_report.py, champion_challenger_eval.py,
                        calibration_fit.py, drift_check.py,
                        promotion_gate.py, check_retrain_policy.py,
                        assemble_kaggle_batch.py, trigger_kaggle_kernel.py
kaggle/                training kernel + metadata (AWS branch only)
notebooks/             Stage-1 training notebook (SpeechBrain + WavLM)
infra/aws-serverless/   Terraform, Dockerfile, deployment docs (AWS branch only)
alerts/                webhook.py
docs/decisions/         ADRs — what changed between branches and why,
                        not just the vendor-name swap
.github/workflows/     ci-eval-gate.yml, drift-check.yml,
                        retrain-policy-check.yml, retrain.yml,
                        deploy-aws-serverless.yml
```

## Contributing to either branch

Changes to `app/model/`, `app/logging_pipeline/consent.py`, or anything
in the core inference path should work identically on both branches —
if a change only makes sense on one, it belongs in `infra/` or a
branch-specific backend adapter, not in shared code. See
`app/logging_pipeline/README_INTEGRATION.md`-style notes in the AWS
docs for how the pluggable storage backends are meant to be added
without touching shared logic.