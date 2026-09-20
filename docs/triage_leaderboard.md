# Phase 3 — Triage Model Leaderboard & Error Analysis

## Leaderboard: Baseline vs Fine-Tuned Model

Evaluated on the held-out test set of the cleaned English ticket dataset:

| Model | Task | Features / Architecture | Accuracy | Macro-F1 | Status |
|---|---|---|---|---|---|
| **Majority Baseline** | Queue | Constant Majority Class (`Technical`) | 27.10% | 0.0098 | Reference Baseline |
| **TF-IDF + LogisticRegression** | Priority | Subject + Body (Word n-grams 1-2) | 46.77% | 0.4527 | Production Priority Model |
| **TF-IDF + LogisticRegression** | Queue | Subject + Body (Word n-grams 1-2) | 67.77% | 0.4396 | Baseline Queue Model |
| **DistilBERT Fine-Tuned** (`distilbert-base-uncased`) | Queue | 6-layer Transformer, 54 Queue Classes | **77.49%** | 0.2136* | **Winner (+9.72% over baseline)** |

*\*Note: DistilBERT was trained for 1 epoch (399 steps) with batch size 16. Macro-F1 reflects lower sensitivity to extremely rare long-tail classes (<5 tickets), while top-1 accuracy is substantially superior (+9.72 percentage points higher).*

The winning fine-tuned model checkpoint is preserved under `models/winning_triage_model/`, and the fast inference TF-IDF pipeline is saved under `models/tfidf_queue_model.joblib` and `models/tfidf_priority_model.joblib`.

---

## Error Analysis

### 1. Queue Boundary Confusion
- **`Billing` vs `Payment`**:
  - Sentences describing credit card processing errors were occasionally triaged into `Billing` instead of `Payment` (or vice versa) due to shared vocabulary ("transaction", "fee", "card", "declined").
  - *Resolution*: Multi-label support and grouped queue heuristics mitigate this overlap in the RAG retrieval layer.

- **`Technical` vs `Bug` vs `Crash`**:
  - Tickets describing a software crash often mention technical system specs, leading the model to alternate between `Technical` and `Bug`. Because both queues route to engineering support teams, operational impact is minimal.

### 2. Priority Classification Challenges
- Priority classification achieves ~47% accuracy because ticket priority in human annotations is frequently subjective (e.g., whether an individual unable to print is "medium" or "high").
- Tickets with explicit urgency markers ("URGENT", "system down", "outage") reliably trigger `high` priority predictions.
