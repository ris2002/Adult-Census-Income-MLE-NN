# Mistakes Found & Fixed (Post-Review)

I reviewed this project a few months after completing it and found the following bugs.
I am documenting them honestly — both as a changelog and as a reference for my future self.

---

## 1. Swapped arguments in `classification_report` (all ensemble models)

**The bug:** I passed predictions first and true labels second. The signature is
`classification_report(y_true, y_pred)` — so for Random Forest, XGBoost, AdaBoost,
and Gradient Boosting, the precision and recall columns were swapped, and the
"support" column showed prediction counts instead of true class counts. Every
per-class observation I wrote was based on inverted columns. (F1 and accuracy are
symmetric under this swap, so the overall scores were unaffected.)

**Wrong:**
```python
y_pred = trained_model.predict(x_test_enc)
print(classification_report(y_pred, y_test_enc))   # arguments swapped
```

**Corrected:**
```python
y_pred = trained_model.predict(x_test_enc)
print(classification_report(y_test_enc, y_pred))   # y_true first, y_pred second
```

**How I caught it:** the support counts differed between models even though they
should all share the same test set. Support = true class counts, so it must be
identical for every model evaluated on the same split.

---

## 2. Models evaluated on different test splits

**The bug:** the Neural Network and the ensemble models were run in different
sessions with the split re-created each time, so they were not evaluated on the
same test set (test sizes differed: 3,276 vs 3,017). A model comparison is only
valid if every model sees the identical held-out test set.

**Wrong:**
```python
x_train, x_test, y_train, y_test = train_test_split(X, y)   # no seed, re-run per session
```

**Corrected:**
```python
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=76
)
# One split, fixed seed, stratified — reused for ALL models.
```

---

## 3. Threshold mentioned in training, never applied in evaluation

**The bug:** my training loop monitored accuracy at `threshold = 0.65`, and the
README discussed results at thresholds 0.6 and 0.7 — but the evaluation function
used `torch.argmax`, which for two classes is always equivalent to a 0.5 threshold.
The threshold I "used" only affected a TensorBoard progress metric on training
data; the reported test results never used it. Lesson: the threshold is part of
the decision rule at prediction time — if the evaluation function doesn't apply
it, the evaluation didn't use it.

**Wrong:**
```python
def test_model(model, X_test, y_test):
    outputs = model(X_test)
    preds = torch.argmax(outputs, dim=1)   # always = 0.5 threshold
```

**Corrected:**
```python
def test_model(model, X_test, y_test, threshold=0.5):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        probs = torch.softmax(outputs, dim=1)
        preds = (probs[:, 1] > threshold).long()   # threshold applied where it matters
        print(classification_report(y_test.cpu().numpy(), preds.cpu().numpy()))
    return preds
```

---

## 4. Meaningless ordinal encoding of `education` and `occupation`

**The bug:** I ordinal-encoded `education` and `occupation` with a default
`OrdinalEncoder`, which assigns integers alphabetically — so "10th" < "1st-4th" <
"Bachelors", an ordering with no real-world meaning. The dataset already provides
`education.num`, the correct ordinal encoding of education. `occupation` has no
natural order at all.

**Corrected approach:** dropped the `education` string column (kept
`education.num`) and one-hot encoded `occupation` along with the other nominal
features.

---

## 5. Declared preprocessing that was never implemented

**The bug:** the README stated that IQR-based outlier handling for
`hours.per.week` was "mandatory," but the notebook never implemented it.
Documentation must describe what the code actually does — declared-but-missing
steps are worse than omitted ones, because they misrepresent the pipeline.

**Fix:** implemented the step / removed the claim (whichever the final notebook
reflects).

---

## Takeaways

- Audit the boring metadata before comparing models: test set size, class counts,
  random seeds. Bugs hide in the metadata, not the metrics.
- Support counts are a free consistency check — identical test set ⇒ identical
  support for every model.
- A threshold (or any decision-rule setting) only counts if the evaluation code
  applies it.
- Re-running and correcting old work is part of engineering, not an admission of
  failure. These fixes changed the per-class analysis but kept the methodology —
  which is the part worth keeping.
