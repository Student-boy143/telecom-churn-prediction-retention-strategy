# Methodology

This document explains the analytical decisions behind the project in more depth than the README. All numbers come from `results/metrics.json`, written by `notebooks/telecom_churn_analysis.ipynb`.

## 1. Prediction setup and leakage control
- **Prediction date (T):** the data snapshot. **Features:** everything known at T (monthly averages, recent-change features, lifetime totals up to T, account and handset data). **Label:** churn **31–60 days after T**, giving roughly one month to act.
- **Leakage checks:** (a) column-by-column definition review; (b) consistency test — the handset-age/tenure ratio has the same ceiling for churners and stayers (99th percentile 1.043 vs 1.049), so tenure was not measured at the churn date; (c) "too good to be true" test — the best single feature reaches only ROC-AUC 0.574.
- **Preprocessing leakage:** clipping limits, medians, scaling and category encodings are learned **inside each training fold** (explicit `Pipeline` per fold). No feature selection is performed.

## 2. Data preparation decisions
| Issue | Decision |
|---|---|
| `Customer_ID` | Dropped (identifier) |
| `ethnic` | Dropped (protected attribute — fairness and legal risk in targeting) |
| 6 household columns > 30% missing | Dropped |
| `eqpdays < 0` (133 rows) | Set to missing (impossible) |
| 357 customers missing a whole usage block | Kept with a `usage_missing` flag. They show normal recent revenue and longer tenure, so this looks like a **data-extraction gap, not inactivity** |
| 1,615 customers with zero minutes | `zero_usage` flag (~80% churn — may already have effectively left) |
| Engineered features | `in_risk_window` (months 10–12), `handset_vs_tenure`, `drop_rate`, `rev_per_min`, `is_refurb` |

## 3. Hypothesis validation (statistics)
- Churn by tenure month with **95% Wilson confidence intervals**: month 9 28.8% (27.4%–30.2%), month 11 65.1% (64.1%–66.2%).
- Logistic regression with progressively more controls: months 10–12 odds ratio 1.87 → 1.89 → 1.92; handset age odds ratio 1.34 per year.
- **Interpretation:** these are associations. The data has no contract field, so months 10–12 are treated as an observed *risk window*, not proof of contract expiry.

## 4. Modelling
- 80/20 stratified split (seed 42); model selection by **5-fold stratified cross-validation on the training set only**; the 20,000-customer test set was used once.
- Logistic Regression (L2, clipped + log-transformed + scaled numerics, one-hot categoricals) vs LightGBM (native categorical handling, missing values kept).
- A conservative LightGBM (15 leaves, ≥300 customers per leaf, L2 = 5, 800 trees) matched the default's validation AUC with roughly half the train–validation gap (0.067 vs 0.112) and was selected.

## 5. Interpretation
SHAP (TreeExplainer) on the test set. The model independently learned a risk jump at months 11–12 and a step once handsets pass about one year — the same patterns found statistically. SHAP explains *what the model uses*, not *what would change behaviour*.

**Segments** (highest-risk 20% of the balanced test sample; sample churn rates):

| Segment | Customers | Sample churn | Mean monthly revenue |
|---|---|---|---|
| Data-quality review | 40 | 82.5% | $65 |
| Likely inactive | 281 | 83.6% | $24 |
| Renewal window | 1,352 | 75.9% | $60 |
| Ageing handset | 1,906 | 71.1% | $53 |
| Service quality | 133 | 78.2% | $74 |
| Other high risk | 288 | 69.4% | $84 |

## 6. Probability calibration (prior-shift adjustment)
The model learned on a 49.6%-churn sample, so its raw probabilities are far too high for a real customer base. Under the assumption that **only the class proportion differs** between sample and reality (class-conditional feature distributions are stable):

`p_real = p·k / (p·k + 1 − p)`, with `k = [c/(1−c)] / [π/(1−π)]`, π = sample churn rate, c = assumed real churn rate.

The adjustment preserves risk ordering (Spearman = 1.0) but **changes risk × revenue ordering**. At 2% churn, about a third of the value-ranked top-2% list changes (weighted overlap 66%), and the corrected list holds more real churners (3.4% → 5.4% precision). This is an internal-consistency correction, **not** verified real-world calibration: that needs a recent, unbalanced customer sample with observed outcomes.

## 7. Financial model
Per monthly campaign, base of 100,000 customers; the test set is re-weighted so churners make up the scenario churn rate `c`.

| Quantity | Formula |
|---|---|
| List size `L`, would-be churners `C` | weighted counts on the contact list |
| Value of a saved customer `V` | avg revenue of would-be churners × margin × Σ(1−c)^t over `H` months |
| Benefit | `C × u × V` (u = uplift = share of would-be churners saved; margin, not revenue) |
| Cost | `L × contact cost + L × acceptance × offer cost` (offer cost paid per actual acceptor) |
| Break-even uplift | `Cost / (C × V)` |
| Break-even offer cost | `(C·u·V − L·contact cost) / (L·acceptance)` |

**Acceptance vs uplift.** Overall acceptance drives cost; uplift drives benefit. Overall acceptance is **not** an upper bound on uplift: under the exclusion restriction (an offer can only save a customer who accepts it), uplift ≤ acceptance *among would-be churners*, which is unknown. The model enforces only `u ≤ min(1, a / list precision)`.

**Audit.** Design A was recomputed by hand to the cent; break-even values plugged back give net = 0; pilot sample sizes agree with `statsmodels` power analysis within 0.3%. Stress test: if saved customers keep churning at their elevated rate, Design B's break-even uplift rises from 18.8% to 22.5%.

## 8. Randomised pilot
- Randomly split a pilot list into treatment and control; measure churn for **both groups over the same 31–60-day window**.
- **Uplift (intention-to-treat)** = (control churn − treatment churn) ÷ control churn.
- **Complier-effect estimate** = absolute churn reduction ÷ acceptance — valid only if assignment is random, control customers cannot receive the offer, the offer works only through acceptance, and customers do not affect each other's outcomes.
- Sample size (two-sided, 5% significance, 80% power, 2% churn, 20% uplift): 5,952 per group on a top-10% risk list (≈1.2 months) vs 6,218 per group on the 2% value list (≈6 months). The broader list measures uplift; the value-ranked list is the proposed scale-up strategy.
