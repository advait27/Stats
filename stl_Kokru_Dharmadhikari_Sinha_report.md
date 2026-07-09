<!--
  MIS41120 Statistical Learning — Practical Assignment
  Written report (Deliverable 1).
  Convert to .docx/.pdf with e.g.:  pandoc stl_Kokru_Dharmadhikari_Sinha_report.md -o report.docx
  Body word count target: <= 5000 words (teams of 3), EXCLUDING the title page,
  references, and figure captions. Replace every [PLACEHOLDER] before hand-up.
-->

# Predicting Credit-Card Default: A Comparison of Regularised Linear Models, Support Vector Machines, and Multilayer Perceptrons

**MIS41120 Statistical Learning — Practical Assignment**
**Hand-up date:** 10 July 2026

---

## Title Page

**Team members**

| Full name | Student number |
|---|---|
| Kokru, [Firstname] | [Student number] |
| Dharmadhikari, [Firstname] | [Student number] |
| Sinha, [Firstname] | [Student number] |

**Statement of own work.** We declare that this assignment is our own work, that
it has not been submitted for assessment in any other module or programme, and
that all sources used have been acknowledged. We understand the University's
policy on plagiarism and academic integrity and confirm that this submission
complies with it.

**Declaration on the use of AI tools.** In accordance with the module's policy on
generative AI, we declare the following. [EDIT THIS PARAGRAPH TO REFLECT YOUR
ACTUAL USE AND THE EXACT WORDING THE BRIEF REQUIRES.] Generative AI tools were
used as a coding and drafting aid: to help structure and comment the R scripts,
to debug package-specific issues, and to assist in drafting and proof-reading
this report. All modelling decisions, the interpretation of results, and the
final wording were reviewed, verified, and are owned by the authors. Every
numeric result reported here was produced by running our own R scripts on the
dataset; no results were generated or fabricated by an AI tool. Tool(s) used:
[NAME/VERSION]. Extent of use: [BRIEF DESCRIPTION].

**Per-member contribution**

- **Kokru** — [e.g. EDA and preprocessing pipeline (Stages 1–2), report
  sections 2–3]. Contribution: **[33]%**.
- **Dharmadhikari** — [e.g. logistic/regularised models and SVMs (Stages 3–5),
  report sections 4–5]. Contribution: **[33]%**.
- **Sinha** — [e.g. MLP, cross-method comparison and figures (Stages 6–7),
  report sections 6–7]. Contribution: **[34]%**.

*(Percentages sum to 100%. Adjust the split and task allocation to match your
team's actual division of work.)*

---

## 1. Introduction

The assignment has two parts. Part A returns to ISLR Chapter 3, Question 15(b)
on the **Boston** dataset (MASS library): fit a multiple regression of the
per-capita crime rate on all other variables, report for which predictors
H₀: βⱼ = 0 can be rejected, then repeat the problem with ridge, lasso and
elastic-net regularisation, and carry all of it over to SVM and MLP learners
(Section 2). Part B repeats the full programme on a publically available
dataset of our choice with at least 20 predictors — the UCI credit-card default
data (Sections 3–7). Because the credit study is the larger and richer of the
two, we introduce it first below and give the Boston benchmark in compact form
in Section 2.

Consumer credit risk is one of the oldest and most consequential applications of
statistical learning. A lender who can identify, in advance, which customers are
likely to miss their next payment can price risk correctly, set credit limits
prudently, and intervene early. The problem we study is a canonical instance of
this task: predicting whether a credit-card client will **default on their next
monthly payment**, using demographic attributes and six months of billing and
repayment history.

The application sits inside a **heavily regulated** domain. In retail banking,
credit-scoring models are subject to supervisory scrutiny and, increasingly, to
"right to explanation" expectations: an institution must be able to justify *why*
an applicant was declined. This makes **explainability a first-class modelling
objective**, not an afterthought. A model that predicts marginally better but
cannot be explained to a regulator or a customer may be unusable in practice.
Our comparison is therefore framed not only around predictive accuracy but
explicitly around the **trade-off between predictive performance and
interpretability**, alongside computational efficiency.

**Dataset and justification.** We use the UCI *Default of Credit Card Clients*
dataset (Yeh & Lien, 2009), comprising **30,000 clients** described by **23
predictors** plus an ID column and a binary target. The dataset satisfies the
assignment's structural requirements: it has **p = 23 ≥ 20** predictors and
**n = 30,000 ≥ 5p = 115** observations, giving a large sample relative to the
dimensionality — ample data to fit flexible models (SVM, MLP) without immediate
overfitting. The predictors span demographic variables (LIMIT_BAL, SEX,
EDUCATION, MARRIAGE, AGE), six months of ordinal repayment-status codes
(PAY_0, PAY_2–PAY_6), six monthly bill amounts (BILL_AMT1–6), and six monthly
payment amounts (PAY_AMT1–6). The target, *default payment next month*, is
binary. The dataset is publicly available, well-studied, and directly analogous
to the ISLR Chapter 3 setting we are asked to mirror.

Dataset URL: <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>

**Relation to ISLR Q15(b).** ISLR Chapter 3, Question 15(b) asks the analyst to
fit a multiple regression on *all* predictors and report for which predictors the
null hypothesis H₀: βⱼ = 0 can be rejected. The literal Boston answer is given in
Section 2; for the credit data we adapt the question to our binary target,
addressing the regression-versus-classification tension directly (Section 4), and
report the answer in Section 5.

---

## 2. Part A: the Boston benchmark — ISLR Q15(b) and its extensions

**Setup.** The Boston dataset (MASS library) records n = 506 census tracts and
14 variables. Following ISLR Question 15, the response is **crim**, the
per-capita crime rate, and the other 13 variables are the predictors. Q15(b) is
an *inference* question, so it is answered with a single OLS fit on the full,
untouched data, exactly as in the ISLR lab. Everything after it is a
*prediction* exercise and uses a 70/30 train/test split, predictors
standardised on training statistics only, and one shared 10-fold
cross-validation partition reused by every tuned model (script
`stl_Kokru_Dharmadhikari_Sinha_boston.R`).

**Q15(b): which predictors reject H₀: βⱼ = 0?** The full regression achieves
R² = 0.454 (F = 31.5, p < 2.2×10⁻¹⁶). We reject H₀ at α = 0.05 for

> zn, dis, rad, black, medv;

at α = 0.01 only dis, rad and medv survive, and at α = 0.001 only **dis and
rad**. Although many predictors correlate strongly with crim marginally, few
survive the joint fit: the predictors are highly collinear (rad/tax, nox/dis/
indus), so the shared signal is spread across near-redundant columns — the
same condition that motivates regularisation on the credit data.

**Ridge, lasso, elastic net.** All three cross-validate to nearly identical
error (CV MSE ≈ 36.2; the tuned elastic-net α = 0.3), and on the held-out test
set none beats unregularised OLS (RMSE 7.68 for OLS versus 7.72/7.72/7.76 for
lasso/elastic net/ridge). In this n ≫ p regime there is little overfitting for
a penalty to fix; the payoff is interpretability. At λ₁ₛₑ the lasso keeps
exactly **one** predictor — **rad**, accessibility to radial highways — which
is also the strongest Q15(b) signal (t = 6.7): the inferential test and the L1
selection agree on the dominant driver, foreshadowing the identical finding on
the credit data (Figures 16–17).

**SVM variants.** L1 and elastic-net SVMs (sparseSVM) exist only for binary
classification, so — exactly as the assignment's §2.1 anticipates — we convert
the problem: **high_crim = 1 if crim exceeds the training-set median
(T = 0.251)**, a balanced and practically meaningful "is this tract in the
high-crime half?" task. On it, the tuned **RBF (L2) SVM dominates: AUC 0.988,
accuracy 0.947**; the linear variants (near-hard-margin "unregularised",
linear L2, L1, elastic net) reach AUC 0.93–0.95 with accuracy 0.83–0.88, the
sparse ones keeping 9 of 13 features (Figure 19). A support-vector *regression*
(ε-SVR, RBF) was also fitted in the regression track — SVR supports only the
built-in L2 penalty, which is why it cannot cover the other variants — and its
test RMSE of 7.70 is indistinguishable from OLS.

**MLP variants.** The MLP needs no conversion: all four penalty variants
(unregularised, L2, L1, elastic net) were fitted as regression networks from
one identical h2o architecture with l1/l2 grids tuned on the shared folds.
Their test RMSEs span 7.75–7.75 — within noise of each other and no better
than OLS: with only 354 training tracts, the network has too little data for
its flexibility to pay (Figure 18).

| Regression track | Test RMSE | Test R² | Features |
|---|---|---|---|
| ols_unreg | **7.68** | 0.374 | 13 |
| svr_l2_rbf | 7.70 | 0.371 | 13 |
| lasso_l1 | 7.72 | 0.368 | 10 |
| enet | 7.72 | 0.367 | 10 |
| mlp_l1 / mlp_enet / mlp_unreg / mlp_l2 | 7.75–7.75 | 0.362–0.363 | 13 |
| ridge_l2 | 7.76 | 0.362 | 13 |

*Table B1. Boston regression track (predicting crim), held-out test set. The
SVM classification track (crim > median): RBF-L2 AUC 0.988 / accuracy 0.947;
linear "unregularised" 0.947 / 0.882; linear L2 0.948 / 0.868; L1 0.932 /
0.829; elastic net 0.941 / 0.882.*

**Which parts of Q15(b) do not carry over — and why ignoring them is valid.**
Q15(b)'s deliverable is a set of coefficient t-tests. Those exist for OLS
because its estimator has a known sampling distribution. An SVM decision
function is the optimum of a regularised hinge loss (often in an implicit
kernel space) and an MLP is a non-convex solution found by gradient descent:
neither has coefficients with a sampling distribution, so "for which βⱼ do we
reject H₀" is not a well-posed question for them. We therefore ignore the
hypothesis-testing component for SVM/MLP — the quantity being tested does not
exist — while keeping everything that does carry over: fit on all predictors,
regularise, evaluate out of sample. For the penalised linear models the
coefficients exist but classical p-values are invalid after data-driven
shrinkage (the post-selection-inference problem), so we report which
coefficients *survive* the penalty and cross-check that set against the
t-tests instead.

---

## 3. Part B: the credit-card study — data and preprocessing

**Class imbalance.** The target is imbalanced: **6,636 of 30,000 clients (22.1%)
defaulted**, against 77.9% who did not (Figure 1). This imbalance is central to
everything that follows. A trivial classifier that always predicts "no default"
already achieves **77.9% accuracy**, so raw accuracy is a misleading yardstick.
We therefore evaluate models primarily on **ROC-AUC** (threshold-independent
ranking quality) and on **recall/sensitivity** (the fraction of true defaulters
caught), reporting accuracy alongside for context.

**Repayment status (PAY_\*).** The six PAY_ variables record how many months
behind a client was in each of the last six months (−2/−1/0 denote paid duly or
no consumption; 1…8 denote months of delay). Exploratory analysis showed these
to be the **strongest signals** in the data: the empirical default rate rises
steeply and near-monotonically with the most recent repayment status (Figures 2
and 3). For the latest month (PAY_0), the default rate climbs from roughly 13%
for clients who paid duly to **69% at two months' delay and 76% at three**. This
pattern recurs across all six months (Figure 4).

**Bill-amount collinearity (BILL_AMT\*).** The six BILL_AMT columns are
consecutive monthly statement balances for the same clients. Because balances
change slowly, they are **highly correlated**: the mean off-diagonal correlation
across BILL_AMT1–6 is **0.886**, with pairwise values from 0.80 to 0.95
(Figure 5). Strong multicollinearity inflates the variance of unregularised
coefficients and makes them unstable and hard to interpret — the empirical
motivation for the ridge/lasso/elastic-net models introduced later.

**Recoding undocumented categories.** The data dictionary documents only a
subset of the categorical codes. EDUCATION contains undocumented codes 0, 5 and
6 (14, 280 and 51 clients respectively); MARRIAGE contains an undocumented code
0 (54 clients). Following common practice, we fold these rare, unlabelled codes
into the existing "other" category: **EDUCATION {0, 5, 6} → 4** and
**MARRIAGE {0} → 3**. This avoids sparse, noisy dummy levels while preserving the
documented category structure.

**Variable types.** We set types deliberately. SEX, EDUCATION and MARRIAGE are
treated as **unordered factors** (one-hot encoded for the matrix-based models).
The PAY_ variables are kept as a **single numeric column each**: they are
genuinely ordinal and roughly monotone in default risk, so a numeric encoding
preserves the ordering and the "more delay → more risk" signal while keeping the
design matrix compact. LIMIT_BAL, AGE, BILL_AMT\* and PAY_AMT\* are **continuous**.
The target is already binary, so — as noted in the assignment's Section 2.1 — no
regression-to-classification conversion is required; we simply type it as a
two-level factor with levels ("0", "1"), fixing "1" (default) as the positive
class throughout.

**Standardisation (leakage-free).** The continuous predictors span vastly
different scales (integer ages versus six-figure currency amounts). Because the
penalised models apply an equal penalty to every coefficient, and SVM/MLP are
distance/gradient based, all continuous predictors are **standardised to zero
mean and unit variance**. Crucially, the centre and scale are **learned from the
training set only** and then applied unchanged to the test set. Computing these
parameters over the full data would leak test-set information into preprocessing
and inflate the apparent test performance; fitting on train only keeps the
estimate honest. Factor dummies are left unscaled.

**Split and cross-validation.** We use a **stratified 70/30 train/test split**
(caret::createDataPartition), which preserves the 22.1% default rate in both
partitions — verified: full, train and test all show 22.1% default. This yields
**21,001 training** and **8,999 test** rows. For model tuning we construct a
**single shared set of stratified 10-fold cross-validation folds** on the
training target and reuse it across *every* model. Shared folds are essential for
a fair comparison: if each model used its own random folds, measured differences
could reflect the partition rather than the method.

---

## 4. Methods (credit-card study)

All models were fitted in R. We compare three model families, each in an
unregularised form and with L2, L1 and elastic-net (L1+L2) regularisation where
the family admits it.

**4.1 Linear models — OLS/logistic (+ ridge, lasso, elastic net).**
To answer ISLR Q15(b) literally we first fit a **linear probability model (LPM)**
— ordinary least squares on the 0/1 target — which provides the classical
t-tests the question asks for. We then fit **logistic regression**, the
statistically appropriate model for a binary response, as our unregularised
baseline. Regularisation is added via **penalised logistic regression** (glmnet,
family = "binomial"): **ridge** (α = 0, L2), **lasso** (α = 1, L1), and
**elastic net** (α tuned over 0.1–0.9). Every cross-validation call used the
shared folds and was tuned to maximise CV AUC.

**4.2 Support Vector Machines (+ variants).**
A support vector machine is **never truly "unregularised"**: the (½)‖w‖² term in
the soft-margin objective is an intrinsic L2 penalty, and the cost parameter C
controls the amount of regularisation (large C → narrow/"hard" margin → minimal
regularisation). We therefore map the four required variants onto SVMs as
follows, and state explicitly where a variant is an approximation:

- **"Unregularised" analogue** — a linear-kernel SVM with a large cost. We use
  **C = 10**. Empirically, the decision boundary (support-vector count) is
  already essentially constant for C ≥ 5, so C = 10 sits firmly in the
  weak-regularisation/near-hard-margin regime; larger values (e.g. 10³) merely
  increase training time and cause non-convergence without changing the model.
  This is the closest *valid* analogue to an (ill-defined) unregularised SVM.
- **L2 (standard SVM)** — the ordinary soft-margin SVM. We fit both an **RBF
  kernel** (nonlinear boundary; cost and γ tuned by cross-validation) and a
  **linear kernel** (as an apples-to-apples baseline against the sparse models).
  Class weights counter the imbalance.
- **L1 (lasso SVM)** — sparseSVM with α = 1: an L1 penalty on the weights that
  drives some to exactly zero (feature selection). The target is already binary,
  so no conversion is needed (assignment §2.1).
- **Elastic-net SVM** — sparseSVM with α tuned in {0.25, 0.5, 0.75}.

**4.3 Multilayer Perceptron (+ variants).**
The MLP was designed so that all four penalty variants come from **one engine**,
h2o.deeplearning, which exposes both `l1` and `l2` arguments on the same network
— giving unregularised (l1 = l2 = 0), L2, L1, and elastic-net (l1 > 0 and l2 > 0)
models that differ *only* in the penalty. Standard R MLP tooling (nnet,
neuralnet) centres on **L2 weight decay only** and has no L1 penalty, which is
precisely why a specialised engine is needed for the L1/elastic-net variants.
Inputs were already standardised (Section 2), so h2o's internal standardisation
was disabled to avoid double-scaling; class balancing was enabled to counter the
imbalance; the architecture was a fixed two-hidden-layer design.

**Reproducibility note on the MLP.** h2o runs on a Java Virtual Machine, so the
script keeps a **fallback path** using nnet (unregularised and L2 weight-decay
only — nnet has no L1 penalty) for machines without Java, and states clearly
which engine produced its output. The results reported below come from the h2o
path (OpenJDK 17): all four MLP variants — unregularised, L2, L1 and elastic
net — were fitted from one identical architecture, tuned on the shared folds
and selected by cross-validated AUC.

**Evaluation protocol.** Every model is evaluated on the **same held-out test
set** (8,999 rows) at a 0.5 probability threshold, and scored with an identical
metric set: accuracy, ROC-AUC, recall, specificity, precision, F1, balanced
accuracy, and the number of features used.

---

## 5. Results (credit-card study)

### 5.1 ISLR Q15(b), credit-data mirror: which predictors reject H₀: βⱼ = 0?

Fitting the multiple regression on all predictors (the linear probability model)
and testing each coefficient, we **reject H₀ at α = 0.05** for the following
predictors:

> LIMIT_BAL, SEX (level 2), EDUCATION (levels 2 and 4), MARRIAGE (level 2), AGE,
> PAY_0, PAY_2, PAY_3, PAY_5, BILL_AMT1, PAY_AMT1.

The logistic regression (using Wald z-tests, the appropriate test for a binary
response) yields an almost identical significant set at α = 0.05, adding PAY_AMT2
and dropping PAY_5. In **both** models the dominant predictor is **PAY_0**, the
most recent repayment status, whose logistic coefficient is positive with an
**odds ratio of 1.80** — each additional month of delay multiplies the odds of
default by roughly 1.8 — and a p-value of order **10⁻¹⁶⁹**. This directly
confirms the exploratory finding that repayment status is the primary driver of
default.

Two caveats justify not stopping at the LPM. First, the LPM produces fitted
"probabilities" outside [0, 1] for **1,569 clients (7.5%)** of the training set,
which are nonsensical. Second, its errors are inherently heteroskedastic, so its
t-tests are not strictly valid. Logistic regression resolves both issues, which
is why it — not the LPM — is our baseline going forward.

### 5.2 Multicollinearity confirmed

Variance Inflation Factors on the logistic model confirm the collinearity flagged
in Section 2: all six BILL_AMT variables have **VIF > 17** (BILL_AMT2 ≈ 43),
while every other predictor sits below 5. This instability of the unregularised
coefficients is exactly the condition the penalised models are designed to
address, and it is visible in the coefficients: under lasso, five of the six
BILL_AMT variables are driven to zero and only BILL_AMT1 is retained, whereas the
unregularised logistic model assigns them large, sign-flipping coefficients.

### 5.3 The leaderboard

Table 1 (Figure 13b) and Figure 13 present the full leaderboard, sorted by test
AUC. The headline numbers are:

| Method | Family | AUC | F1 | Recall | Precision | Accuracy | Features |
|---|---|---|---|---|---|---|---|
| mlp_l1 | MLP | **0.774** | 0.478 | 0.378 | 0.650 | 0.818 | 23 |
| mlp_l2 | MLP | 0.771 | 0.472 | 0.378 | 0.626 | 0.813 | 23 |
| mlp_unreg | MLP | 0.767 | 0.471 | 0.381 | 0.618 | 0.811 | 23 |
| mlp_enet | MLP | 0.767 | 0.473 | 0.383 | 0.616 | 0.811 | 23 |
| svm_l2_rbf | SVM | 0.746 | 0.462 | 0.392 | 0.564 | 0.799 | 27 |
| logistic_unreg | Logistic | 0.723 | 0.344 | 0.229 | 0.690 | 0.807 | 26 |
| enet | Penalised linear | 0.722 | 0.345 | 0.229 | 0.696 | 0.807 | 26 |
| lasso_l1 | Penalised linear | 0.722 | 0.344 | 0.228 | 0.695 | 0.807 | 25 |
| ridge_l2 | Penalised linear | 0.722 | 0.314 | 0.202 | 0.708 | 0.805 | 27 |
| svm_unreg_hardmargin | SVM | 0.720 | 0.344 | 0.229 | 0.693 | 0.807 | 27 |
| svm_l2_linear | SVM | 0.720 | 0.347 | 0.232 | 0.693 | 0.807 | 27 |
| svm_l1 | SVM | 0.718 | 0.271 | 0.169 | 0.690 | 0.799 | 25 |
| svm_enet | SVM | 0.718 | 0.269 | 0.167 | 0.692 | 0.799 | 26 |

*Table 1. Test-set performance of all models, sorted by AUC. Metrics computed on
the common 8,999-row held-out test set at a 0.5 threshold. "Features" = non-zero
coefficients (or p for kernel/MLP models). See Figure 13b.*

### 5.4 Accuracy versus AUC/recall under imbalance

The leaderboard makes the imbalance lesson concrete. Every model's accuracy
sits in a narrow **0.799–0.818 band** that barely exceeds the **trivial 0.779**
baseline, so accuracy compresses genuinely different models into near-ties. The
differences that matter appear in recall and AUC. The linear and penalised
models buy their ~0.807 accuracy by predicting "no default" almost always —
**recall ≈ 0.23**, missing over three-quarters of actual defaulters — whereas
the MLPs and the RBF SVM catch **38–39% of defaulters** at similar or better
accuracy and clearly higher AUC (0.746–0.774 versus ≈ 0.722). Ranking models by
accuracy alone would hide exactly this difference; judged on AUC and recall —
the metrics that matter under imbalance — the nonlinear learners come out ahead
(Figures 13–14).

Key figures to include: Figure 6 (baseline ROC), Figures 7–10 (glmnet CV curves
and lasso path), Figure 11 (SVM AUCs), Figure 12 (MLP AUCs), Figures 13–15
(leaderboard AUC, F1/recall, and interpretability-versus-AUC).

---

## 6. Discussion

**Which method won — and on which axis?** There is no single winner; the answer
depends on the objective, which is exactly the point of the comparison.

- **On predictive ranking (AUC) and recall**, the **MLP family wins** — best is
  the **L1-penalised MLP** (AUC 0.774), with all four MLP variants between
  0.767 and 0.774 and recall ≈ 0.38, and the **RBF SVM** next (AUC 0.746,
  recall 0.39). All clearly beat the linear/penalised family (AUC ≈ 0.722,
  recall ≈ 0.23).
- **On accuracy**, the L1 MLP nominally leads too (0.818), but the entire field
  sits within 0.799–0.818 against a trivial 0.779 baseline — as argued in
  Section 5.4, accuracy differences here mostly reflect the imbalance and the
  0.5 threshold, not model quality.
- **On interpretability**, the **logistic and lasso models win**. They deliver
  transparent, signed coefficients and odds ratios (lasso using only ~25
  features), at an AUC cost of about 0.05 relative to the MLP. In a regulated
  lending context this trade may well be worth making.
- **On efficiency**, the **linear/penalised models win** decisively. A single
  logistic fit took **0.18 s** and a cross-validated lasso **3.6 s**, versus
  **~5.5 s** for one RBF SVM fit on even an 8,000-row subsample and **~5.6 s**
  for one nnet MLP fit on this hardware — and the gap widens sharply once
  tuning is included (the h2o grid search behind the four reported MLP variants
  ran for tens of minutes) and, for the kernel SVM, the poor scaling in n
  (Section 7 note).

**Did the nonlinear learners beat the interpretable "banking" models?** Yes, but
modestly and with a cost. The best nonlinear model (L1 MLP, AUC 0.774) beats the
best interpretable linear model (logistic, AUC 0.723) by **+0.052 AUC (≈7%)**,
and lifts recall from 0.23 to 0.38 (0.39 for the RBF SVM). This indicates
**genuine, exploitable nonlinear structure** in default risk that a linear
decision boundary cannot capture. The
cost is explainability: the MLP and RBF SVM use all inputs through opaque
transformations and cannot be reduced to a coefficient table. Whether the gain
justifies the loss of transparency is a governance decision, not a purely
statistical one — and in a regulated setting the interpretable model with 93% of
the ranking performance is a defensible choice.

**Did lasso/elastic net agree with the classical significance tests?** Yes — and
this agreement is one of the most reassuring findings. The classical logistic
Wald tests flagged PAY_0 as overwhelmingly significant; lasso independently
**retained PAY_0 as its largest coefficient**, and elastic net agreed. More
broadly, the lasso-selected feature set overlaps strongly with the set of
Wald-significant predictors. When an **inferential** procedure (hypothesis
testing) and a **predictive** procedure (L1 selection) converge on the same
drivers, we can be confident the repayment-status signal is real and not an
artefact of either method.

**Did regularisation improve prediction?** For the linear models, essentially
**no**: ridge, lasso and elastic net all land within ±0.0004 AUC of the
unregularised logistic model. For the MLP the penalties helped modestly but
consistently — L1 gave the family's best AUC (0.774 versus 0.767 unregularised),
which is coherent: a neural network is the one model here flexible enough to
overfit, so it is the one that benefits from shrinkage. For the linear models
this is the expected result in a **low-dimensional regime** (n ≈ 21,000,
p ≈ 27): there is little overfitting for a penalty to correct. Regularisation's payoff here is therefore not accuracy but
**stability and interpretability** — lasso resolves the BILL_AMT collinearity by
selecting a single representative month, producing coefficients one can trust and
explain. This is itself an important, data-dependent conclusion: on a wider or
noisier problem the penalties would be expected to help predictive performance
too.

**Does the conclusion depend on the data?** Yes — and having run the identical
programme on two datasets (Section 2), we can answer this directly rather than
speculate. **Stable across both datasets:** regularisation did not improve
prediction (both are n ≫ p regimes) but bought stability and sparsity; L1
selection agreed with classical inference on the dominant signal (PAY_0 on the
credit data, rad on Boston); and nonlinear learners extracted signal a linear
boundary missed. **Dataset-dependent:** the *size* and *location* of the
nonlinear advantage. On the credit data it was modest (+0.05 AUC) and belonged
to the MLP; on Boston's converted classification task it was dramatic (RBF SVM
AUC 0.988 versus ≈ 0.93–0.95 for the linear SVMs), while the MLP added nothing
over OLS on the small Boston regression — with n = 354 training rows a network
has too little data to exploit its flexibility. Which learner wins therefore
depends on the sample size and structure of the data; on a genuinely
high-dimensional problem the regularised models would likely also separate
from the unregularised baseline.

---

## 7. Conclusion

We completed ISLR Q15(b) on the Boston data and then compared linear models,
support vector machines, and multilayer perceptrons — each in unregularised,
ridge (L2), lasso (L1) and elastic-net forms where the family admits them — on
both Boston and the UCI credit-card default data. On the credit data the
**L1-regularised MLP achieved the best predictive ranking** (AUC 0.774, with
all four MLP variants at 0.767–0.774), the RBF SVM second (0.746); the
nonlinear models caught 38–39% of true defaulters against ≈ 23% for the entire
linear/penalised family, evidencing real nonlinear structure. The
**interpretable models** (logistic, lasso) retained ~93% of that ranking
performance while remaining transparent and by far the cheapest to fit, and
their L1-selected features **agreed with classical significance tests** on the
key drivers (PAY_0 on the credit data; rad on Boston). Under the class
imbalance, **accuracy was shown to be a misleading metric**: every model sat
within two points of the 0.779 trivial baseline while differing hugely in how
many defaulters it caught. For a regulated lending application, we would
recommend the **lasso logistic model** as the primary scorecard — near-best
ranking, transparent, stable under collinearity, and instant to fit — while
noting that an MLP could serve as a challenger model where its higher recall
justifies the loss of explainability.

**Limitations.** Two constraints affected the results. First, e1071's SVM solver
scales poorly in n, so on the credit data the kernel/linear SVMs were trained on
an 8,000-row stratified subsample of the training data (all models were
evaluated on the full test set, keeping the comparison honest). Second, the four
MLP variants require h2o on a Java runtime (OpenJDK 17 here); on a machine
without Java the scripts fall back to nnet and fit only the unregularised and
L2 variants, saying so explicitly rather than fabricating the missing rows.

---

## References

- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to
  Statistical Learning with Applications in R* (2nd ed.). Springer. [ISLR;
  Chapter 3, Question 15.]
- Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for
  the predictive accuracy of probability of default of credit card clients.
  *Expert Systems with Applications, 36*(2), 2473–2480.
- UCI Machine Learning Repository. *Default of Credit Card Clients Data Set.*
  <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>

<!--
  FIGURE LIST (insert images from figures/ with these captions):
  Fig 1  01_class_balance.png            — Target class balance (22.1% default).
  Fig 2  02_pay_status_distributions.png — Distribution of PAY_0..PAY_6 codes.
  Fig 3  03_default_rate_by_pay0.png     — Default rate rises with PAY_0.
  Fig 4  04_default_rate_by_pay_all.png  — Default rate by status, all 6 months.
  Fig 5  05_billamt_corrplot.png         — BILL_AMT1..6 correlation (mean 0.886).
  Fig 6  06_baseline_roc.png             — Unregularised logistic test ROC.
  Fig 7  07_ridge_cv.png                 — Ridge CV AUC vs log(lambda).
  Fig 8  08_lasso_cv.png                 — Lasso CV AUC vs log(lambda).
  Fig 9  09_enet_cv.png                  — Elastic-net CV AUC (winning alpha).
  Fig 10 10_lasso_path.png               — Lasso coefficient paths.
  Fig 11 11_svm_auc.png                  — SVM variants, test AUC.
  Fig 12 12_mlp_auc.png                  — MLP variants, test AUC.
  Fig 13 13_leaderboard_auc.png          — All methods, test AUC.
  Fig 13b 13b_leaderboard_table.png      — Leaderboard table (= Table 1).
  Fig 14 14_leaderboard_f1_recall.png    — F1 and recall by method.
  Fig 15 15_interpretability_vs_auc.png  — Features used vs AUC (trade-off).
  Fig 16 16_boston_lasso_cv.png          — Boston: lasso CV MSE vs log(lambda).
  Fig 17 17_boston_lasso_path.png        — Boston: lasso coefficient paths.
  Fig 18 18_boston_regression_rmse.png   — Boston: regression track, test RMSE.
  Fig 19 19_boston_svm_auc.png           — Boston: SVM variants, test AUC.
-->
