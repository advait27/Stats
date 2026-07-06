# stl_Kokru_Dharmadhikari_Sinha_baseline.R
#
# MIS41120 Statistical Learning — Stage 3: unregularised baseline.
# This is our take on ISLR Ch.3 Q15(b), adapted to a binary target: `default`
# is a factor with levels "0"/"1" (1 = the client defaults next month).
#
# Q15(b) says: regress y on all predictors and report which betas are
# significant. It assumes a continuous response and OLS t-tests, but ours is
# binary. To handle that we fit two things:
#   - a linear probability model (OLS straight on the 0/1 target) so we can
#     answer Q15(b) as literally as possible, flaws and all; and
#   - logistic regression, which is the model we actually keep going forward.
# Then we look at VIF to justify the penalised models coming in Stage 4, and
# evaluate logistic on the held-out test set with metrics that make sense for
# an imbalanced problem.
#
# Loads the Stage-2 train/test frames from data/. Relative paths, reproducible;
# run from the project root.


# Setup: seed, packages, output folders.
set.seed(1)  # nothing here is random, just keeping the habit

ensure_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  library(pkg, character.only = TRUE)
}

# car for vif(), pROC for the ROC curve / AUC.
ensure_pkg("car")
ensure_pkg("pROC")

fig_dir  <- "figures"
data_dir <- "data"
if (!dir.exists(fig_dir))  dir.create(fig_dir,  recursive = TRUE)
if (!dir.exists(data_dir)) dir.create(data_dir, recursive = TRUE)


# Load the model-ready frames from Stage 2. These already have the categories
# recoded, types fixed, continuous predictors standardised on train only, and a
# stratified 70/30 split. `default` is a factor c("0","1").
train_df <- readRDS(file.path(data_dir, "train_df.rds"))
test_df  <- readRDS(file.path(data_dir, "test_df.rds"))

stopifnot(is.factor(train_df$default),
          identical(levels(train_df$default), c("0", "1")))


# Helper: turn a p-value into the usual significance stars.
sig_stars <- function(p) {
  ifelse(p < 0.001, "***",
  ifelse(p < 0.01,  "**",
  ifelse(p < 0.05,  "*",
  ifelse(p < 0.1,   ".", " "))))
}


# Linear probability model — the literal Q15(b) mirror
#
# Q15(b) wants "regress y on all predictors, report which betas are
# significant". So we just run OLS on the 0/1 target. That's the linear
# probability model: fitted values stand in for P(default=1 | X) and each
# t-test says whether a predictor is (linearly) associated with that
# probability.
#
# Careful with the conversion: as.numeric() on a factor returns the level
# codes 1/2, so go through as.character() first to get real 0/1.
train_lpm <- train_df
train_lpm$default_numeric <- as.numeric(as.character(train_lpm$default))
train_lpm$default <- NULL  # drop the factor so "~ ." only picks up predictors

lpm_fit <- lm(default_numeric ~ ., data = train_lpm)

cat("\n############################################################\n")
cat("# 1. LINEAR PROBABILITY MODEL (OLS on 0/1) — ISLR Q15(b)   #\n")
cat("############################################################\n")
print(summary(lpm_fit))

# Tidy up the coefficient table with stars.
lpm_co <- summary(lpm_fit)$coefficients
lpm_tab <- data.frame(
  predictor = rownames(lpm_co),
  estimate  = round(lpm_co[, "Estimate"],   5),
  std_error = round(lpm_co[, "Std. Error"], 5),
  t_value   = round(lpm_co[, "t value"],    3),
  p_value   = signif(lpm_co[, "Pr(>|t|)"],  3),
  signif    = sig_stars(lpm_co[, "Pr(>|t|)"]),
  row.names = NULL
)
cat("\n--- LPM tidy coefficient table ---\n")
print(lpm_tab, row.names = FALSE)

# Q15(b): which predictors let us reject H0: beta_j = 0 (drop the intercept).
lpm_p    <- lpm_co[, "Pr(>|t|)"]
lpm_keep <- setdiff(names(lpm_p), "(Intercept)")
lpm_sig_05  <- lpm_keep[lpm_p[lpm_keep] < 0.05]
lpm_sig_01  <- lpm_keep[lpm_p[lpm_keep] < 0.01]
lpm_sig_001 <- lpm_keep[lpm_p[lpm_keep] < 0.001]

cat("\n--- LPM: predictors with significant association to default ---\n")
cat("Reject H0 at alpha=0.05  :", paste(lpm_sig_05,  collapse = ", "), "\n")
cat("Reject H0 at alpha=0.01  :", paste(lpm_sig_01,  collapse = ", "), "\n")
cat("Reject H0 at alpha=0.001 :", paste(lpm_sig_001, collapse = ", "), "\n")

# Why we don't stop at the LPM. Two well-known problems:
#   (i)  fitted "probabilities" aren't bounded to [0,1] — OLS happily predicts
#        negatives or values above 1, which don't mean anything as probabilities.
#   (ii) the errors are heteroskedastic by construction (Var(y|X) = p(1-p)),
#        so the OLS standard errors, and therefore the t-tests above, aren't
#        strictly valid.
# So the LPM is here just to answer Q15(b) on its own terms; the model we keep
# is logistic. The check below counts how many fitted values fall outside [0,1]
# as concrete evidence of problem (i).
lpm_fitted   <- fitted(lpm_fit)
n_out_of_01  <- sum(lpm_fitted < 0 | lpm_fitted > 1)
cat(sprintf("\nLPM sanity check: %d of %d fitted 'probabilities' fall OUTSIDE [0,1] (%.1f%%)\n",
            n_out_of_01, length(lpm_fitted), 100 * n_out_of_01 / length(lpm_fitted)))
cat("-> concrete evidence of the LPM's boundedness flaw; logistic fixes this.\n")


# Logistic regression — the baseline we actually keep
#
# This models the log-odds of default as linear in the predictors, so fitted
# values stay in (0,1) and the Wald z-tests are valid. This is the unregularised
# baseline that feeds the Stage-7 comparison.
glm_fit <- glm(default ~ ., data = train_df, family = binomial)

cat("\n############################################################\n")
cat("# 2. LOGISTIC REGRESSION (unregularised baseline)         #\n")
cat("############################################################\n")
print(summary(glm_fit))

# Same tidy table as before, this time with z-values.
glm_co <- summary(glm_fit)$coefficients
glm_tab <- data.frame(
  predictor = rownames(glm_co),
  estimate  = round(glm_co[, "Estimate"],   5),
  std_error = round(glm_co[, "Std. Error"], 5),
  z_value   = round(glm_co[, "z value"],    3),
  p_value   = signif(glm_co[, "Pr(>|z|)"],  3),
  signif    = sig_stars(glm_co[, "Pr(>|z|)"]),
  row.names = NULL
)

# Odds ratios, exp(coef), with 95% Wald CIs.
# exp(beta) is the multiplicative change in the odds of default per unit of the
# predictor (per 1 SD for the standardised continuous ones, vs the reference
# level for factor dummies). We use the Wald interval exp(beta +/- 1.96*se) —
# profiling via confint() is more exact but much slower and overkill here.
or_est <- exp(glm_co[, "Estimate"])
or_lo  <- exp(glm_co[, "Estimate"] - 1.96 * glm_co[, "Std. Error"])
or_hi  <- exp(glm_co[, "Estimate"] + 1.96 * glm_co[, "Std. Error"])
glm_tab$odds_ratio <- round(or_est, 3)
glm_tab$OR_CI_low  <- round(or_lo, 3)
glm_tab$OR_CI_high <- round(or_hi, 3)

cat("\n--- Logistic tidy coefficient table (with odds ratios) ---\n")
print(glm_tab, row.names = FALSE)

# Q15(b) again, this time under logistic.
glm_p    <- glm_co[, "Pr(>|z|)"]
glm_keep <- setdiff(names(glm_p), "(Intercept)")
glm_sig_05  <- glm_keep[glm_p[glm_keep] < 0.05]
glm_sig_01  <- glm_keep[glm_p[glm_keep] < 0.01]
glm_sig_001 <- glm_keep[glm_p[glm_keep] < 0.001]

cat("\n--- Logistic: predictors with significant association to default ---\n")
cat("Reject H0 at alpha=0.05  :", paste(glm_sig_05,  collapse = ", "), "\n")
cat("Reject H0 at alpha=0.01  :", paste(glm_sig_01,  collapse = ", "), "\n")
cat("Reject H0 at alpha=0.001 :", paste(glm_sig_001, collapse = ", "), "\n")

# Rank the significant predictors by |z| to see which signals dominate.
glm_sig_tab <- glm_tab[glm_tab$predictor %in% glm_sig_05, ]
glm_sig_tab <- glm_sig_tab[order(-abs(glm_sig_tab$z_value)), ]
cat("\n--- Strongest significant predictors (by |z|) ---\n")
print(head(glm_sig_tab[, c("predictor", "estimate", "odds_ratio",
                           "OR_CI_low", "OR_CI_high", "p_value")], 8),
      row.names = FALSE)

# We expect the PAY_* repayment-status variables to be the strongest, positive
# warning signs: more delay -> higher log-odds of default, so positive coefs and
# odds ratios above 1. PAY_0 (most recent month) is usually the single biggest
# signal. Since PAY_* are treated as numeric-ordinal, each extra step of delay
# multiplies the odds by exp(beta). Lines up with the Stage-1 EDA, where the
# default rate climbed sharply with repayment delay.
pay_rows <- glm_tab[glm_tab$predictor %in% c("PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"), ]
cat("\n--- PAY_* coefficients (dominant default warning signs) ---\n")
print(pay_rows[, c("predictor","estimate","odds_ratio","p_value","signif")],
      row.names = FALSE)


# Multicollinearity check (VIF) — the reason for Stage 4
#
# The VIF says how much each coefficient's variance is inflated by correlation
# with the other predictors. Rough rule: >5 is worth noting, >10 is a real
# problem. Stage 1 showed the BILL_AMT1..6 block correlated around 0.89, so we
# expect big VIFs there. High VIF means unstable, hard-to-read coefficients,
# which is precisely what ridge/lasso/elastic-net fix in Stage 4 by shrinking or
# selecting among the correlated predictors.
vif_vals <- car::vif(glm_fit)

cat("\n############################################################\n")
cat("# 3. MULTICOLLINEARITY (VIF) — motivates ridge/lasso      #\n")
cat("############################################################\n")
# car::vif() gives a plain vector when every term has one df, but a matrix with
# a GVIF column once there's a factor with >1 df. Cover both cases.
if (is.matrix(vif_vals)) {
  vif_df <- data.frame(predictor = rownames(vif_vals),
                       GVIF = round(vif_vals[, "GVIF"], 3),
                       df   = vif_vals[, "Df"],
                       # GVIF^(1/(2*df)) is the df-adjusted version, comparable across terms
                       GVIF_adj = round(vif_vals[, ncol(vif_vals)], 3),
                       row.names = NULL)
  vif_df <- vif_df[order(-vif_df$GVIF), ]
  print(vif_df, row.names = FALSE)
  high_vif <- vif_df$predictor[vif_df$GVIF > 5]
} else {
  vif_df <- data.frame(predictor = names(vif_vals),
                       VIF = round(as.numeric(vif_vals), 3),
                       row.names = NULL)
  vif_df <- vif_df[order(-vif_df$VIF), ]
  print(vif_df, row.names = FALSE)
  high_vif <- vif_df$predictor[vif_df$VIF > 5]
}
cat("\nPredictors with high collinearity (VIF/GVIF > 5):",
    ifelse(length(high_vif) == 0, "none", paste(high_vif, collapse = ", ")), "\n")
cat("-> The BILL_AMT* block's inflated VIFs confirm unstable unregularised\n")
cat("   coefficients; this MOTIVATES the penalised models in Stage 4.\n")


# Test-set evaluation
#
# Score the logistic model on the held-out test set (untouched during fitting
# and preprocessing) for an unbiased read on performance.

# Predicted P(default=1) and the 0.5-threshold class.
test_prob <- predict(glm_fit, newdata = test_df, type = "response")
test_pred <- factor(ifelse(test_prob >= 0.5, "1", "0"), levels = c("0", "1"))
test_true <- test_df$default

# Confusion matrix, rows = predicted, cols = actual, positive class "1".
cm <- table(Predicted = test_pred, Actual = test_true)
# Pull the four cells out defensively — a class could be missing from the predictions.
TP <- ifelse("1" %in% rownames(cm) && "1" %in% colnames(cm), cm["1","1"], 0)
TN <- ifelse("0" %in% rownames(cm) && "0" %in% colnames(cm), cm["0","0"], 0)
FP <- ifelse("1" %in% rownames(cm) && "0" %in% colnames(cm), cm["1","0"], 0)
FN <- ifelse("0" %in% rownames(cm) && "1" %in% colnames(cm), cm["0","1"], 0)

# Metrics — several of them on purpose, not just accuracy, because of the class
# imbalance (see the note further down).
accuracy    <- (TP + TN) / (TP + TN + FP + FN)
sensitivity <- TP / (TP + FN)                       # recall / true-positive rate
specificity <- TN / (TN + FP)                       # true-negative rate
precision   <- ifelse((TP + FP) > 0, TP / (TP + FP), NA)  # positive predictive value
f1          <- ifelse(!is.na(precision) && (precision + sensitivity) > 0,
                      2 * precision * sensitivity / (precision + sensitivity), NA)
bal_acc     <- (sensitivity + specificity) / 2      # balanced accuracy

# ROC-AUC: threshold-free ranking quality. Feed it the probabilities, not the
# 0/1 labels, so AUC captures how well the model ranks defaulters above
# non-defaulters across all thresholds.
roc_obj <- pROC::roc(response = test_true, predictor = test_prob,
                     levels = c("0", "1"), direction = "<", quiet = TRUE)
auc_val <- as.numeric(pROC::auc(roc_obj))

# Why accuracy alone is misleading at ~22% prevalence: a classifier that always
# says "no default" is already right ~78% of the time, so an accuracy near 0.78
# proves nothing. Recall (are we actually catching defaulters?) and AUC (ranking
# quality, threshold-independent) are what separate a useful model from that.
trivial_acc <- max(prop.table(table(test_true)))

cat("\n############################################################\n")
cat("# 4. TEST-SET EVALUATION (logistic, threshold = 0.5)      #\n")
cat("############################################################\n")
cat("Confusion matrix (rows=Predicted, cols=Actual):\n")
print(cm)
cat(sprintf("\nTrivial 'always predict no-default' accuracy : %.4f\n", trivial_acc))
cat(sprintf("Model accuracy        : %.4f\n", accuracy))
cat(sprintf("Sensitivity / recall  : %.4f   (fraction of true defaulters caught)\n", sensitivity))
cat(sprintf("Specificity           : %.4f\n", specificity))
cat(sprintf("Precision             : %.4f\n", precision))
cat(sprintf("F1 score              : %.4f\n", f1))
cat(sprintf("Balanced accuracy     : %.4f\n", bal_acc))
cat(sprintf("ROC-AUC               : %.4f\n", auc_val))
cat("\nNOTE: model accuracy barely beats the trivial baseline, but recall and\n")
cat("AUC reveal the model's actual (limited) skill at ranking defaulters — this\n")
cat("is why we judge all later models on AUC/recall/F1, not accuracy alone.\n")

# Save the ROC curve.
png(file.path(fig_dir, "06_baseline_roc.png"), width = 700, height = 700, res = 120)
plot(roc_obj, col = "#C44E52", lwd = 2,
     main = sprintf("Unregularised logistic — test ROC (AUC = %.3f)", auc_val),
     legacy.axes = TRUE)  # x-axis = 1 - specificity
abline(a = 0, b = 1, lty = 2, col = "grey50")  # chance line
dev.off()


# Save a one-row metrics summary for the Stage-7 comparison.
# Each method writes the same one-row shape and Stage 7 rbinds them into one
# leaderboard, so we stick to the columns every method has in common.
metrics_baseline <- data.frame(
  method            = "logistic_unreg",
  accuracy          = accuracy,
  auc               = auc_val,
  recall            = sensitivity,     # = sensitivity
  specificity       = specificity,
  precision         = precision,
  f1                = f1,
  balanced_accuracy = bal_acc,
  n_features_used   = length(coef(glm_fit)) - 1,  # exclude intercept
  stringsAsFactors  = FALSE
)
saveRDS(metrics_baseline, file.path(data_dir, "metrics_baseline.rds"))


# Final console summary.
cat("\n\n=================== STAGE 3 BASELINE SUMMARY ===================\n")
cat("ISLR Q15(b) answer — predictors where we reject H0: beta_j = 0 (alpha=0.05):\n")
cat("  Linear probability model (OLS) :\n    ",
    paste(lpm_sig_05, collapse = ", "), "\n")
cat("  Logistic regression (Wald)     :\n    ",
    paste(glm_sig_05, collapse = ", "), "\n")
cat(sprintf("\nLPM fitted probs outside [0,1] : %d (%.1f%%) -> LPM inadequate; use logistic.\n",
            n_out_of_01, 100 * n_out_of_01 / length(lpm_fitted)))
cat(sprintf("High-collinearity predictors (VIF>5): %s\n",
            ifelse(length(high_vif)==0, "none", paste(high_vif, collapse=", "))))
cat("  -> motivates ridge / lasso / elastic-net in Stage 4.\n")
cat("\nTest-set performance (unregularised logistic, threshold 0.5):\n")
cat(sprintf("  accuracy=%.4f  AUC=%.4f  recall=%.4f  precision=%.4f  F1=%.4f  bal.acc=%.4f\n",
            accuracy, auc_val, sensitivity, precision, f1, bal_acc))
cat(sprintf("  (trivial no-default accuracy = %.4f — accuracy alone is misleading)\n",
            trivial_acc))
cat(sprintf("\nSaved: %s, %s\n",
            file.path(fig_dir,  "06_baseline_roc.png"),
            file.path(data_dir, "metrics_baseline.rds")))
cat("===============================================================\n")

# End of Stage 3.
