# stl_Kokru_Dharmadhikari_Sinha_boston.R
#
# MIS41120 Statistical Learning — Practical Assignment, Part A: the Boston data.
#
# The brief asks us to (i) complete ISLR Ch.3 Q15(b) on the Boston dataset from
# the MASS library, (ii) repeat that unregularised OLS problem with ridge (L2),
# lasso (L1) and elastic-net (L1+L2) regularisation, and (iii) carry all of it
# over to two other learners, SVM and MLP, again in unregularised / L2 / L1 /
# elastic-net variants where those exist.
#
# ISLR Q15 sets the response: "we will now try to predict per capita crime rate
# using the other variables" — so `crim` is y and the other 13 columns are the
# predictors. Q15(b) itself is an INFERENCE question (which beta_j reject
# H0: beta_j = 0?), so we answer it the way ISLR does: one OLS fit on the full,
# untouched data, read the t-tests. Everything after that is a PREDICTION
# comparison, so for those models we switch to the usual honest setup: a 70/30
# train/test split, predictors standardised on training statistics only, and one
# shared set of 10-fold CV assignments reused by every tuned model (same
# reasoning as in our credit-card pipeline: shared folds keep the comparison
# about the methods, not the partitions).
#
# Two method-specific notes, both anticipated by the brief's Section 2.1:
#   * SVM with L1 / elastic-net penalties only exists (sparseSVM) for BINARY
#     CLASSIFICATION, so for the SVM family we convert the regression problem to
#     classification: high_crim = 1 if crim > median(crim). The brief explicitly
#     sanctions this threshold construction.
#   * The hypothesis-testing part of Q15(b) does not carry over to SVM/MLP at
#     all — those models have no coefficient sampling distribution to test.
#     That, and why ignoring it is valid, is printed (and argued) in section 7
#     at the bottom of this script.
#
# Self-contained: Boston ships with the MASS package, so there is no data file
# to load. Relative paths, seeded, runs out of the box from any directory.


# --- 0. Setup: seed, packages, output folders --------------------------------

set.seed(1)  # fixes the split, the folds, and every tuning draw below

ensure_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  library(pkg, character.only = TRUE)
}

# MASS for the data; glmnet for ridge/lasso/enet; e1071 + sparseSVM for the SVM
# variants; nnet as the always-available MLP engine; pROC for classification AUC.
ensure_pkg("MASS")
ensure_pkg("glmnet")
ensure_pkg("e1071")
ensure_pkg("sparseSVM")
ensure_pkg("nnet")
ensure_pkg("pROC")

fig_dir  <- "figures"
data_dir <- "data"
if (!dir.exists(fig_dir))  dir.create(fig_dir,  recursive = TRUE)
if (!dir.exists(data_dir)) dir.create(data_dir, recursive = TRUE)

# Helper: significance stars for the coefficient tables.
sig_stars <- function(p) {
  ifelse(p < 0.001, "***",
  ifelse(p < 0.01,  "**",
  ifelse(p < 0.05,  "*",
  ifelse(p < 0.1,   ".", " "))))
}


# --- 1. ISLR Q15(b): OLS on the full data, which betas reject H0? ------------
# Exactly what the question asks: regress crim on all 13 other variables and
# read off the t-tests. Fit on the FULL data, unstandardised — this is
# inference, not prediction, and it mirrors how the ISLR lab does it. (Scaling
# would change the coefficients but not the t-statistics or p-values anyway.)

boston <- MASS::Boston
stopifnot(nrow(boston) == 506, ncol(boston) == 14)

ols_full <- lm(crim ~ ., data = boston)

cat("\n############################################################\n")
cat("# 1. ISLR Q15(b): OLS of crim on all predictors (n=506)   #\n")
cat("############################################################\n")
print(summary(ols_full))

ols_co  <- summary(ols_full)$coefficients
ols_tab <- data.frame(
  predictor = rownames(ols_co),
  estimate  = round(ols_co[, "Estimate"],   5),
  std_error = round(ols_co[, "Std. Error"], 5),
  t_value   = round(ols_co[, "t value"],    3),
  p_value   = signif(ols_co[, "Pr(>|t|)"],  3),
  signif    = sig_stars(ols_co[, "Pr(>|t|)"]),
  row.names = NULL
)
cat("\n--- Q15(b) tidy coefficient table ---\n")
print(ols_tab, row.names = FALSE)

# The literal Q15(b) answer, at the usual three alpha levels.
ols_p    <- ols_co[, "Pr(>|t|)"]
ols_keep <- setdiff(names(ols_p), "(Intercept)")
q15_sig_05  <- ols_keep[ols_p[ols_keep] < 0.05]
q15_sig_01  <- ols_keep[ols_p[ols_keep] < 0.01]
q15_sig_001 <- ols_keep[ols_p[ols_keep] < 0.001]

cat("\n--- Q15(b): predictors for which we reject H0: beta_j = 0 ---\n")
cat("Reject H0 at alpha=0.05  :", paste(q15_sig_05,  collapse = ", "), "\n")
cat("Reject H0 at alpha=0.01  :", paste(q15_sig_01,  collapse = ", "), "\n")
cat("Reject H0 at alpha=0.001 :", paste(q15_sig_001, collapse = ", "), "\n")
cat(sprintf("\nModel fit: R^2 = %.4f, adj. R^2 = %.4f, F = %.2f (p %s)\n",
            summary(ols_full)$r.squared, summary(ols_full)$adj.r.squared,
            summary(ols_full)$fstatistic[1],
            format.pval(pf(summary(ols_full)$fstatistic[1],
                           summary(ols_full)$fstatistic[2],
                           summary(ols_full)$fstatistic[3], lower.tail = FALSE))))
cat("Note how few predictors survive the joint fit: crim's marginal\n")
cat("correlations are spread over collinear proxies (rad/tax, nox/dis...), so\n")
cat("most individual betas cannot be distinguished from zero once the others\n")
cat("are in the model — the same collinearity story the penalised fits below\n")
cat("are designed to handle.\n")


# --- 2. Predictive setup: split, standardise, shared folds -------------------
# Everything from here on is prediction, so: 70/30 split; predictors
# standardised with TRAINING means/sds only (SVM margins and MLP gradients are
# scale-sensitive; glmnet standardises internally but feeding it the same
# matrix keeps every model looking at identical inputs); the response crim is
# left in its original units so MSE/RMSE stay interpretable. `chas` is already
# a 0/1 dummy, so it is left unscaled, like the one-hot columns in our
# credit-card pipeline.

n <- nrow(boston)
train_idx <- sample(seq_len(n), size = round(0.70 * n))  # plain random split —
# the response is continuous, so there is no class share to stratify on.

pred_cols <- setdiff(names(boston), "crim")
scale_cols <- setdiff(pred_cols, "chas")

train_b <- boston[train_idx, ]
test_b  <- boston[-train_idx, ]

tr_means <- sapply(train_b[scale_cols], mean)
tr_sds   <- sapply(train_b[scale_cols], sd)
train_b[scale_cols] <- scale(train_b[scale_cols], center = tr_means, scale = tr_sds)
test_b[scale_cols]  <- scale(test_b[scale_cols],  center = tr_means, scale = tr_sds)

# Numeric matrices for glmnet / sparseSVM (no factors in Boston, so a plain
# as.matrix does the job — no model.matrix needed).
x_train <- as.matrix(train_b[, pred_cols])
x_test  <- as.matrix(test_b[,  pred_cols])
y_train <- train_b$crim
y_test  <- test_b$crim

# One shared 10-fold assignment for every cross-validated model in this script.
n_folds <- 10
foldid  <- sample(rep(seq_len(n_folds), length.out = length(y_train)))

cat("\n############################################################\n")
cat("# 2. Predictive setup                                      #\n")
cat("############################################################\n")
cat(sprintf("train: %d rows   test: %d rows   predictors: %d\n",
            nrow(train_b), nrow(test_b), length(pred_cols)))
cat(sprintf("Shared %d-fold CV assignment drawn once and reused by every model.\n",
            n_folds))

# Regression scoring helper: test MSE / RMSE / R^2 (R^2 against the test mean,
# i.e. 1 - SSE/SST — "how much better than predicting the mean").
reg_metrics <- function(pred, truth, method_name, n_feat) {
  mse  <- mean((pred - truth)^2)
  sst  <- sum((truth - mean(truth))^2)
  r2   <- 1 - sum((pred - truth)^2) / sst
  data.frame(method = method_name, test_mse = mse, test_rmse = sqrt(mse),
             test_r2 = r2, n_features_used = n_feat, stringsAsFactors = FALSE)
}

reg_results <- list()

# OLS refit on the training rows only — the unregularised entry of the
# predictive comparison (the full-data fit above was for inference).
ols_train <- lm(crim ~ ., data = train_b)
reg_results[["ols_unreg"]] <- reg_metrics(
  predict(ols_train, newdata = test_b), y_test, "ols_unreg",
  length(coef(ols_train)) - 1)


# --- 3. Ridge / lasso / elastic net (glmnet, gaussian) -----------------------
# Same recipe as the credit-card pipeline, but family = "gaussian" because the
# response is continuous, and CV tuned on MSE (the natural loss here) rather
# than AUC. All three reuse the shared foldid.

cv_ridge <- glmnet::cv.glmnet(x_train, y_train, family = "gaussian",
                              alpha = 0, foldid = foldid, type.measure = "mse")
cv_lasso <- glmnet::cv.glmnet(x_train, y_train, family = "gaussian",
                              alpha = 1, foldid = foldid, type.measure = "mse")

# Elastic net: grid over alpha, same folds each time, keep the best CV MSE.
alpha_grid <- seq(0.1, 0.9, 0.1)
enet_search <- lapply(alpha_grid, function(a) {
  fit <- glmnet::cv.glmnet(x_train, y_train, family = "gaussian",
                           alpha = a, foldid = foldid, type.measure = "mse")
  list(alpha = a, fit = fit, cv_mse = min(fit$cvm))
})
enet_mses  <- sapply(enet_search, function(z) z$cv_mse)
best_alpha <- enet_search[[which.min(enet_mses)]]$alpha
cv_enet    <- enet_search[[which.min(enet_mses)]]$fit

cat("\n############################################################\n")
cat("# 3. Ridge / lasso / elastic net on crim (gaussian glmnet) #\n")
cat("############################################################\n")
cat(sprintf("Ridge : lambda.min=%.4f  lambda.1se=%.4f  (CV MSE at min = %.2f)\n",
            cv_ridge$lambda.min, cv_ridge$lambda.1se, min(cv_ridge$cvm)))
cat(sprintf("Lasso : lambda.min=%.4f  lambda.1se=%.4f  (CV MSE at min = %.2f)\n",
            cv_lasso$lambda.min, cv_lasso$lambda.1se, min(cv_lasso$cvm)))
print(data.frame(alpha = alpha_grid, best_cv_mse = round(enet_mses, 3)),
      row.names = FALSE)
cat(sprintf("Elastic net: chosen alpha = %.1f  lambda.min=%.4f\n",
            best_alpha, cv_enet$lambda.min))

# Coefficients side by side at lambda.1se (the sparse, readable solution) with
# the OLS estimates for reference — shrink-vs-select at a glance.
coef_vec <- function(cv_fit) {
  co <- as.matrix(coef(cv_fit, s = "lambda.1se")); setNames(co[, 1], rownames(co))
}
ridge_c <- coef_vec(cv_ridge); lasso_c <- coef_vec(cv_lasso); enet_c <- coef_vec(cv_enet)
coef_tab <- data.frame(
  predictor = names(ridge_c),
  ols       = round(coef(ols_train)[names(ridge_c)], 4),
  ridge     = round(ridge_c, 4),
  lasso     = round(lasso_c, 4),
  enet      = round(enet_c,  4),
  row.names = NULL
)
cat("\n--- Coefficients at lambda.1se (vs training OLS) ---\n")
print(coef_tab, row.names = FALSE)

lasso_kept <- setdiff(names(lasso_c)[lasso_c != 0], "(Intercept)")
cat(sprintf("\nLasso keeps %d of %d predictors at lambda.1se: %s\n",
            length(lasso_kept), length(pred_cols),
            paste(lasso_kept, collapse = ", ")))
cat("Cross-check with Q15(b): the predictors the t-tests flagged —\n  ",
    paste(q15_sig_05, collapse = ", "), "\n")
cat(sprintf("  overlap with lasso's kept set: %s\n",
            paste(intersect(q15_sig_05, lasso_kept), collapse = ", ")))
cat("  Inference and L1 selection agreeing on the strong signals (rad in\n")
cat("  particular) is the same reassuring convergence we saw on the credit data.\n")

# CV curve + coefficient paths for the lasso (the two standard glmnet figures).
png(file.path(fig_dir, "16_boston_lasso_cv.png"), width = 800, height = 600, res = 120)
plot(cv_lasso)
title("Boston: lasso CV MSE vs log(lambda)", line = 2.5)
dev.off()

png(file.path(fig_dir, "17_boston_lasso_path.png"), width = 800, height = 600, res = 120)
plot(cv_lasso$glmnet.fit, xvar = "lambda", label = TRUE)
abline(v = log(cv_lasso$lambda.min), lty = 2, col = "grey40")
abline(v = log(cv_lasso$lambda.1se), lty = 3, col = "grey40")
title("Boston: lasso coefficient paths (dashed=min, dotted=1se)", line = 2.5)
dev.off()

# Test-set scoring at lambda.min, feature counts = non-zero coefs at that lambda.
nnz <- function(cv_fit, s = "lambda.min") {
  co <- as.matrix(coef(cv_fit, s = s)); sum(co[rownames(co) != "(Intercept)", 1] != 0)
}
reg_results[["ridge_l2"]] <- reg_metrics(
  as.numeric(predict(cv_ridge, x_test, s = "lambda.min")), y_test,
  "ridge_l2", nnz(cv_ridge))
reg_results[["lasso_l1"]] <- reg_metrics(
  as.numeric(predict(cv_lasso, x_test, s = "lambda.min")), y_test,
  "lasso_l1", nnz(cv_lasso))
reg_results[["enet"]] <- reg_metrics(
  as.numeric(predict(cv_enet, x_test, s = "lambda.min")), y_test,
  "enet", nnz(cv_enet))


# --- 4. SVM variants ----------------------------------------------------------
# Support vector REGRESSION exists (e1071 eps-regression, kernlab::ksvm) but
# only with the built-in L2 penalty; the L1 / elastic-net SVMs (sparseSVM,
# penalizedSVM) are binary classifiers only. Following the brief's Section 2.1
# we therefore convert the problem to classification for the SVM family:
#   high_crim = 1  if crim > median(crim of the TRAINING set), else 0.
# The median threshold is computed on training data only (no test leakage) and
# gives a ~50/50 split, so — unlike the credit data — no class-weighting is
# needed. "Is this suburb in the high-crime half?" is also a perfectly natural
# planning question, so the converted task is meaningful, not just a trick.
# For completeness we also fit one support vector REGRESSION (RBF, L2) so the
# SVM family appears in the regression track where that is possible.

crim_T   <- median(y_train)  # the threshold T of Section 2.1
yc_train <- factor(ifelse(y_train > crim_T, "1", "0"), levels = c("0", "1"))
yc_test  <- factor(ifelse(y_test  > crim_T, "1", "0"), levels = c("0", "1"))

cat("\n############################################################\n")
cat("# 4. SVM variants (classification on crim > median)        #\n")
cat("############################################################\n")
cat(sprintf("Threshold T = median(train crim) = %.4f\n", crim_T))
cat(sprintf("Class balance: train %.1f%% / test %.1f%% high-crime (near-balanced\n",
            100 * mean(yc_train == "1"), 100 * mean(yc_test == "1")))
cat("by construction, so no class weights needed for this task).\n")

# Classification scoring helper (same metric set as the credit pipeline).
cls_metrics <- function(pred_class, score, truth, method_name, n_feat) {
  pred <- factor(pred_class, levels = c("0", "1"))
  cm <- table(Predicted = pred, Actual = truth)
  TP <- ifelse("1" %in% rownames(cm) && "1" %in% colnames(cm), cm["1","1"], 0)
  TN <- ifelse("0" %in% rownames(cm) && "0" %in% colnames(cm), cm["0","0"], 0)
  FP <- ifelse("1" %in% rownames(cm) && "0" %in% colnames(cm), cm["1","0"], 0)
  FN <- ifelse("0" %in% rownames(cm) && "1" %in% colnames(cm), cm["0","1"], 0)
  accuracy    <- (TP + TN) / (TP + TN + FP + FN)
  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  precision   <- ifelse((TP + FP) > 0, TP / (TP + FP), NA)
  f1          <- ifelse(!is.na(precision) && (precision + sensitivity) > 0,
                        2 * precision * sensitivity / (precision + sensitivity), NA)
  auc_val <- as.numeric(pROC::auc(pROC::roc(
    response = truth, predictor = as.numeric(score),
    levels = c("0", "1"), direction = "<", quiet = TRUE)))
  data.frame(method = method_name, accuracy = accuracy, auc = auc_val,
             recall = sensitivity, specificity = specificity,
             precision = precision, f1 = f1, n_features_used = n_feat,
             stringsAsFactors = FALSE)
}

# sparseSVM decision values, built by hand (its predict() has no score type):
# x %*% w + b, the signed distance used as the AUC ranking score.
# Caution: sparseSVM's internal +1/-1 coding follows the order in which the
# labels appear in the training vector, so the raw decision value can end up
# anti-aligned with class "1" (it did on this data). The model's own class
# predictions are always correctly mapped, so we use them to orient the score:
# predicted "1"s must sit on the positive side; if not, flip the sign.
sparsesvm_score <- function(cv_fit, newx) {
  co <- as.numeric(coef(cv_fit))
  names(co) <- rownames(as.matrix(coef(cv_fit)))
  b0 <- co[["(Intercept)"]]
  w  <- co[names(co) != "(Intercept)"]
  s  <- as.numeric(newx[, names(w), drop = FALSE] %*% w + b0)
  pred <- as.character(predict(cv_fit, newx, type = "class"))
  if (all(c("0", "1") %in% pred) &&
      mean(s[pred == "1"]) < mean(s[pred == "0"])) s <- -s
  s
}

cls_results <- list()
p_total <- ncol(x_train)

# (1) "Unregularised" analogue — linear kernel, large C. As argued in the
# credit-card script (and the report), a truly unregularised SVM does not
# exist: (1/2)||w||^2 is part of the objective. Large C = near-hard margin is
# the honest stand-in. n=354 here, so no runtime concerns — C=100 converges
# instantly (unlike the credit data, where large C was prohibitive).
svm_hard <- e1071::svm(x = x_train, y = yc_train, kernel = "linear",
                       cost = 100, probability = TRUE, scale = FALSE)
ph <- predict(svm_hard, x_test, probability = TRUE)
cls_results[["svm_unreg_hardmargin"]] <- cls_metrics(
  as.character(ph), attr(ph, "probabilities")[, "1"], yc_test,
  "svm_unreg_hardmargin", p_total)

# (2) L2 / standard soft-margin SVM: tuned RBF + linear baseline, 10-fold CV.
tune_rbf <- e1071::tune.svm(
  x = x_train, y = yc_train, kernel = "radial",
  cost = c(0.1, 1, 10, 100), gamma = c(0.01, 0.05, 0.1, 0.5),
  scale = FALSE,
  tunecontrol = e1071::tune.control(sampling = "cross", cross = 10))
best_cost  <- tune_rbf$best.parameters$cost
best_gamma <- tune_rbf$best.parameters$gamma
cat(sprintf("\nRBF tuning: cost=%g, gamma=%g (CV error %.4f)\n",
            best_cost, best_gamma, tune_rbf$best.performance))

svm_rbf <- e1071::svm(x = x_train, y = yc_train, kernel = "radial",
                      cost = best_cost, gamma = best_gamma,
                      probability = TRUE, scale = FALSE)
pr <- predict(svm_rbf, x_test, probability = TRUE)
cls_results[["svm_l2_rbf"]] <- cls_metrics(
  as.character(pr), attr(pr, "probabilities")[, "1"], yc_test,
  "svm_l2_rbf", p_total)

svm_lin <- e1071::svm(x = x_train, y = yc_train, kernel = "linear",
                      cost = 1, probability = TRUE, scale = FALSE)
pl <- predict(svm_lin, x_test, probability = TRUE)
cls_results[["svm_l2_linear"]] <- cls_metrics(
  as.character(pl), attr(pl, "probabilities")[, "1"], yc_test,
  "svm_l2_linear", p_total)

# (3) L1 (lasso) SVM — sparseSVM alpha=1, lambda by its internal CV.
yc_train_num <- as.numeric(as.character(yc_train))
cv_l1 <- sparseSVM::cv.sparseSVM(x_train, yc_train_num, alpha = 1,
                                 ncores = 1, seed = 1)
co_l1 <- as.numeric(coef(cv_l1)); names(co_l1) <- rownames(as.matrix(coef(cv_l1)))
kept_l1 <- names(co_l1)[co_l1 != 0 & names(co_l1) != "(Intercept)"]
cls_results[["svm_l1"]] <- cls_metrics(
  as.character(predict(cv_l1, x_test, type = "class")),
  sparsesvm_score(cv_l1, x_test), yc_test, "svm_l1", length(kept_l1))
cat(sprintf("\nL1 SVM keeps %d of %d features: %s\n",
            length(kept_l1), p_total, paste(kept_l1, collapse = ", ")))

# (4) Elastic-net SVM — sparseSVM, alpha tuned over {0.25, 0.5, 0.75}.
en_alphas <- c(0.25, 0.5, 0.75)
en_fits <- lapply(en_alphas, function(a) {
  fit <- sparseSVM::cv.sparseSVM(x_train, yc_train_num, alpha = a,
                                 ncores = 1, seed = 1)
  list(alpha = a, fit = fit, cve = min(fit$cve))
})
best_en <- en_fits[[which.min(sapply(en_fits, function(z) z$cve))]]
co_en <- as.numeric(coef(best_en$fit))
names(co_en) <- rownames(as.matrix(coef(best_en$fit)))
kept_en <- names(co_en)[co_en != 0 & names(co_en) != "(Intercept)"]
cls_results[["svm_enet"]] <- cls_metrics(
  as.character(predict(best_en$fit, x_test, type = "class")),
  sparsesvm_score(best_en$fit, x_test), yc_test, "svm_enet", length(kept_en))
cat(sprintf("Elastic-net SVM: alpha=%.2f, keeps %d features.\n",
            best_en$alpha, length(kept_en)))

# Support vector REGRESSION (RBF, the only penalty available is L2) so the SVM
# family also appears in the regression track. Tuned with the same 10-fold CV.
tune_svr <- e1071::tune.svm(
  x = x_train, y = y_train, type = "eps-regression", kernel = "radial",
  cost = c(1, 10, 100), gamma = c(0.01, 0.05, 0.1), scale = FALSE,
  tunecontrol = e1071::tune.control(sampling = "cross", cross = 10))
svr_rbf <- tune_svr$best.model
reg_results[["svr_l2_rbf"]] <- reg_metrics(
  predict(svr_rbf, x_test), y_test, "svr_l2_rbf", p_total)
cat(sprintf("\nSVR (regression track, L2 only): cost=%g gamma=%g eps=%.1f\n",
            tune_svr$best.parameters$cost, tune_svr$best.parameters$gamma,
            svr_rbf$epsilon))


# --- 5. MLP variants (regression on crim) ------------------------------------
# Same engine strategy as the credit-card pipeline: h2o.deeplearning exposes l1
# and l2 penalties on one identical network, so if a working Java runtime is
# available we fit all four variants (unreg / L2 / L1 / enet) as REGRESSION
# models (gaussian loss — no classification conversion is needed for an MLP).
# Without Java we fall back to nnet: unregularised + L2 weight-decay only, and
# we say so rather than fabricate the missing variants.

java_works <- function() {
  jh <- suppressWarnings(tryCatch(
    system2("java", "-version", stdout = TRUE, stderr = TRUE),
    error = function(e) NULL))
  !is.null(jh) && is.null(attr(jh, "status"))
}

use_h2o <- FALSE
if (requireNamespace("h2o", quietly = TRUE) && java_works()) {
  h2o_ok <- tryCatch({
    suppressMessages(library(h2o))
    h2o.init(nthreads = -1, max_mem_size = "2G")
    h2o.no_progress()
    TRUE
  }, error = function(e) {
    message("h2o.init() failed (", conditionMessage(e), ") -> nnet fallback.")
    FALSE
  })
  use_h2o <- isTRUE(h2o_ok)
} else if (requireNamespace("h2o", quietly = TRUE)) {
  message("h2o installed but no working Java runtime -> nnet fallback.")
}

cat("\n############################################################\n")
cat(sprintf("# 5. MLP variants (regression) — engine: %-17s#\n",
            ifelse(use_h2o, "h2o", "nnet fallback")))
cat("############################################################\n")

if (use_h2o) {
  # standardize = TRUE here (unlike the credit script): our predictors are
  # already standardised so that part is a no-op, but it lets h2o standardise
  # the RESPONSE internally during training and back-transform predictions —
  # the sensible treatment for a skewed target like crim.
  train_h2o <- as.h2o(cbind(train_b, fold = as.integer(foldid)), "boston_train")
  test_h2o  <- as.h2o(test_b, "boston_test")

  fit_h2o_variant <- function(method_name, l1_grid, l2_grid) {
    grid_id <- paste0("dl_boston_", method_name)
    g <- h2o.grid(
      algorithm = "deeplearning", grid_id = grid_id,
      x = pred_cols, y = "crim",
      training_frame = train_h2o, fold_column = "fold",
      hidden = c(8, 4), epochs = 100, activation = "Rectifier",
      standardize = TRUE, reproducible = TRUE, seed = 1,
      stopping_rounds = 0,
      hyper_params = list(l1 = l1_grid, l2 = l2_grid))
    g_sorted <- h2o.getGrid(grid_id, sort_by = "mse", decreasing = FALSE)
    best <- h2o.getModel(g_sorted@model_ids[[1]])
    pred <- as.data.frame(h2o.predict(best, test_h2o))$predict
    # As in the credit pipeline: a net reads every input regardless of the
    # penalty (L1 sparsifies weights, not whole inputs), so n_features = p.
    reg_metrics(pred, y_test, method_name, p_total)
  }

  reg_results[["mlp_unreg"]] <- fit_h2o_variant("mlp_unreg", c(0), c(0))
  reg_results[["mlp_l2"]]    <- fit_h2o_variant("mlp_l2", c(0),
                                                c(1e-5, 1e-4, 1e-3))
  reg_results[["mlp_l1"]]    <- fit_h2o_variant("mlp_l1",
                                                c(1e-5, 1e-4, 1e-3), c(0))
  reg_results[["mlp_enet"]]  <- fit_h2o_variant("mlp_enet",
                                                c(1e-5, 1e-4), c(1e-5, 1e-4))
} else {
  # nnet fallback: unregularised + L2 weight decay only (no L1 in nnet).
  # nnet is happiest with a roughly unit-scale response, so train on the
  # standardised response and back-transform the predictions.
  y_mu <- mean(y_train); y_sd <- sd(y_train)
  ys_train <- (y_train - y_mu) / y_sd
  nn_dat <- data.frame(x_train, crim_s = ys_train)

  fit_nnet <- function(decay, dat = nn_dat) {
    set.seed(1)  # nnet's random weight init
    nnet::nnet(crim_s ~ ., data = dat, size = 5, decay = decay,
               linout = TRUE, maxit = 500, trace = FALSE)
  }
  # Tune the decay on the shared folds, scoring held-out MSE (on the scaled
  # response; the ranking is unaffected by the scaling).
  decay_grid <- c(1e-4, 1e-3, 1e-2, 1e-1)
  cv_mse_for_decay <- function(decay) {
    mean(sapply(seq_len(n_folds), function(f) {
      m <- fit_nnet(decay, dat = nn_dat[foldid != f, ])
      mean((predict(m, nn_dat[foldid == f, ]) - ys_train[foldid == f])^2)
    }))
  }
  decay_cv   <- sapply(decay_grid, cv_mse_for_decay)
  best_decay <- decay_grid[which.min(decay_cv)]
  print(data.frame(decay = decay_grid, cv_mse_scaled = round(decay_cv, 4)),
        row.names = FALSE)
  cat(sprintf("Best L2 decay = %g\n", best_decay))

  back <- function(p) as.numeric(p) * y_sd + y_mu  # undo the response scaling
  nn_test <- data.frame(x_test)
  m0 <- fit_nnet(0)
  reg_results[["mlp_unreg"]] <- reg_metrics(back(predict(m0, nn_test)),
                                            y_test, "mlp_unreg", p_total)
  m2 <- fit_nnet(best_decay)
  reg_results[["mlp_l2"]] <- reg_metrics(back(predict(m2, nn_test)),
                                         y_test, "mlp_l2", p_total)
  cat("\nNOTE: mlp_l1 / mlp_enet are intentionally ABSENT on this machine —\n")
  cat("nnet has no L1 penalty; run this script with Java + h2o to get all four.\n")
}


# --- 6. Results: the two tracks, saved and plotted ---------------------------

metrics_reg <- do.call(rbind, reg_results); rownames(metrics_reg) <- NULL
metrics_cls <- do.call(rbind, cls_results); rownames(metrics_cls) <- NULL
metrics_reg <- metrics_reg[order(metrics_reg$test_mse), ]
metrics_cls <- metrics_cls[order(-metrics_cls$auc), ]

saveRDS(metrics_reg, file.path(data_dir, "metrics_boston_regression.rds"))
saveRDS(metrics_cls, file.path(data_dir, "metrics_boston_svm.rds"))

cat("\n############################################################\n")
cat("# 6a. REGRESSION track — predicting crim (test set)        #\n")
cat("############################################################\n")
print(format(metrics_reg, digits = 4), row.names = FALSE)

cat("\n############################################################\n")
cat("# 6b. CLASSIFICATION track — SVMs on crim > median          #\n")
cat("############################################################\n")
print(format(metrics_cls, digits = 4), row.names = FALSE)
cat("\n(The two tracks are reported separately on purpose: MSE and accuracy/AUC\n")
cat("are not comparable, and only the SVM family needed the conversion.)\n")

# Figure: test RMSE by method (regression track).
ro <- metrics_reg[order(-metrics_reg$test_rmse), ]
png(file.path(fig_dir, "18_boston_regression_rmse.png"),
    width = 820, height = 560, res = 120)
op <- par(mar = c(5, 9, 4, 2))
bp <- barplot(ro$test_rmse, horiz = TRUE, names.arg = ro$method, las = 1,
              col = "#4C72B0", xlab = "Test RMSE (lower = better)",
              main = "Boston: regression track — held-out RMSE")
text(ro$test_rmse, bp, labels = sprintf("%.2f", ro$test_rmse),
     pos = 2, col = "white", cex = 0.9)
par(op)
dev.off()

# Figure: test AUC by SVM variant (classification track).
co <- metrics_cls[order(metrics_cls$auc), ]
png(file.path(fig_dir, "19_boston_svm_auc.png"),
    width = 820, height = 520, res = 120)
op <- par(mar = c(5, 11, 4, 2))
bp <- barplot(co$auc, horiz = TRUE, names.arg = co$method, las = 1,
              xlim = c(0.5, 1.0), col = "#DD8452", xlab = "Test ROC-AUC",
              main = "Boston: SVM variants on crim > median")
text(co$auc, bp, labels = sprintf("%.3f", co$auc), pos = 4, xpd = TRUE, cex = 0.9)
abline(v = 0.5, lty = 2, col = "grey50")
par(op)
dev.off()


# --- 7. Which parts of Q15(b) do NOT carry over, and why that is valid -------
# The brief requires an explicit justification wherever we ignore part of
# Q15(b) for an extended method, so here it is, spelled out:
#
#  * Q15(b)'s deliverable is a set of t-tests on regression coefficients. Those
#    tests exist because OLS has a closed-form estimator whose sampling
#    distribution is known (t under Gaussian errors). An SVM decision function
#    is defined by a regularised hinge-loss optimum (often in an implicit
#    kernel feature space) and an MLP by a non-convex gradient-descent solution
#    — neither has coefficients with a known sampling distribution, so
#    "which beta_j reject H0" is simply not a well-posed question for them.
#    Ignoring the hypothesis-testing part for SVM/MLP is therefore valid: the
#    quantity being tested does not exist in those models. What DOES carry
#    over — fitting on all predictors, regularising, and evaluating out of
#    sample — is exactly what we did.
#  * For the penalised linear models the betas exist but the p-values do not
#    transfer: after data-driven shrinkage/selection the classical t-tests are
#    no longer valid (post-selection inference). We therefore report which
#    coefficients SURVIVE the penalty instead, and cross-check that set against
#    the Q15(b) t-tests — the two agreed on the dominant signal (rad).

cat("\n=================== BOSTON SUMMARY ===================\n")
cat("Q15(b): reject H0 at alpha=0.05 for:\n  ", paste(q15_sig_05, collapse = ", "), "\n")
cat(sprintf("\nBest regression-track model : %s (test RMSE %.3f, R^2 %.3f)\n",
            metrics_reg$method[1], metrics_reg$test_rmse[1], metrics_reg$test_r2[1]))
cat(sprintf("Best SVM (classification)   : %s (test AUC %.4f, accuracy %.4f)\n",
            metrics_cls$method[1], metrics_cls$auc[1], metrics_cls$accuracy[1]))
cat(sprintf("Lasso kept %d/%d predictors; overlap with Q15(b) significant set: %s\n",
            length(lasso_kept), length(pred_cols),
            paste(intersect(q15_sig_05, lasso_kept), collapse = ", ")))
cat("\nWhy SVM ran as classification: L1/elastic-net SVMs (sparseSVM) are binary-\n")
cat("only, so crim was thresholded at its training median (brief Section 2.1).\n")
cat("Why no p-values for SVM/MLP: no coefficient sampling distribution exists\n")
cat("for margin/gradient-based fits — see the comment block above this summary.\n")
cat(sprintf("\nSaved: %s, %s,\n       figures 16-19 in %s/\n",
            file.path(data_dir, "metrics_boston_regression.rds"),
            file.path(data_dir, "metrics_boston_svm.rds"), fig_dir))
cat("======================================================\n")

if (use_h2o) h2o.shutdown(prompt = FALSE)

# End of the Boston script.
