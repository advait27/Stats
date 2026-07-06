# stl_Kokru_Dharmadhikari_Sinha_svm.R
#
# MIS41120 Statistical Learning — Practical Assignment, Stage 5: SVMs.
# We fit the four "regularisation variants" the brief asks for, but for an SVM
# these don't map the way they do for OLS/logistic, so a note on that first.
#
# How the four variants translate to SVMs
# ----------------------------------------
# An SVM is never really "unregularised". The soft-margin SVM minimises
#   (1/2)||w||^2 + C * sum(hinge losses),
# and that (1/2)||w||^2 is already an L2 penalty on the weights. The cost C
# just sets how hard we fit: small C = wide margin / strong regularisation,
# large C = narrow margin, closer to a hard margin. So:
#   (1) "unregularised"  -> linear kernel with a large C (near hard margin).
#                           There's no true unregularised SVM since ||w||^2 is
#                           baked in; large C is the closest honest stand-in.
#   (2) L2 / standard    -> ordinary soft-margin SVM. We do an RBF (nonlinear
#                           boundary) plus a plain linear kernel as a like-for-
#                           like baseline against the sparse models.
#   (3) L1 / lasso       -> swap the L2 penalty for L1 so weights hit exactly
#                           zero (feature selection). sparseSVM, alpha=1.
#   (4) elastic net      -> L1+L2 mix on the weights. sparseSVM, 0<alpha<1.
#
# Inputs come from data/ (x_train, y_train, x_test, y_test in model-matrix form).
# Seeded, relative paths, run from the project root.


# --- Setup: seed, packages, output folders ---
set.seed(1)  # fixes the tuning subsample and the internal CV draws

ensure_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  library(pkg, character.only = TRUE)
}

# e1071 gives us svm()/tune.svm() for the kernel SVMs; sparseSVM does the
# L1/elastic-net linear fits with variable selection; pROC for the AUC.
ensure_pkg("e1071")
ensure_pkg("sparseSVM")
ensure_pkg("pROC")

fig_dir  <- "figures"
data_dir <- "data"
if (!dir.exists(fig_dir))  dir.create(fig_dir,  recursive = TRUE)
if (!dir.exists(data_dir)) dir.create(data_dir, recursive = TRUE)


# --- Load the Stage-2 design matrices ---
# One-hot numeric matrices, no intercept, so every SVM (e1071 kernels and the
# sparseSVM linear fits alike) sees the same features on the same scale. Scaling
# matters here because the margin and the RBF kernel are distance-based, so an
# unscaled big-range feature would dominate. Stage 2 already standardised the
# continuous columns using the training set only.
x_train <- readRDS(file.path(data_dir, "x_train.rds"))
y_train <- readRDS(file.path(data_dir, "y_train.rds"))
x_test  <- readRDS(file.path(data_dir, "x_test.rds"))
y_test  <- readRDS(file.path(data_dir, "y_test.rds"))

stopifnot(identical(levels(y_train), c("0", "1")),
          identical(colnames(x_train), colnames(x_test)))

p_total <- ncol(x_train)  # feature count, reused as n_features for the kernel models

# Inverse-frequency class weights for the ~22% imbalance, otherwise the SVM just
# leans towards the majority "no default" class.
cls_counts  <- table(y_train)
class_wts   <- sum(cls_counts) / (length(cls_counts) * cls_counts)
class_wts   <- setNames(as.numeric(class_wts), names(cls_counts))

# Runtime caveat, and why we subsample the e1071 fits.
# e1071::svm runs a dense QP solver that scales badly in n (roughly n^2, and
# worse still as C grows). On this data one fit is under a second at n=2,000,
# about 7s at n=8,000, but minutes at the full n=21,001 — and a large-C
# near-hard-margin fit on the full set basically never finishes (>10 min). Not
# viable for a script that has to run out of the box.
# So the e1071 kernel/linear SVMs train on an 8,000-row stratified subsample
# (default rate preserved), and tuning happens on a smaller 3,000-row nested
# subsample. The sparseSVM L1/elastic-net fits are cheap, so those use the full
# training set. Every model is scored on the same full held-out test set, so the
# comparison stays fair — only the amount of e1071 training data changes.
# The subsamples are pinned by set.seed(1).

# Draw `size` row indices from y, keeping the class proportions.
stratified_idx <- function(y, size) {
  if (length(y) <= size) return(seq_along(y))
  i0 <- which(y == "0"); i1 <- which(y == "1")
  n1 <- round(size * length(i1) / length(y))
  n0 <- size - n1
  sort(c(sample(i0, n0), sample(i1, n1)))
}

# Training subsample for the e1071 SVMs.
svm_n   <- 8000
svm_idx <- stratified_idx(y_train, svm_n)
x_svm   <- x_train[svm_idx, , drop = FALSE]
y_svm   <- y_train[svm_idx]

# Nested subsample drawn from within the e1071 training rows, used only for the
# RBF cost x gamma grid search.
tune_n  <- 3000
tune_rel <- stratified_idx(y_svm, tune_n)  # indices relative to x_svm
x_tune  <- x_svm[tune_rel, , drop = FALSE]
y_tune  <- y_svm[tune_rel]


# Metric helper, same one used in the earlier stages. `score` is whatever we
# rank on for AUC — a probability, or an SVM decision value.
eval_metrics <- function(pred_class, score, truth, method_name, n_feat) {
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
  bal_acc     <- (sensitivity + specificity) / 2
  roc_obj     <- pROC::roc(response = truth, predictor = as.numeric(score),
                           levels = c("0","1"), direction = "<", quiet = TRUE)
  auc_val     <- as.numeric(pROC::auc(roc_obj))

  data.frame(method = method_name, accuracy = accuracy, auc = auc_val,
             recall = sensitivity, specificity = specificity,
             precision = precision, f1 = f1, balanced_accuracy = bal_acc,
             n_features_used = n_feat, stringsAsFactors = FALSE)
}

# Decision-value helper for sparseSVM. predict.cv.sparseSVM has no
# type="response", only class/coefficients/nvars, so we build the decision value
# by hand as x %*% w + intercept at the CV-selected lambda. That's the signed
# distance to the hyperplane — fine as a ranking score for AUC (its sign gives
# back the predicted class), but it's not a calibrated probability.
sparsesvm_score <- function(cv_fit, newx) {
  co <- as.numeric(coef(cv_fit))
  names(co) <- rownames(as.matrix(coef(cv_fit)))
  b0 <- co[["(Intercept)"]]
  w  <- co[names(co) != "(Intercept)"]
  as.numeric(newx[, names(w), drop = FALSE] %*% w + b0)
}

results <- list()  # one metrics row per model


# --- 1. "Unregularised" analogue: linear kernel, large cost ---
# There's no truly unregularised SVM (the ||w||^2 term is built into the
# objective), so the nearest thing is a linear SVM with a large C, pushed
# towards a hard margin. probability=TRUE gives us Platt-scaled scores for AUC.
#
# On why C=10: on this non-separable data the libsvm runtime is dominated by C
# and blows up as it grows (on the 8k subsample: C=10 ~15s, C=25 ~27s, C=50 ~44s
# with a max-iterations non-convergence warning; C=1e3 didn't finish in minutes).
# But the support-vector count — the actual boundary — is basically flat from
# C>=5 (nSV around 5,980 at C=5,10,25,50), so cranking C higher doesn't harden
# the margin any further, it just costs runtime and convergence. C=10 already
# sits in the weak-regularisation / near-hard-margin regime while staying fast
# and convergent, so it's the practical stand-in for the (ill-defined)
# unregularised SVM. Trained on the 8k subsample.
hard_cost <- 10
cat("\n############################################################\n")
cat("# 1. 'Unregularised' analogue: linear SVM, large C=10     #\n")
cat("############################################################\n")
svm_hard <- e1071::svm(x = x_svm, y = y_svm, kernel = "linear",
                       cost = hard_cost, class.weights = class_wts,
                       probability = TRUE, scale = FALSE)  # already standardised
ph <- predict(svm_hard, x_test, probability = TRUE)
prob_hard <- attr(ph, "probabilities")[, "1"]              # P(default) column
results[["svm_unreg_hardmargin"]] <- eval_metrics(
  pred_class = as.character(ph), score = prob_hard, truth = y_test,
  method_name = "svm_unreg_hardmargin", n_feat = p_total)
cat(sprintf("Done. Support vectors: %d. (Linear kernel uses all %d features.)\n",
            svm_hard$tot.nSV, p_total))


# --- 2. L2 standard SVM: tuned RBF + linear baseline ---
# The plain soft-margin SVM (its (1/2)||w||^2 is the L2 penalty). Tune the RBF's
# cost and gamma with e1071's 10-fold CV on the small tuning subsample, then
# refit the winner on the 8k training subsample. class.weights for imbalance,
# probability=TRUE for AUC.
cat("\n############################################################\n")
cat("# 2. L2 standard SVM: RBF (tuned) + linear baseline       #\n")
cat("############################################################\n")

# Grid search over cost x gamma. Grids kept small for runtime; they bracket the
# usual sensible values.
cat("Tuning RBF SVM (cost x gamma) on a stratified subsample ...\n")
tune_rbf <- e1071::tune.svm(
  x = x_tune, y = y_tune, kernel = "radial",
  cost  = c(0.1, 1, 10),
  gamma = c(0.01, 0.05, 0.1),
  class.weights = class_wts, scale = FALSE,
  tunecontrol = e1071::tune.control(sampling = "cross", cross = 10)
)
best_cost  <- tune_rbf$best.parameters$cost
best_gamma <- tune_rbf$best.parameters$gamma
cat(sprintf("Best RBF params: cost=%g, gamma=%g (CV error=%.4f on subsample)\n",
            best_cost, best_gamma, tune_rbf$best.performance))

# Refit the winning RBF on the full 8k training subsample.
cat("Refitting RBF on the 8k SVM-training subsample ...\n")
svm_rbf <- e1071::svm(x = x_svm, y = y_svm, kernel = "radial",
                      cost = best_cost, gamma = best_gamma,
                      class.weights = class_wts, probability = TRUE, scale = FALSE)
pr <- predict(svm_rbf, x_test, probability = TRUE)
prob_rbf <- attr(pr, "probabilities")[, "1"]
# A kernel model isn't feature-sparse — the RBF uses all p features implicitly —
# so n_features_used is just p_total.
results[["svm_l2_rbf"]] <- eval_metrics(
  pred_class = as.character(pr), score = prob_rbf, truth = y_test,
  method_name = "svm_l2_rbf", n_feat = p_total)
cat(sprintf("RBF support vectors: %d (of %d subsample train rows).\n",
            svm_rbf$tot.nSV, nrow(x_svm)))

# Linear-kernel L2 SVM at moderate cost. This is the like-for-like linear
# baseline for the sparse models below — same kernel family, no selection.
cat("Fitting linear-kernel L2 SVM (cost=1) ...\n")
svm_lin <- e1071::svm(x = x_svm, y = y_svm, kernel = "linear",
                      cost = 1, class.weights = class_wts,
                      probability = TRUE, scale = FALSE)
pl <- predict(svm_lin, x_test, probability = TRUE)
prob_lin <- attr(pl, "probabilities")[, "1"]
results[["svm_l2_linear"]] <- eval_metrics(
  pred_class = as.character(pl), score = prob_lin, truth = y_test,
  method_name = "svm_l2_linear", n_feat = p_total)


# --- 3. L1 (lasso) SVM: sparseSVM, alpha = 1 ---
# sparseSVM is a linear SVM with an elastic-net penalty; alpha=1 is pure L1, so
# some weights go to exactly zero and we get feature selection. Target is already
# binary ("0"/"1"), so no regression-to-classification step needed (assignment
# 2.1). lambda is chosen by the package's own cross-validation.
cat("\n############################################################\n")
cat("# 3. L1 (lasso) SVM: sparseSVM alpha=1                    #\n")
cat("############################################################\n")
# sparseSVM wants a numeric response.
y_train_num <- as.numeric(as.character(y_train))
y_test_num  <- as.numeric(as.character(y_test))

cv_l1 <- sparseSVM::cv.sparseSVM(x_train, y_train_num, alpha = 1,
                                 ncores = 1, seed = 1)
# Coefficients at the CV-selected lambda (first row is the intercept).
coef_l1 <- as.numeric(coef(cv_l1))
names(coef_l1) <- rownames(as.matrix(coef(cv_l1)))
w_l1 <- coef_l1[names(coef_l1) != "(Intercept)"]
kept_l1 <- names(w_l1)[w_l1 != 0]
n_l1    <- length(kept_l1)
cat(sprintf("At CV-selected lambda: %d of %d weights non-zero.\n", n_l1, length(w_l1)))
cat("KEPT (selected features):\n  ",
    paste(kept_l1, collapse = ", "), "\n")

# Class labels, plus a decision-value score for AUC. Reminder: that score is a
# signed distance to the hyperplane, not a calibrated probability — only good as
# a ranking for ROC-AUC.
pred_l1  <- as.character(predict(cv_l1, x_test, type = "class"))
score_l1 <- sparsesvm_score(cv_l1, x_test)  # decision value (ranking score, not prob)
results[["svm_l1"]] <- eval_metrics(
  pred_class = pred_l1, score = score_l1, truth = y_test,
  method_name = "svm_l1", n_feat = n_l1)


# --- 4. Elastic-net SVM: sparseSVM, alpha in {0.25, 0.5, 0.75} ---
# L1+L2 blend on the weights. Run cv.sparseSVM at each alpha and keep the one
# whose best CV model has the lowest CV error.
cat("\n############################################################\n")
cat("# 4. Elastic-net SVM: sparseSVM alpha in {0.25,0.5,0.75}  #\n")
cat("############################################################\n")
enet_alphas <- c(0.25, 0.5, 0.75)
enet_fits <- lapply(enet_alphas, function(a) {
  fit <- sparseSVM::cv.sparseSVM(x_train, y_train_num, alpha = a,
                                 ncores = 1, seed = 1)
  list(alpha = a, fit = fit, cve = min(fit$cve))  # min of the CV error curve
})
enet_cves  <- sapply(enet_fits, function(z) z$cve)
best_j     <- which.min(enet_cves)
best_alpha <- enet_fits[[best_j]]$alpha
cv_enet    <- enet_fits[[best_j]]$fit

enet_tab <- data.frame(alpha = enet_alphas, best_cv_error = round(enet_cves, 5))
print(enet_tab, row.names = FALSE)
cat(sprintf("Chosen elastic-net alpha = %.2f\n", best_alpha))

coef_en <- as.numeric(coef(cv_enet))
names(coef_en) <- rownames(as.matrix(coef(cv_enet)))
w_en   <- coef_en[names(coef_en) != "(Intercept)"]
kept_en <- names(w_en)[w_en != 0]
n_en    <- length(kept_en)
cat(sprintf("At CV-selected lambda: %d of %d weights non-zero.\n", n_en, length(w_en)))
cat("KEPT (selected features):\n  ",
    paste(kept_en, collapse = ", "), "\n")

pred_en  <- as.character(predict(cv_enet, x_test, type = "class"))
score_en <- sparsesvm_score(cv_enet, x_test)  # decision value (ranking score, not prob)
results[["svm_enet"]] <- eval_metrics(
  pred_class = pred_en, score = score_en, truth = y_test,
  method_name = "svm_enet", n_feat = n_en)


# --- 5. Collect the results, compare, save ---
metrics_svm <- do.call(rbind, results[c(
  "svm_unreg_hardmargin", "svm_l2_rbf", "svm_l2_linear", "svm_l1", "svm_enet")])
rownames(metrics_svm) <- NULL

# Lock the column schema so Stage 7 can rbind every stage together.
stopifnot(identical(
  names(metrics_svm),
  c("method","accuracy","auc","recall","specificity","precision","f1",
    "balanced_accuracy","n_features_used")))
saveRDS(metrics_svm, file.path(data_dir, "metrics_svm.rds"))

cat("\n############################################################\n")
cat("# 5. TEST-SET METRICS (all SVMs, threshold 0.5)          #\n")
cat("############################################################\n")
print(format(metrics_svm, digits = 4), row.names = FALSE)

# The main question for this stage: does the nonlinear RBF boundary actually
# beat the linear/penalised models?
auc_rbf <- metrics_svm$auc[metrics_svm$method == "svm_l2_rbf"]
auc_lin <- metrics_svm$auc[metrics_svm$method == "svm_l2_linear"]
auc_l1  <- metrics_svm$auc[metrics_svm$method == "svm_l1"]
cat(sprintf("\nNarrative — nonlinear vs linear:\n"))
cat(sprintf("  RBF (nonlinear)  AUC = %.4f, recall = %.4f\n",
            auc_rbf, metrics_svm$recall[metrics_svm$method=="svm_l2_rbf"]))
cat(sprintf("  Linear L2        AUC = %.4f\n", auc_lin))
cat(sprintf("  L1 sparse        AUC = %.4f (kept %d features)\n",
            auc_l1, metrics_svm$n_features_used[metrics_svm$method=="svm_l1"]))
if (auc_rbf > auc_lin + 0.005) {
  cat("  -> The RBF's nonlinear boundary clearly beats the linear models:\n")
  cat("     there is nonlinear structure worth exploiting, though the price is a\n")
  cat("     black-box, non-sparse model using all features implicitly.\n")
} else {
  cat("  -> The RBF doesn't really beat the linear/penalised SVMs: the default\n")
  cat("     boundary is close to linear here, so the simpler sparse SVM wins on\n")
  cat("     interpretability — similar AUC, far fewer features, readable coefs.\n")
}
cat("Reminder: sparseSVM 'scores' are signed distances to the hyperplane, not\n")
cat("calibrated probabilities — only used as a ranking for ROC-AUC.\n")


# --- 6. Figure: test AUC across the SVM variants ---
auc_order <- metrics_svm[order(metrics_svm$auc), ]
png(file.path(fig_dir, "11_svm_auc.png"), width = 820, height = 560, res = 120)
op <- par(mar = c(5, 11, 4, 2))  # extra left margin for the method labels
bp <- barplot(auc_order$auc, horiz = TRUE, names.arg = auc_order$method,
              las = 1, xlim = c(0.5, 0.8), col = "#4C72B0",
              xlab = "Test ROC-AUC",
              main = "SVM variants — held-out test AUC")
text(auc_order$auc, bp, labels = sprintf("%.3f", auc_order$auc),
     pos = 4, xpd = TRUE, cex = 0.9)
abline(v = 0.5, lty = 2, col = "grey50")  # chance line
par(op)
dev.off()


# --- 7. Console summary ---
cat("\n\n=================== STAGE 5 SVM SUMMARY ===================\n")
cat(sprintf("e1071 SVMs trained on : %d-row stratified subsample (tuned on %d);\n",
            nrow(x_svm), nrow(x_tune)))
cat(sprintf("                        sparseSVM on full %d; ALL evaluated on full test %d.\n",
            nrow(x_train), nrow(x_test)))
cat(sprintf("RBF chosen params     : cost=%g, gamma=%g\n", best_cost, best_gamma))
cat(sprintf("L1 SVM features kept  : %d ; elastic-net (alpha=%.2f) kept %d\n",
            n_l1, best_alpha, n_en))
cat("\nTest metrics (threshold 0.5):\n")
for (i in seq_len(nrow(metrics_svm))) {
  m <- metrics_svm[i, ]
  cat(sprintf("  %-20s : acc=%.4f AUC=%.4f recall=%.4f prec=%.4f F1=%.4f bal=%.4f feat=%d\n",
              m$method, m$accuracy, m$auc, m$recall, m$precision, m$f1,
              m$balanced_accuracy, m$n_features_used))
}
cat(sprintf("\nSaved: %s and %s\n",
            file.path(data_dir, "metrics_svm.rds"),
            file.path(fig_dir,  "11_svm_auc.png")))
cat("==========================================================\n")

# End of Stage 5 SVM script.
