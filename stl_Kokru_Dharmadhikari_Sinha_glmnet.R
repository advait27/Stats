# stl_Kokru_Dharmadhikari_Sinha_glmnet.R
# MIS41120 Statistical Learning — Practical Assignment
#
# Stage 4: penalised logistic regression via glmnet (ridge / lasso / elastic net).
# Picks up from the Stage-3 unregularised baseline and adds an L1/L2 penalty:
#   ridge (alpha=0)  L2, shrinks everything but keeps all predictors
#   lasso (alpha=1)  L1, zeros some coefficients so it doubles as selection
#   elastic net      L1+L2 blend, tune alpha in between
# family = "binomial" throughout since `default` is binary — this is penalised
# logistic, not the default gaussian glmnet.
#
# The whole point of the comparison is that every cv.glmnet() call reuses the
# same foldid saved in Stage 2. Same partitions for ridge, lasso and every alpha
# in the enet grid, so any AUC gap is down to the method and not the split.
#
# Inputs from data/: the Stage-2 model matrices (x/y train+test) and cv_foldid.
# Relative paths, seeded, runs on its own.


# Setup: seed, packages, output folders
set.seed(1)  # foldid already pins glmnet's CV, but seed the rest for safety

ensure_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  library(pkg, character.only = TRUE)
}

# glmnet for the penalised fits, pROC for test-set AUC (same as Stage 3).
ensure_pkg("glmnet")
ensure_pkg("pROC")

fig_dir  <- "figures"
data_dir <- "data"
if (!dir.exists(fig_dir))  dir.create(fig_dir,  recursive = TRUE)
if (!dir.exists(data_dir)) dir.create(data_dir, recursive = TRUE)


# If the Stage-2 outputs aren't on disk yet (fresh checkout), run the prep
# script now — it only needs the raw .xls and writes everything we load below.
# Re-seed afterwards so this script behaves identically either way.
if (!all(file.exists(file.path(data_dir, c("x_train.rds", "y_train.rds",
                                           "x_test.rds", "y_test.rds",
                                           "cv_foldid.rds"))))) {
  message("Stage-2 outputs not found in data/ — running the prep script first ...")
  source("stl_Kokru_Dharmadhikari_Sinha_prep.R")
  set.seed(1)
}

# Load the Stage-2 matrices and the shared fold vector.
# glmnet wants a numeric design matrix and a response. Reuse the one-hot,
# no-intercept matrices from Stage 2 (glmnet adds its own intercept and
# standardises internally). y_* are factors c("0","1") with 1 = default; glmnet
# treats the second level as the positive class.
x_train <- readRDS(file.path(data_dir, "x_train.rds"))
y_train <- readRDS(file.path(data_dir, "y_train.rds"))
x_test  <- readRDS(file.path(data_dir, "x_test.rds"))
y_test  <- readRDS(file.path(data_dir, "y_test.rds"))

# One fold number per training row, from the stratified folds every model shares.
foldid <- readRDS(file.path(data_dir, "cv_foldid.rds"))

stopifnot(nrow(x_train) == length(y_train),
          length(foldid) == length(y_train),
          identical(levels(y_train), c("0", "1")))

# So type = "response" gives P(default = 1), matching the Stage-3 convention.


# Helpers, so every model gets scored the same way Stage 3 was.
# Full metric set from predicted probabilities and true labels.
eval_metrics <- function(prob, truth, method_name, n_feat) {
  pred <- factor(ifelse(prob >= 0.5, "1", "0"), levels = c("0", "1"))
  cm   <- table(Predicted = pred, Actual = truth)
  TP <- ifelse("1" %in% rownames(cm) && "1" %in% colnames(cm), cm["1","1"], 0)
  TN <- ifelse("0" %in% rownames(cm) && "0" %in% colnames(cm), cm["0","0"], 0)
  FP <- ifelse("1" %in% rownames(cm) && "0" %in% colnames(cm), cm["1","0"], 0)
  FN <- ifelse("0" %in% rownames(cm) && "1" %in% colnames(cm), cm["0","1"], 0)

  accuracy    <- (TP + TN) / (TP + TN + FP + FN)
  sensitivity <- TP / (TP + FN)                        # aka recall
  specificity <- TN / (TN + FP)
  precision   <- ifelse((TP + FP) > 0, TP / (TP + FP), NA)
  f1          <- ifelse(!is.na(precision) && (precision + sensitivity) > 0,
                        2 * precision * sensitivity / (precision + sensitivity), NA)
  bal_acc     <- (sensitivity + specificity) / 2
  roc_obj     <- pROC::roc(response = truth, predictor = as.numeric(prob),
                           levels = c("0","1"), direction = "<", quiet = TRUE)
  auc_val     <- as.numeric(pROC::auc(roc_obj))

  data.frame(method = method_name, accuracy = accuracy, auc = auc_val,
             recall = sensitivity, specificity = specificity,
             precision = precision, f1 = f1, balanced_accuracy = bal_acc,
             n_features_used = n_feat, stringsAsFactors = FALSE)
}

# Non-zero coefficients (dropping the intercept) at a given lambda. Our stand-in
# for model complexity — fewer features, simpler model.
n_nonzero <- function(cv_fit, s) {
  co <- as.matrix(coef(cv_fit, s = s))
  sum(co[rownames(co) != "(Intercept)", 1] != 0)
}


# --- Ridge (alpha = 0), L2 penalty ---
# Penalty is lambda * sum(beta^2): shrinks toward zero but never all the way, so
# nothing gets dropped. Handy for the collinear BILL_AMT block from Stage 3 —
# ridge spreads the shared signal across the correlated variables rather than
# picking one at random.
# type.measure = "auc" tunes lambda to maximise CV AUC, which is the metric we
# care about given the class imbalance (argued back in Stage 3).
cv_ridge <- glmnet::cv.glmnet(x_train, y_train, family = "binomial",
                              alpha = 0, foldid = foldid, type.measure = "auc")

# CV curve with lambda.min / lambda.1se marked.
png(file.path(fig_dir, "07_ridge_cv.png"), width = 800, height = 600, res = 120)
plot(cv_ridge)
title("Ridge (alpha=0): CV AUC vs log(lambda)", line = 2.5)
dev.off()

# lambda.min is the best mean CV AUC. lambda.1se is the biggest lambda still
# within one SE of it — a more regularised, simpler model that's statistically
# hard to tell apart from the best. Report both.
cat("\n############################################################\n")
cat("# 1. RIDGE (alpha = 0)                                     #\n")
cat("############################################################\n")
cat(sprintf("lambda.min = %.5f  (best CV AUC = %.4f)\n",
            cv_ridge$lambda.min, max(cv_ridge$cvm)))
cat(sprintf("lambda.1se = %.5f  (most parsimonious within 1 SE)\n",
            cv_ridge$lambda.1se))
cat("Ridge keeps ALL predictors (no exact zeros) — shrinks collinear ones together.\n")


# --- Lasso (alpha = 1), L1 penalty ---
# Penalty is lambda * sum(|beta|). The L1 corner geometry sends some coefficients
# to exactly zero, so lasso selects variables for us. In a correlated group it
# usually keeps one and zeros the others.
cv_lasso <- glmnet::cv.glmnet(x_train, y_train, family = "binomial",
                              alpha = 1, foldid = foldid, type.measure = "auc")

png(file.path(fig_dir, "08_lasso_cv.png"), width = 800, height = 600, res = 120)
plot(cv_lasso)
title("Lasso (alpha=1): CV AUC vs log(lambda)", line = 2.5)
dev.off()

cat("\n############################################################\n")
cat("# 2. LASSO (alpha = 1)                                     #\n")
cat("############################################################\n")
cat(sprintf("lambda.min = %.5f  (best CV AUC = %.4f)\n",
            cv_lasso$lambda.min, max(cv_lasso$cvm)))
cat(sprintf("lambda.1se = %.5f\n", cv_lasso$lambda.1se))

# What survives at lambda.1se? That's the sparsest defensible model, so it's the
# natural place to read off the "selected" default warning signs.
lasso_co_1se <- as.matrix(coef(cv_lasso, s = "lambda.1se"))
lasso_co_1se <- lasso_co_1se[rownames(lasso_co_1se) != "(Intercept)", 1]
lasso_kept    <- names(lasso_co_1se)[lasso_co_1se != 0]
lasso_dropped <- names(lasso_co_1se)[lasso_co_1se == 0]

cat(sprintf("\nAt lambda.1se: %d of %d coefficients driven to EXACTLY zero.\n",
            length(lasso_dropped), length(lasso_co_1se)))
cat("KEPT (selected default warning signs):\n  ",
    paste(lasso_kept, collapse = ", "), "\n")
cat("DROPPED (zeroed):\n  ",
    ifelse(length(lasso_dropped) == 0, "none", paste(lasso_dropped, collapse = ", ")), "\n")

# Sanity-check lasso's picks against the Stage-3 significant predictors. Rather
# than depend on the Stage-3 object, just restate that set here in the print-out
# and comment on how well the two agree.
cat("\nComparison to Stage-3 significant predictors (alpha=0.05):\n")
cat("  Stage-3 logistic flagged e.g. LIMIT_BAL, SEX2, EDUCATION2/4, MARRIAGE2,\n")
cat("  AGE, PAY_0, PAY_2, PAY_3, BILL_AMT1, PAY_AMT1, PAY_AMT2.\n")
cat("  Lasso's KEPT set should overlap strongly, and — crucially — PAY_0 (the\n")
cat("  dominant signal) is expected to SURVIVE. Agreement between an inferential\n")
cat("  test and an L1-selection is reassuring: the signal is robust.\n")
cat(sprintf("  PAY_0 kept by lasso? %s\n", "PAY_0" %in% lasso_kept))


# --- Elastic net, alpha searched over 0.1..0.9 ---
# penalty = lambda * [ alpha*|beta| + (1-alpha)/2*beta^2 ], so alpha=1 is lasso,
# alpha=0 is ridge, and anything in between is a compromise that can hold a
# correlated group together while still zeroing some coefficients. Grid-search
# alpha; each fit reuses the same foldid so the AUCs are comparable, then take
# the (alpha, lambda) with the best CV AUC.
alpha_grid <- seq(0.1, 0.9, 0.1)

enet_search <- lapply(alpha_grid, function(a) {
  fit <- glmnet::cv.glmnet(x_train, y_train, family = "binomial",
                           alpha = a, foldid = foldid, type.measure = "auc")
  best_auc <- max(fit$cvm)  # this alpha's CV AUC at its own lambda.min
  list(alpha = a, fit = fit, cv_auc = best_auc)
})

# Winner = highest best-CV-AUC.
enet_aucs   <- sapply(enet_search, function(z) z$cv_auc)
best_i      <- which.max(enet_aucs)
best_alpha  <- enet_search[[best_i]]$alpha
cv_enet     <- enet_search[[best_i]]$fit

cat("\n############################################################\n")
cat("# 3. ELASTIC NET (alpha tuned over 0.1..0.9)              #\n")
cat("############################################################\n")
enet_grid_tab <- data.frame(alpha = alpha_grid,
                            best_cv_auc = round(enet_aucs, 4))
print(enet_grid_tab, row.names = FALSE)
cat(sprintf("\nChosen alpha = %.1f  (best CV AUC = %.4f)\n",
            best_alpha, max(enet_aucs)))
cat(sprintf("  -> sits BETWEEN ridge (alpha=0) and lasso (alpha=1): %s.\n",
            ifelse(best_alpha < 0.5, "leans toward ridge (more grouping/shrinkage)",
                                     "leans toward lasso (more selection)")))
cat(sprintf("lambda.min = %.5f  lambda.1se = %.5f\n",
            cv_enet$lambda.min, cv_enet$lambda.1se))

# CV curve for the chosen alpha.
png(file.path(fig_dir, "09_enet_cv.png"), width = 800, height = 600, res = 120)
plot(cv_enet)
title(sprintf("Elastic net (alpha=%.1f): CV AUC vs log(lambda)", best_alpha),
      line = 2.5)
dev.off()


# --- Coefficients side by side at lambda.1se (ridge / lasso / enet) ---
# Line them up so the shrink-vs-select behaviour is visible at a glance. Using
# lambda.1se here, not min, because the sparser solution is the one worth reading
# off. Test scoring below uses lambda.min instead.
get_coef_vec <- function(cv_fit, s = "lambda.1se") {
  co <- as.matrix(coef(cv_fit, s = s))
  setNames(co[, 1], rownames(co))
}
ridge_c <- get_coef_vec(cv_ridge)
lasso_c <- get_coef_vec(cv_lasso)
enet_c  <- get_coef_vec(cv_enet)

all_names <- union(union(names(ridge_c), names(lasso_c)), names(enet_c))
coef_tab <- data.frame(
  predictor = all_names,
  ridge     = round(ridge_c[all_names], 4),
  lasso     = round(lasso_c[all_names], 4),
  enet      = round(enet_c[all_names],  4),
  row.names = NULL
)

# Tack on the plain logistic coefficients for reference, but only if train_df is
# lying around to refit cheaply. Optional.
unreg_path <- file.path(data_dir, "train_df.rds")
if (file.exists(unreg_path)) {
  tr <- readRDS(unreg_path)
  unreg_fit <- glm(default ~ ., data = tr, family = binomial)
  unreg_c   <- coef(unreg_fit)
  coef_tab$unreg_logistic <- round(unreg_c[match(coef_tab$predictor,
                                                 names(unreg_c))], 4)
}

cat("\n############################################################\n")
cat("# 4. COEFFICIENTS at lambda.1se (side by side)            #\n")
cat("############################################################\n")
print(coef_tab, row.names = FALSE)

# Zoom in on the collinear BILL_AMT block — it shows off the three behaviours.
bill_rows <- coef_tab[grepl("^BILL_AMT", coef_tab$predictor), ]
cat("\n--- BILL_AMT block (the collinear group from Stage-3 VIF) ---\n")
print(bill_rows, row.names = FALSE)
cat("Interpretation (ties back to Stage-3 VIF > 17 for BILL_AMT*):\n")
cat("  * RIDGE keeps ALL BILL_AMT* but SHRINKS them together (spreads the\n")
cat("    shared signal across the correlated group -> stable, none dominate).\n")
cat("  * LASSO tends to KEEP one representative BILL_AMT and ZERO the rest\n")
cat("    (selection resolves the collinearity by dropping redundancy).\n")
cat("  * ELASTIC NET compromises: partial shrinkage + partial selection.\n")

# Lasso path: which coefficients enter, and how they grow, as lambda drops.
png(file.path(fig_dir, "10_lasso_path.png"), width = 800, height = 600, res = 120)
plot(cv_lasso$glmnet.fit, xvar = "lambda", label = TRUE)
abline(v = log(cv_lasso$lambda.min), lty = 2, col = "grey40")
abline(v = log(cv_lasso$lambda.1se), lty = 3, col = "grey40")
title("Lasso coefficient paths (dashed=lambda.min, dotted=lambda.1se)", line = 2.5)
dev.off()


# --- Test evaluation at lambda.min, P(default) cut at 0.5 ---
# Score the held-out test set at each model's lambda.min (best CV performance),
# reusing the 0.5 threshold from the Stage-3 baseline so nothing about the
# comparison shifts. n_features_used = non-zero coefficients at lambda.min.
predict_prob <- function(cv_fit, s = "lambda.min") {
  as.numeric(predict(cv_fit, newx = x_test, s = s, type = "response"))
}

ridge_prob <- predict_prob(cv_ridge)
lasso_prob <- predict_prob(cv_lasso)
enet_prob  <- predict_prob(cv_enet)

ridge_metrics <- eval_metrics(ridge_prob, y_test, "ridge_l2",
                              n_nonzero(cv_ridge, "lambda.min"))
lasso_metrics <- eval_metrics(lasso_prob, y_test, "lasso_l1",
                              n_nonzero(cv_lasso, "lambda.min"))
enet_metrics  <- eval_metrics(enet_prob,  y_test, "enet",
                              n_nonzero(cv_enet,  "lambda.min"))

metrics_glmnet <- rbind(ridge_metrics, lasso_metrics, enet_metrics)

cat("\n############################################################\n")
cat("# 5. TEST-SET METRICS at lambda.min (threshold 0.5)      #\n")
cat("############################################################\n")
print(format(metrics_glmnet, digits = 4), row.names = FALSE)

# Line up against the Stage-3 baseline if it's on disk.
base_path <- file.path(data_dir, "metrics_baseline.rds")
if (file.exists(base_path)) {
  base_m <- readRDS(base_path)
  cat("\n--- vs unregularised baseline (Stage 3) ---\n")
  cat(sprintf("baseline logistic : AUC=%.4f recall=%.4f f1=%.4f  (features=%d)\n",
              base_m$auc, base_m$recall, base_m$f1, base_m$n_features_used))
  for (i in seq_len(nrow(metrics_glmnet))) {
    m <- metrics_glmnet[i, ]
    cat(sprintf("%-9s        : AUC=%.4f recall=%.4f f1=%.4f  (features=%d)  dAUC=%+.4f\n",
                m$method, m$auc, m$recall, m$f1, m$n_features_used,
                m$auc - base_m$auc))
  }
  cat("\nComment: regularisation typically yields AUC within a hair of the\n")
  cat("unregularised model here (the problem is low-dimensional relative to n,\n")
  cat("so overfitting is mild), BUT it buys STABILITY and, for lasso/enet, a\n")
  cat("SMALLER, more interpretable feature set — resolving the BILL_AMT\n")
  cat("collinearity that made the unregularised coefficients unreliable.\n")
}


# Save the three-row metrics table for Stage 7.
# Columns must line up with Stage 3 so Stage 7 can just rbind everything.
stopifnot(identical(
  names(metrics_glmnet),
  c("method","accuracy","auc","recall","specificity","precision","f1",
    "balanced_accuracy","n_features_used")))
saveRDS(metrics_glmnet, file.path(data_dir, "metrics_glmnet.rds"))


# Short console summary.
cat("\n\n=================== STAGE 4 GLMNET SUMMARY ===================\n")
cat(sprintf("Ridge  : lambda.min=%.5f lambda.1se=%.5f  (all %d features kept)\n",
            cv_ridge$lambda.min, cv_ridge$lambda.1se,
            n_nonzero(cv_ridge, "lambda.min")))
cat(sprintf("Lasso  : lambda.min=%.5f lambda.1se=%.5f  (features at min=%d, at 1se=%d)\n",
            cv_lasso$lambda.min, cv_lasso$lambda.1se,
            n_nonzero(cv_lasso, "lambda.min"), length(lasso_kept)))
cat(sprintf("E-net  : alpha=%.1f lambda.min=%.5f lambda.1se=%.5f  (features at min=%d)\n",
            best_alpha, cv_enet$lambda.min, cv_enet$lambda.1se,
            n_nonzero(cv_enet, "lambda.min")))
cat("\nTest metrics at lambda.min (threshold 0.5):\n")
for (i in seq_len(nrow(metrics_glmnet))) {
  m <- metrics_glmnet[i, ]
  cat(sprintf("  %-9s : accuracy=%.4f AUC=%.4f recall=%.4f precision=%.4f F1=%.4f bal.acc=%.4f (feat=%d)\n",
              m$method, m$accuracy, m$auc, m$recall, m$precision, m$f1,
              m$balanced_accuracy, m$n_features_used))
}
cat(sprintf("\nSaved: %s and CV/path plots 07-10 in %s/\n",
            file.path(data_dir, "metrics_glmnet.rds"), fig_dir))
cat("=============================================================\n")

# End of Stage 4 glmnet script.
