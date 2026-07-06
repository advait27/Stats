# stl_Kokru_Dharmadhikari_Sinha_mlp.R
#
# MIS41120 Statistical Learning — Stage 6: the MLP.
# Same four penalty variants as the other stages: unregularised, L2, L1, elastic net.
#
# We use h2o here. The point is to get all four variants out of one engine so the
# comparison stays fair — h2o.deeplearning takes both an l1 and an l2 argument on
# the same network, so the four models only differ in those two numbers and share
# an identical architecture otherwise. The usual R neural-net packages (nnet,
# neuralnet) only do L2 "weight decay" and have no L1 at all, which is why we can't
# get the L1 / elastic-net variants out of them.
#
# The catch is that h2o needs a Java JVM (h2o.init() spins one up), and that won't
# always be there on whoever's marking this. So there are two paths, chosen at
# runtime:
#   - if h2o is installed and Java actually works: all four variants, tuned with
#     h2o.grid on our shared CV folds and picked by CV AUC.
#   - otherwise: fall back to nnet, which can only give us the unregularised and L2
#     models. We say so clearly and save only those two rows — no made-up numbers
#     for the L1 / elastic-net cases.
#
# Inputs from data/: train_df, test_df, cv_foldid. Relative paths, seeded, run from
# the project root.


# Setup — seed, packages, output folders.
set.seed(1)

ensure_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  library(pkg, character.only = TRUE)
}

# pROC for AUC, nnet as the fallback engine (always load these two).
ensure_pkg("pROC")
ensure_pkg("nnet")

fig_dir  <- "figures"
data_dir <- "data"
if (!dir.exists(fig_dir))  dir.create(fig_dir,  recursive = TRUE)
if (!dir.exists(data_dir)) dir.create(data_dir, recursive = TRUE)


# Load the Stage-2 data and the shared fold assignment.
train_df <- readRDS(file.path(data_dir, "train_df.rds"))
test_df  <- readRDS(file.path(data_dir, "test_df.rds"))
foldid   <- readRDS(file.path(data_dir, "cv_foldid.rds"))  # same 10 folds used everywhere
stopifnot(nrow(train_df) == length(foldid),
          identical(levels(train_df$default), c("0", "1")))

p_total  <- ncol(train_df) - 1  # predictor count (drop the target column)
y_test   <- test_df$default


# Architecture / training constants, shared across the variants.
# The hidden-layer design is fixed on purpose so the penalty is the only thing
# that moves between the four models. Two layers (16, 8), epochs kept low so this
# doesn't take forever.
HIDDEN     <- c(16, 8)
EPOCHS     <- 50
NN_SIZE    <- 8          # single hidden layer for the nnet fallback
NN_MAXIT   <- 200


# Metric helper, same as the earlier stages. prob is the P(default) score used for AUC.
eval_metrics <- function(prob, truth, method_name, n_feat) {
  pred <- factor(ifelse(prob >= 0.5, "1", "0"), levels = c("0", "1"))
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
  roc_obj     <- pROC::roc(response = truth, predictor = as.numeric(prob),
                           levels = c("0","1"), direction = "<", quiet = TRUE)
  auc_val     <- as.numeric(pROC::auc(roc_obj))

  data.frame(method = method_name, accuracy = accuracy, auc = auc_val,
             recall = sensitivity, specificity = specificity,
             precision = precision, f1 = f1, balanced_accuracy = bal_acc,
             n_features_used = n_feat, stringsAsFactors = FALSE)
}


# Pick the engine: h2o if the package and a working JVM are both there, else nnet.
#
# We only trust h2o if the package is installed, Java actually runs, and h2o.init()
# comes up cleanly. The reason for the extra care: macOS ships a stub /usr/bin/java
# that exists but errors ("Unable to locate a Java Runtime") when no JRE is
# installed. A plain Sys.which("java") passes that stub, and then h2o.init() sits
# there for about a minute before giving up. So we run java -version first to
# fail fast, and still wrap h2o.init() in tryCatch as a backstop.
java_works <- function() {
  jh <- suppressWarnings(tryCatch(
    system2("java", "-version", stdout = TRUE, stderr = TRUE),
    error = function(e) NULL))
  # A real JRE prints its version and returns with no error status; the macOS
  # stub attaches a non-zero status attribute.
  !is.null(jh) && is.null(attr(jh, "status"))
}

use_h2o <- FALSE
if (requireNamespace("h2o", quietly = TRUE) && java_works()) {
  h2o_ok <- tryCatch({
    suppressMessages(library(h2o))
    h2o.init(nthreads = -1, max_mem_size = "2G")  # starts the JVM
    h2o.no_progress()
    TRUE
  }, error = function(e) {
    message("h2o.init() failed (", conditionMessage(e), ") -> using nnet fallback.")
    FALSE
  })
  use_h2o <- isTRUE(h2o_ok)
} else if (requireNamespace("h2o", quietly = TRUE)) {
  message("h2o installed but no working Java runtime found -> using nnet fallback.")
}

metrics_mlp <- NULL  # holds the rows we end up fitting


# h2o path: all four penalty variants.
if (use_h2o) {
  cat("\n############################################################\n")
  cat("# MLP via h2o.deeplearning (4 penalty variants)          #\n")
  cat("############################################################\n")

  # Move the data into H2OFrames. The fold column carries our shared assignment
  # so h2o's cross-validation splits on the same partition as the other stages.
  train_h2o <- as.h2o(cbind(train_df, fold = as.integer(foldid)), "train_h2o")
  test_h2o  <- as.h2o(test_df, "test_h2o")

  x_cols <- setdiff(colnames(train_df), "default")
  y_col  <- "default"

  # Grid-search the given l1/l2 ranges on the shared folds, keep the best CV-AUC
  # model, and return its test-set metrics row.
  # standardize = FALSE because Stage 2 already standardised the inputs on the
  # training set — letting h2o scale them again would be double-scaling.
  fit_h2o_variant <- function(method_name, l1_grid, l2_grid) {
    grid_id <- paste0("dl_grid_", method_name)
    hyper   <- list(l1 = l1_grid, l2 = l2_grid)
    g <- h2o.grid(
      algorithm  = "deeplearning",
      grid_id    = grid_id,
      x = x_cols, y = y_col,
      training_frame = train_h2o,
      fold_column    = "fold",     # our shared folds
      hidden = HIDDEN, epochs = EPOCHS,
      activation = "Rectifier",
      standardize = FALSE,         # already standardised in Stage 2
      balance_classes = TRUE,      # deal with the ~22% imbalance
      reproducible = TRUE, seed = 1,
      stopping_rounds = 0,
      hyper_params = hyper
    )
    # Sort by CV AUC and grab the top model.
    g_sorted <- h2o.getGrid(grid_id, sort_by = "auc", decreasing = TRUE)
    best     <- h2o.getModel(g_sorted@model_ids[[1]])
    # "p1" is P(default = 1) on the test set.
    pr   <- as.data.frame(h2o.predict(best, test_h2o))
    prob <- pr[["p1"]]
    # A net uses every input, so it isn't feature-sparse the way lasso is. L1 can
    # zero out some first-layer weights but the model still reads all p inputs, so
    # we record n_features_used = p_total; interpretability is low regardless.
    eval_metrics(prob, y_test, method_name, p_total)
  }

  # Unregularised: l1 = 0, l2 = 0 (just the one point).
  m_unreg <- fit_h2o_variant("mlp_unreg", l1_grid = c(0), l2_grid = c(0))
  # L2: l1 = 0, tune l2.
  m_l2    <- fit_h2o_variant("mlp_l2", l1_grid = c(0),
                             l2_grid = c(1e-5, 1e-4, 1e-3))
  # L1: tune l1, l2 = 0.
  m_l1    <- fit_h2o_variant("mlp_l1", l1_grid = c(1e-5, 1e-4, 1e-3),
                             l2_grid = c(0))
  # Elastic net: both l1 and l2 positive.
  m_enet  <- fit_h2o_variant("mlp_enet", l1_grid = c(1e-5, 1e-4),
                             l2_grid = c(1e-5, 1e-4))

  metrics_mlp <- rbind(m_unreg, m_l2, m_l1, m_enet)
  cat("\nNote: n_features_used = p (", p_total, ") for all MLPs — neural nets use\n")
  cat("every input; L1 only sparsifies first-layer weights, not whole inputs.\n")
  cat("MLP interpretability is LOW relative to the linear/penalised models.\n")

} else {
  # nnet fallback: unregularised + L2 (weight decay) only.
  # nnet has no L1 penalty, so it can't do the L1 or elastic-net variants. We say
  # so, fit the two it can, and save only those (nothing fabricated for L1/EN).
  cat("\n############################################################\n")
  cat("# h2o unavailable -> nnet FALLBACK (unreg + L2 only)      #\n")
  cat("############################################################\n")
  cat("nnet centres on L2 'weight decay' and has NO L1 penalty, so the L1 and\n")
  cat("elastic-net MLP variants CANNOT be produced here — they require h2o (or\n")
  cat("another engine exposing an L1 penalty). We fit the two variants nnet can:\n")
  cat("  * mlp_unreg : decay = 0 (no weight penalty)\n")
  cat("  * mlp_l2    : decay tuned by CV over the SHARED folds (L2 weight decay)\n")

  # Class weights: nnet has no class.weights arg, so we up-weight the minority
  # default class through observation `weights` to counter the imbalance.
  n_by_class  <- table(train_df$default)
  obs_wts     <- ifelse(train_df$default == "1",
                        sum(n_by_class) / (2 * n_by_class["1"]),
                        sum(n_by_class) / (2 * n_by_class["0"]))
  obs_wts     <- as.numeric(obs_wts)

  fit_nnet <- function(decay, dat = train_df, w = obs_wts) {
    set.seed(1)  # nnet inits weights randomly, so seed for reproducibility
    nnet::nnet(default ~ ., data = dat, weights = w, size = NN_SIZE,
               decay = decay, maxit = NN_MAXIT, MaxNWts = 10000,
               trace = FALSE)
  }

  # Tune the L2 decay by CV on the shared folds, scoring on AUC. Reusing foldid
  # keeps model selection on the same partition as the rest of the assignment;
  # for each candidate decay we average held-out AUC over the 10 folds.
  decay_grid <- c(1e-4, 1e-3, 1e-2, 1e-1)
  cv_auc_for_decay <- function(decay) {
    aucs <- sapply(sort(unique(foldid)), function(f) {
      tr_idx <- which(foldid != f); va_idx <- which(foldid == f)
      m <- fit_nnet(decay, dat = train_df[tr_idx, ], w = obs_wts[tr_idx])
      pv <- as.numeric(predict(m, train_df[va_idx, ], type = "raw"))
      as.numeric(pROC::auc(pROC::roc(train_df$default[va_idx], pv,
                                     levels = c("0","1"), direction = "<",
                                     quiet = TRUE)))
    })
    mean(aucs)
  }
  cat("\nTuning nnet L2 weight-decay over shared CV folds ...\n")
  decay_cv <- sapply(decay_grid, cv_auc_for_decay)
  best_decay <- decay_grid[which.max(decay_cv)]
  print(data.frame(decay = decay_grid, cv_auc = round(decay_cv, 4)),
        row.names = FALSE)
  cat(sprintf("Best L2 decay = %g (CV AUC = %.4f)\n",
              best_decay, max(decay_cv)))

  # Refit on the full training set, evaluate on test.
  # Unregularised (decay = 0).
  m0    <- fit_nnet(0)
  p0    <- as.numeric(predict(m0, test_df, type = "raw"))
  r_un  <- eval_metrics(p0, y_test, "mlp_unreg", p_total)
  # L2 with the best decay.
  m2    <- fit_nnet(best_decay)
  p2    <- as.numeric(predict(m2, test_df, type = "raw"))
  r_l2  <- eval_metrics(p2, y_test, "mlp_l2", p_total)

  metrics_mlp <- rbind(r_un, r_l2)
  cat("\nSaved ONLY mlp_unreg and mlp_l2 (the variants nnet can fit).\n")
  cat("mlp_l1 / mlp_enet are intentionally ABSENT — they require h2o.\n")
  cat(sprintf("n_features_used = p (%d) for all MLPs — nets use every input;\n", p_total))
  cat("MLP interpretability is LOW relative to the linear/penalised models.\n")
}


# Save metrics (same schema as the earlier stages) and the figure.
stopifnot(identical(
  names(metrics_mlp),
  c("method","accuracy","auc","recall","specificity","precision","f1",
    "balanced_accuracy","n_features_used")))
saveRDS(metrics_mlp, file.path(data_dir, "metrics_mlp.rds"))

cat("\n############################################################\n")
cat("# MLP TEST-SET METRICS (threshold 0.5)                   #\n")
cat("############################################################\n")
print(format(metrics_mlp, digits = 4), row.names = FALSE)

# Figure: test AUC across the MLP variants.
auc_order <- metrics_mlp[order(metrics_mlp$auc), ]
png(file.path(fig_dir, "12_mlp_auc.png"), width = 820, height = 520, res = 120)
op <- par(mar = c(5, 9, 4, 2))
bp <- barplot(auc_order$auc, horiz = TRUE, names.arg = auc_order$method,
              las = 1, xlim = c(0.5, 0.8), col = "#55A868",
              xlab = "Test ROC-AUC",
              main = sprintf("MLP variants — test AUC (%s engine)",
                             ifelse(use_h2o, "h2o", "nnet fallback")))
text(auc_order$auc, bp, labels = sprintf("%.3f", auc_order$auc),
     pos = 4, xpd = TRUE, cex = 0.9)
abline(v = 0.5, lty = 2, col = "grey50")
par(op)
dev.off()


# Shut h2o down (releases the JVM), then print the summary.
if (use_h2o) {
  h2o.shutdown(prompt = FALSE)
}

cat("\n\n=================== STAGE 6 MLP SUMMARY ===================\n")
cat(sprintf("Engine used         : %s\n",
            ifelse(use_h2o, "h2o.deeplearning (all 4 variants)",
                            "nnet fallback (mlp_unreg + mlp_l2 only)")))
cat(sprintf("Architecture        : %s\n",
            ifelse(use_h2o, paste0("hidden = c(", paste(HIDDEN, collapse=","), ")"),
                            paste0("nnet size = ", NN_SIZE))))
cat("Test metrics (threshold 0.5):\n")
for (i in seq_len(nrow(metrics_mlp))) {
  m <- metrics_mlp[i, ]
  cat(sprintf("  %-10s : acc=%.4f AUC=%.4f recall=%.4f prec=%.4f F1=%.4f bal=%.4f feat=%d\n",
              m$method, m$accuracy, m$auc, m$recall, m$precision, m$f1,
              m$balanced_accuracy, m$n_features_used))
}
if (!use_h2o) {
  cat("\nNOTE: mlp_l1 / mlp_enet not fitted (nnet has no L1 penalty; needs h2o).\n")
}
cat(sprintf("\nSaved: %s and %s\n",
            file.path(data_dir, "metrics_mlp.rds"),
            file.path(fig_dir,  "12_mlp_auc.png")))
cat("==========================================================\n")

# end of Stage 6.
