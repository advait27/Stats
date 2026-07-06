# stl_Kokru_Dharmadhikari_Sinha_prep.R
#
# MIS41120 Statistical Learning - practical assignment
# Stage 2: preprocessing. Takes the raw UCI credit-card .xls and produces the
# train/test objects that the modelling scripts (stages 3-6) read back in.
#
# What happens here: apply the recodes we settled on in the EDA, fix variable
# types, do a stratified split, standardise on the training data only, and save
# two versions of the data (a data frame and a numeric model matrix) plus one
# shared set of CV folds. Everything is written to data/*.rds so the later
# scripts just readRDS() and go.
#
# Reproducible (seed set below) and uses relative paths only. In RStudio, set
# the working directory to the source file location before sourcing.


# Setup ------------------------------------------------------------------

# Fix the RNG so the split and the folds come out the same every run. This
# matters for the model comparison later - all models need to see the identical
# train/test split and the identical folds.
set.seed(1)

# Small helper so this still works on a clean machine (installs if missing).
ensure_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  library(pkg, character.only = TRUE)
}

# readxl for the .xls, caret for the stratified split and the CV folds.
ensure_pkg("readxl")
ensure_pkg("caret")

data_out_dir <- "data"
if (!dir.exists(data_out_dir)) dir.create(data_out_dir, recursive = TRUE)


# Load ------------------------------------------------------------------

# The first row of the sheet is a title banner, so skip = 1 puts the real
# header on row 2. ID is just a row key, drop it.
#
# The brief expects the file under data/, but in this repo it currently lives
# in the project root with spaces in the name. Try the briefed path first and
# fall back to the root copy so the script runs either way.
candidate_paths <- c(
  file.path("data", "default_of_credit_card_clients.xls"),
  "default of credit card clients.xls"
)
data_path <- candidate_paths[file.exists(candidate_paths)][1]
if (is.na(data_path)) {
  stop("Could not find the dataset. Expected one of:\n  ",
       paste(candidate_paths, collapse = "\n  "))
}
message("Reading data from: ", data_path)

credit <- as.data.frame(readxl::read_excel(data_path, skip = 1))
credit$ID <- NULL  # drop the row identifier


# Recode the category quirks the EDA flagged ----------------------------

# The dictionary only documents some of the codes. The EDA turned up a few
# extra codes that really just mean "other/unknown", so fold them into the
# existing "others" level rather than let them become their own tiny dummies.

# EDUCATION: 1=grad school, 2=university, 3=high school, 4=others.
# 0/5/6 are undocumented and rare -> put them in 4.
credit$EDUCATION[credit$EDUCATION %in% c(0, 5, 6)] <- 4

# MARRIAGE: 1=married, 2=single, 3=others. 0 is undocumented -> put it in 3.
credit$MARRIAGE[credit$MARRIAGE == 0] <- 3


# Variable types --------------------------------------------------------

# Getting these right now means model.matrix() one-hot encoding and the models
# themselves treat each column the way we intend.

# SEX / EDUCATION / MARRIAGE are nominal codes - no meaningful ordering - so
# they're factors. model.matrix() will one-hot them for glmnet/sparseSVM, and
# the formula-based models (e1071/neuralnet/caret) handle factors themselves.
credit$SEX       <- factor(credit$SEX)
credit$EDUCATION <- factor(credit$EDUCATION)
credit$MARRIAGE  <- factor(credit$MARRIAGE)

# PAY_0..PAY_6 (repayment status) I'm keeping numeric. The codes are ordinal
# (-2/-1/0 = paid duly / no consumption, 1..8 = months of delay) and the EDA
# showed default rate rising roughly monotonically with the code, so a single
# numeric column keeps that ordering and gives the linear models one slope per
# month. It also keeps the design matrix small (one column each vs ~10 dummies),
# which helps the penalised models and SVM/MLP. An unordered factor would throw
# the ordering away; an ordered factor would add polynomial contrasts I don't
# want here.
pay_cols <- c("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")
for (col in pay_cols) credit[[col]] <- as.numeric(credit[[col]])

# The genuine continuous predictors (money amounts and age). Listed explicitly
# so the standardisation later hits exactly these columns.
cont_cols <- c("LIMIT_BAL", "AGE", paste0("BILL_AMT", 1:6), paste0("PAY_AMT", 1:6))
for (col in cont_cols) credit[[col]] <- as.numeric(credit[[col]])

# Target: rename to "default" and make it a two-level factor. It's already
# binary in the raw data so there's nothing to convert. Fixing the levels to
# c("0","1") keeps "1" (default) as the positive class everywhere downstream,
# which keeps sensitivity / ROC / confusion matrices consistent across scripts.
target_raw <- "default payment next month"
names(credit)[names(credit) == target_raw] <- "default"
credit$default <- factor(credit$default, levels = c(0, 1))


# Stratified 70/30 split -------------------------------------------------

# ~22% default, so a plain random split could land too few defaults on one
# side. createDataPartition stratifies on the outcome, so both train and test
# keep roughly the 22%. Checked in the summary at the end.
train_idx <- caret::createDataPartition(credit$default, p = 0.70, list = FALSE)
train_df  <- credit[train_idx, ]
test_df   <- credit[-train_idx, ]


# Standardise continuous predictors (train stats only) ------------------

# Why standardise: the penalised models shrink every coefficient with the same
# penalty, so without a common scale the large-magnitude variables (BILL_AMT in
# the 100,000s) get unfairly shrunk relative to small ones. SVM and MLP care
# about feature scale too. Factors stay as-is - the one-hot dummies are already
# 0/1 and scaling them would just distort them.
#
# Why train only: we learn the mean and sd from the training rows and reuse
# those same numbers on the test rows. Computing them over the full data would
# leak test-set information into preprocessing and flatter the test estimate.
train_means <- sapply(train_df[cont_cols], mean)
train_sds   <- sapply(train_df[cont_cols], sd)

# Apply the train-derived (mean, sd) to both sets - scale() with explicit
# center/scale is just (x - mean) / sd column-wise.
train_df[cont_cols] <- scale(train_df[cont_cols],
                             center = train_means, scale = train_sds)
test_df[cont_cols]  <- scale(test_df[cont_cols],
                             center = train_means, scale = train_sds)
# Assigning back into the data frame drops the matrix attributes scale() adds,
# leaving plain numeric columns.


# Two output representations ---------------------------------------------

# Different packages want different input shapes, so save both.

# Data-frame form (train_df/test_df) is for the formula + data-frame models
# that handle factors themselves: e1071::svm, neuralnet, caret. Already built
# above, just saved below.

# Model-matrix form is for glmnet / sparseSVM, which need a numeric design
# matrix plus a separate response vector. model.matrix one-hot encodes the
# factors; the "- 1" drops the intercept column because glmnet fits its own
# intercept (a column of 1s in x would be redundant).
#
# train and test matrices are built separately, but the factor levels were set
# on the full data above so every level is present in both - that's what keeps
# the columns identical between x_train and x_test.
mm_formula <- default ~ . - 1

x_train <- model.matrix(mm_formula, data = train_df)
x_test  <- model.matrix(mm_formula, data = test_df)

# Response vectors - keep the factor with levels c("0","1") so "1" is still the
# positive class (glmnet's binomial family is fine with a factor).
y_train <- train_df$default
y_test  <- test_df$default

# Sanity check: the two matrices must share columns or a model fit on x_train
# can't score x_test.
if (!identical(colnames(x_train), colnames(x_test))) {
  stop("x_train and x_test have mismatched columns — check factor levels.")
}

saveRDS(train_df, file.path(data_out_dir, "train_df.rds"))
saveRDS(test_df,  file.path(data_out_dir, "test_df.rds"))
saveRDS(x_train,  file.path(data_out_dir, "x_train.rds"))
saveRDS(y_train,  file.path(data_out_dir, "y_train.rds"))
saveRDS(x_test,   file.path(data_out_dir, "x_test.rds"))
saveRDS(y_test,   file.path(data_out_dir, "y_test.rds"))

# Also save the standardisation stats. Anything that later needs to transform
# fresh data should reuse these train values, not recompute.
saveRDS(list(center = train_means, scale = train_sds),
        file.path(data_out_dir, "standardisation_params.rds"))


# Shared 10-fold CV on the training target ------------------------------

# One set of folds shared across every model. If each model drew its own random
# folds, some of the performance gap between models would just be different
# partitions rather than the models themselves. Fixing one stratified partition
# and reusing it keeps the comparison fair.
#
# createFolds on the factor target keeps the ~22% default rate in each fold. We
# save cv_folds (held-out row indices per fold, usable directly by cv.glmnet's
# foldid or a manual CV loop) and cv_trainControl (a caret control pinned to the
# same folds).
n_folds <- 10
cv_folds <- caret::createFolds(y_train, k = n_folds,
                               list = TRUE, returnTrain = FALSE)

# glmnet wants a single foldid vector (fold number per training row); derive it
# from cv_folds so cv.glmnet uses the exact same partition.
foldid <- integer(length(y_train))
for (f in seq_along(cv_folds)) foldid[cv_folds[[f]]] <- f

# caret reuses the same folds via `index` (per-fold training rows), which is
# just the complement of the held-out rows.
#
# Note on classProbs: caret's ROC summary (twoClassSummary) needs probabilities
# and syntactically valid factor level names, but "0"/"1" aren't valid R names
# and would make caret::train() error. We deliberately keep the numeric "0"/"1"
# levels on the saved y (so "1" = positive is unambiguous across all packages),
# so classProbs is left off this shared control - its only job is to pin the
# folds. A caret model that wants ROC can relabel its own copy of y to "no"/
# "yes" and set classProbs locally.
cv_index <- lapply(cv_folds, function(held_out) setdiff(seq_along(y_train), held_out))
cv_trainControl <- caret::trainControl(
  method          = "cv",
  number          = n_folds,
  index           = cv_index,        # per-fold TRAINING rows (same partition)
  indexOut        = cv_folds,        # per-fold HELD-OUT rows (same partition)
  savePredictions = "final"
)

saveRDS(cv_folds,        file.path(data_out_dir, "cv_folds.rds"))
saveRDS(foldid,         file.path(data_out_dir, "cv_foldid.rds"))
saveRDS(cv_trainControl, file.path(data_out_dir, "cv_trainControl.rds"))


# Summary ---------------------------------------------------------------

# Sizes, class balance (did the split hold ~22% both sides?), and an NA check.
train_bal <- prop.table(table(train_df$default))
test_bal  <- prop.table(table(test_df$default))
full_bal  <- prop.table(table(credit$default))

na_train_df <- sum(is.na(train_df))
na_test_df  <- sum(is.na(test_df))
na_xtrain   <- sum(is.na(x_train))
na_xtest    <- sum(is.na(x_test))

cat("\n=================== STAGE 2 PREPROCESSING SUMMARY ===================\n")
cat(sprintf("Raw (post-recode)  : %d rows x %d cols (incl. target)\n",
            nrow(credit), ncol(credit)))
cat(sprintf("train_df           : %d rows x %d cols\n", nrow(train_df), ncol(train_df)))
cat(sprintf("test_df            : %d rows x %d cols\n", nrow(test_df),  ncol(test_df)))
cat(sprintf("x_train (matrix)   : %d rows x %d cols (one-hot, no intercept)\n",
            nrow(x_train), ncol(x_train)))
cat(sprintf("x_test  (matrix)   : %d rows x %d cols\n", nrow(x_test), ncol(x_test)))

cat("\nClass balance (proportion default = level '1'):\n")
cat(sprintf("  full  : 0=%.3f  1=%.3f\n", full_bal["0"],  full_bal["1"]))
cat(sprintf("  train : 0=%.3f  1=%.3f\n", train_bal["0"], train_bal["1"]))
cat(sprintf("  test  : 0=%.3f  1=%.3f  (stratification %s)\n",
            test_bal["0"], test_bal["1"],
            ifelse(abs(train_bal["1"] - test_bal["1"]) < 0.01, "OK", "CHECK")))

cat(sprintf("\nCV: %d stratified folds on training target (shared by all models).\n",
            n_folds))
cat("Fold sizes (held-out rows):\n")
print(sapply(cv_folds, length))

cat(sprintf("\nNA audit  : train_df=%d  test_df=%d  x_train=%d  x_test=%d  (all should be 0)\n",
            na_train_df, na_test_df, na_xtrain, na_xtest))

cat(sprintf("\nSaved to %s/: train_df, test_df, x_train, y_train, x_test, y_test,\n",
            data_out_dir))
cat("            standardisation_params, cv_folds, cv_foldid, cv_trainControl (.rds)\n")
cat("These are the inputs Stages 3-6 will readRDS(). \n")
cat("====================================================================\n")

# End of stage 2.
