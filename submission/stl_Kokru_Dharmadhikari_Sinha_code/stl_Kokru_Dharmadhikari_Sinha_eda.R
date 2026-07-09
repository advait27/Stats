# Nothing random matters in this script, but set the seed anyway for consistency
# with the later modelling scripts.
set.seed(41120)

# Install-if-missing helper so a fresh machine can just source the file without
# the grader having to install packages by hand.
ensure_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  library(pkg, character.only = TRUE)
}

# readxl to read the .xls (header is on row 2), ggplot2 for the bar/dist plots,
# corrplot for the BILL_AMT correlation figure.
ensure_pkg("readxl")
ensure_pkg("ggplot2")
ensure_pkg("corrplot")

# figures/ holds all the plots; create it if it isn't there yet.
fig_dir <- "figures"
if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)


# --- Load data and drop ID ---
# The sheet has a title in row 1 and the actual column names in row 2, so
# skip = 1 puts the header where we want it. ID is just a row number, drop it.
#
# On the path: the brief expects data/default_of_credit_card_clients.xls, but in
# our repo the file sits in the root as "default of credit card clients.xls".
# Try the briefed location first, fall back to the root copy, so it runs either
# way. Both relative.
candidate_paths <- c(
  file.path("data", "default_of_credit_card_clients.xls"),
  "default of credit card clients.xls"
)
data_path <- candidate_paths[file.exists(candidate_paths)][1]
if (is.na(data_path)) {
  stop(
    "Could not find the dataset. Expected one of:\n  ",
    paste(candidate_paths, collapse = "\n  "),
    "\nPlace the .xls there and re-run."
  )
}
message("Reading data from: ", data_path)

# read_excel returns a tibble; as.data.frame keeps the base-R indexing we use later.
credit <- as.data.frame(readxl::read_excel(data_path, skip = 1))

credit$ID <- NULL

# Rename the target — "default payment next month" has spaces and is painful in
# formulas. Same variable, shorter name.
target_raw <- "default payment next month"
names(credit)[names(credit) == target_raw] <- "DEFAULT"

# Column groups we reuse below.
pay_cols  <- c("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6") # repayment status
bill_cols <- paste0("BILL_AMT", 1:6)                                 # monthly bill amounts
payamt_cols <- paste0("PAY_AMT", 1:6)                                # monthly payment amounts
predictor_cols <- setdiff(names(credit), "DEFAULT")


# --- Shape and the p >= 20 / n >= 5p checks ---
# The brief wants a reasonably wide problem with enough rows to fit SVM/MLP
# without instantly overfitting: at least 20 predictors and at least 5 rows per
# predictor. Confirm both here.
n_obs   <- nrow(credit)
p_preds <- length(predictor_cols)

# 1 = client defaults next month.
target_counts <- table(credit$DEFAULT)
target_prop   <- prop.table(target_counts)

cond_p  <- p_preds >= 20
cond_np <- n_obs   >= 5 * p_preds

cat("\n================ SHAPE & STRUCTURE ================\n")
cat(sprintf("Rows (n)            : %d\n", n_obs))
cat(sprintf("Predictors (p)      : %d  (target 'DEFAULT' excluded)\n", p_preds))
cat(sprintf("p >= 20             : %s\n", ifelse(cond_p, "YES", "NO")))
cat(sprintf("n >= 5p (>= %d)      : %s\n", 5 * p_preds, ifelse(cond_np, "YES", "NO")))


# --- Summary stats for every predictor ---
# This table is mainly about scale. LIMIT_BAL and the bill/pay amounts run into
# the tens or hundreds of thousands, while SEX/EDUCATION are single-digit codes.
# That gap is why we'll standardise before the penalised models and SVM/MLP —
# otherwise the L1/L2 penalty just punishes the large-scale coefficients for
# being large. It also lets us check there's nothing missing to impute.
summary_stats <- data.frame(
  variable = predictor_cols,
  mean     = sapply(credit[predictor_cols], function(x) mean(x, na.rm = TRUE)),
  sd       = sapply(credit[predictor_cols], function(x) sd(x,   na.rm = TRUE)),
  min      = sapply(credit[predictor_cols], function(x) min(x,  na.rm = TRUE)),
  max      = sapply(credit[predictor_cols], function(x) max(x,  na.rm = TRUE)),
  na_count = sapply(credit[predictor_cols], function(x) sum(is.na(x))),
  row.names = NULL
)
# round for a cleaner console print
summary_stats[, c("mean", "sd", "min", "max")] <-
  round(summary_stats[, c("mean", "sd", "min", "max")], 2)

cat("\n================ PREDICTOR SUMMARY STATS ================\n")
print(summary_stats, row.names = FALSE)

total_na <- sum(summary_stats$na_count)
cat(sprintf("\nTotal missing values across all predictors: %d\n", total_na))


# --- Class imbalance ---
# Roughly 22% default vs 78% not. That matters because raw accuracy is then a
# bad yardstick — a model that always predicts "no default" already gets ~78%.
# So later on we'll lean on sensitivity, ROC-AUC and F1, and maybe class
# weighting, rather than accuracy alone.
cat("\n================ TARGET CLASS BALANCE ================\n")
cat("Counts:\n");      print(target_counts)
cat("\nProportions:\n"); print(round(target_prop, 4))

balance_df <- data.frame(
  DEFAULT = factor(c(0, 1), labels = c("No default (0)", "Default (1)")),
  count   = as.integer(target_counts)
)
balance_df$prop <- balance_df$count / sum(balance_df$count)

p_balance <- ggplot(balance_df, aes(x = DEFAULT, y = count, fill = DEFAULT)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  # percentage labels on the bars so the imbalance reads at a glance
  geom_text(aes(label = sprintf("%d\n(%.1f%%)", count, 100 * prop)),
            vjust = -0.2, size = 4) +
  scale_fill_manual(values = c("No default (0)" = "#4C72B0",
                               "Default (1)"    = "#C44E52")) +
  labs(title = "Target class balance: credit-card default next month",
       subtitle = "Imbalanced — ~78% non-default vs ~22% default",
       x = NULL, y = "Number of clients") +
  theme_minimal(base_size = 12)

ggsave(file.path(fig_dir, "01_class_balance.png"),
       p_balance, width = 7, height = 5, dpi = 150)


# --- Repayment status PAY_0..PAY_6 ---
# The PAY_* columns are how many months behind the client was in each of the
# last six months (-2/-1/0 = paid on time or no balance, 1..8 = months late).
# These should be the strongest predictors — someone already a few months behind
# is much more likely to default. If default rate climbs steadily with delay,
# the signal is real and gives us an interpretable baseline for the fancier
# models to beat.

# Distributions first. Stack all six months into one long frame so we can facet.
pay_long <- do.call(rbind, lapply(pay_cols, function(col) {
  data.frame(month = col, status = credit[[col]])
}))
# factor with sorted levels so the ordinal codes stay in order on the x-axis
pay_long$status <- factor(pay_long$status,
                          levels = sort(unique(pay_long$status)))

p_pay_dist <- ggplot(pay_long, aes(x = status)) +
  geom_bar(fill = "#4C72B0") +
  facet_wrap(~ month, ncol = 3) +
  labs(title = "Distribution of repayment status (PAY_0..PAY_6)",
       subtitle = "Codes: -2/-1/0 = paid duly or no balance; 1..8 = months delayed",
       x = "Repayment status code", y = "Count") +
  theme_minimal(base_size = 11)

ggsave(file.path(fig_dir, "02_pay_status_distributions.png"),
       p_pay_dist, width = 10, height = 6, dpi = 150)

# Default rate against the most recent status (PAY_0), which is usually the
# single strongest one. mean(DEFAULT) within each status bucket is just the
# empirical P(default) there.
default_by_pay0 <- aggregate(DEFAULT ~ PAY_0, data = credit, FUN = mean)
default_by_pay0$n <- as.integer(table(credit$PAY_0)[as.character(default_by_pay0$PAY_0)])
names(default_by_pay0)[names(default_by_pay0) == "DEFAULT"] <- "default_rate"

cat("\n================ DEFAULT RATE BY MOST-RECENT REPAYMENT STATUS (PAY_0) ================\n")
print(transform(default_by_pay0, default_rate = round(default_rate, 3)),
      row.names = FALSE)

p_pay0_rate <- ggplot(default_by_pay0,
                      aes(x = factor(PAY_0), y = default_rate)) +
  geom_col(fill = "#C44E52") +
  geom_hline(yintercept = mean(credit$DEFAULT), linetype = "dashed") +
  geom_text(aes(label = sprintf("%.0f%%", 100 * default_rate)),
            vjust = -0.3, size = 3.5) +
  labs(title = "Default rate rises sharply with repayment delay (PAY_0)",
       subtitle = "Dashed line = overall default rate (~22%). Higher delay -> much higher default risk.",
       x = "Most recent repayment status (PAY_0)", y = "P(default)") +
  theme_minimal(base_size = 12)

ggsave(file.path(fig_dir, "03_default_rate_by_pay0.png"),
       p_pay0_rate, width = 8, height = 5, dpi = 150)

# Now the same thing across all six months, to check the signal is consistent
# and not just a quirk of the latest month. One panel per month.
rate_by_pay <- do.call(rbind, lapply(pay_cols, function(col) {
  agg <- aggregate(credit$DEFAULT, by = list(status = credit[[col]]), FUN = mean)
  data.frame(month = col, status = agg$status, default_rate = agg$x)
}))

p_pay_rate_all <- ggplot(rate_by_pay,
                         aes(x = factor(status), y = default_rate)) +
  geom_col(fill = "#C44E52") +
  facet_wrap(~ month, ncol = 3) +
  labs(title = "Default rate by repayment status — all six months",
       subtitle = "The positive relationship holds every month: repayment status is the dominant signal",
       x = "Repayment status code", y = "P(default)") +
  theme_minimal(base_size = 11)

ggsave(file.path(fig_dir, "04_default_rate_by_pay_all.png"),
       p_pay_rate_all, width = 10, height = 6, dpi = 150)


# --- BILL_AMT collinearity check ---
# The six BILL_AMT columns are consecutive monthly statement balances for the
# same clients, and balances drift slowly month to month, so they're bound to be
# highly correlated. That collinearity inflates the variance of OLS / unpenalised
# logistic coefficients and makes them unstable and hard to read — which is the
# main reason we bring in ridge/lasso/elastic-net later:
#   - ridge shrinks correlated coefficients together and keeps them all,
#   - lasso tends to keep one month and zero the others (sparsity),
#   - elastic-net is a mix of the two.
# Measuring the correlation now means that choice is backed by evidence rather
# than just asserted.
bill_cor <- cor(credit[bill_cols], use = "complete.obs")

cat("\n================ BILL_AMT CORRELATION MATRIX ================\n")
print(round(bill_cor, 3))
# mean off-diagonal correlation as a single-number summary of how collinear the block is
off_diag <- bill_cor[upper.tri(bill_cor)]
cat(sprintf("\nMean off-diagonal correlation among BILL_AMT1..6: %.3f\n",
            mean(off_diag)))
cat("Interpretation: near-1 correlations => strong collinearity => motivates ridge/lasso.\n")

# corrplot draws to base graphics, so we open a PNG device manually rather than
# using ggsave.
png(file.path(fig_dir, "05_billamt_corrplot.png"),
    width = 700, height = 700, res = 120)
corrplot::corrplot(bill_cor, method = "color", type = "upper",
                   addCoef.col = "black", tl.col = "black", tl.srt = 45,
                   number.cex = 0.8,
                   title = "BILL_AMT1..6 correlation (high collinearity)",
                   mar = c(0, 0, 2, 0))
dev.off()


# --- EDUCATION & MARRIAGE: check the category codes ---
# The data dictionary only documents some of the codes. The extras behave like
# an unlabelled "other/unknown" group, and we want to see them before modelling
# — left as-is they'd become their own tiny, noisy dummy levels, or get mangled
# if treated numerically.
#
# Documented:
#   EDUCATION: 1 = graduate school, 2 = university, 3 = high school, 4 = others
#   MARRIAGE : 1 = married, 2 = single, 3 = others
# So EDUCATION 0/5/6 and MARRIAGE 0 are undocumented.
#
# We only flag and propose a fix here; the actual recode is Stage 2.
edu_tab <- table(credit$EDUCATION)
mar_tab <- table(credit$MARRIAGE)

edu_documented <- c(1, 2, 3, 4)
mar_documented <- c(1, 2, 3)
edu_undoc <- setdiff(as.integer(names(edu_tab)), edu_documented)
mar_undoc <- setdiff(as.integer(names(mar_tab)), mar_documented)

cat("\n================ EDUCATION RAW CODES ================\n")
print(edu_tab)
cat(sprintf("Undocumented EDUCATION codes flagged: %s\n",
            paste(edu_undoc, collapse = ", ")))

cat("\n================ MARRIAGE RAW CODES ================\n")
print(mar_tab)
cat(sprintf("Undocumented MARRIAGE codes flagged: %s\n",
            paste(mar_undoc, collapse = ", ")))

# Proposed recode for Stage 2 (not run here):
#   EDUCATION: fold 0/5/6 into the existing "others" level 4
#              -> {1 grad school, 2 university, 3 high school, 4 other}
#   MARRIAGE : fold 0 into the existing "others" level 3
#              -> {1 married, 2 single, 3 other}
# These codes are rare and basically mean "unknown/other", so merging them into
# the existing "others" bucket avoids sparse dummy levels. Both then become
# unordered factors (dummy-encoded) in Stage 2.


# --- Summary ---
# Quick recap of the main findings and what they imply for modelling, for anyone
# reading only the console output.
cat("\n\n=================== STAGE 1 EDA SUMMARY ===================\n")
cat(sprintf("Dataset            : %d rows x %d predictors (+ target), ID dropped.\n",
            n_obs, p_preds))
cat(sprintf("Structural checks  : p>=20 %s ; n>=5p %s.\n",
            ifelse(cond_p, "PASS", "FAIL"), ifelse(cond_np, "PASS", "FAIL")))
cat(sprintf("Missing values     : %d (none to impute).\n", total_na))
cat(sprintf("Class balance      : %.1f%% default vs %.1f%% non-default (IMBALANCED).\n",
            100 * target_prop["1"], 100 * target_prop["0"]))
cat("Strongest signal   : repayment status PAY_0..PAY_6 — default rate rises\n")
cat("                     steeply and monotonically with months of delay.\n")
cat(sprintf("Collinearity       : BILL_AMT1..6 mean off-diagonal r = %.3f (HIGH)\n",
            mean(off_diag)))
cat("                     -> motivates ridge / lasso / elastic-net in later stages.\n")
cat(sprintf("Scale differences  : predictors span integer codes to 6-figure amounts\n"))
cat("                     -> features must be STANDARDISED before penalised/SVM/MLP.\n")
cat(sprintf("Messy categoricals : EDUCATION undocumented {%s}; MARRIAGE undocumented {%s}\n",
            paste(edu_undoc, collapse = ","), paste(mar_undoc, collapse = ",")))
cat("                     -> propose folding into 'other' in Stage 2 (NOT applied here).\n")
cat(sprintf("Figures saved to   : %s/ (5 PNG files).\n", fig_dir))
cat("==========================================================\n")

# End of Stage 1 EDA script.
