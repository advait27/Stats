# Project: Statistical Learning Practical Assignment (MIS41120, UCD)

## Goal
Predict credit-card default using OLS/logistic, SVM, and MLP — each in unregularised + ridge (L2) + lasso (L1) + elastic-net (L1+L2) variants — on the UCI "Default of Credit Card Clients" dataset. Compare methods on accuracy, interpretability, efficiency. Mirror ISLR Ch.3 Q15(b).

## Language & tools
R only (RStudio-runnable). Key packages: glmnet, e1071, sparseSVM, neuralnet/nnet, caret, ggplot2, corrplot.

## Dataset
- File: data/default_of_credit_card_clients.xls (header on row 2)
- 30,000 rows, 23 predictors + ID + target
- Target: `default payment next month` (binary; 22.1% default — IMBALANCED)
- Predictors: LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
  PAY_0/PAY_2..PAY_6 (repayment status, ordinal -2..8),
  BILL_AMT1..6, PAY_AMT1..6
- Quirks: EDUCATION has undocumented 0/5/6; MARRIAGE has 0; drop ID.

## Hard constraints (assignment rules)
- Every script self-contained, runs out-of-the-box, RELATIVE paths only.
- Comment thoroughly (code quality is graded).
- File naming: stl_Surname1_Surname2_Surname3_XXX.R
  (lowercase "stl"; XXX = meaningful script name). Replace surnames.
- Set.seed for reproducibility. No absolute paths.

## Working style
Go ONE stage at a time. Explain choices in comments. Don't jump ahead to modelling until I ask.