#!/usr/bin/env Rscript
# scripts/varima_aic.R
#
# Forecast-skill cross-validation: univariate AR (per state) vs
# bivariate VAR (per pair), for all 45 pairs among the ten largest
# US states. Trained on pre-COVID weekly P&I mortality (2010-2019)
# and tested on post-COVID (2023-2025), one-step-ahead with actual
# lags (coefficients frozen from training).
#
# Deaths are normalized to deaths per 100k population, with the
# state-level population linearly interpolated between July-1
# estimates in data/pni_mortality/populations.csv.
#
# Both sides share the same regressor structure:
#   intercept + linear trend + 51 weekly seasonal dummies + AR lags.
# The VAR for a pair adds cross-lag terms. Order is picked by AIC
# on the training data (same lm-based AIC on both sides).
#
# Run from project root:  Rscript scripts/varima_aic.R
# Output: outputs/varima_cv.csv

# ---- auto-install dependencies --------------------------------------
required <- c("tidyr", "dplyr")
missing  <- setdiff(required, rownames(installed.packages()))
if (length(missing) > 0) {
  cat("Installing missing packages:", paste(missing, collapse = ", "), "\n")
  install.packages(missing, repos = "https://cloud.r-project.org")
}
suppressPackageStartupMessages({
  for (pkg in required) library(pkg, character.only = TRUE)
})

# ---- ten largest US states (by 2020 population) ---------------------
states <- c(
  "California", "Texas", "Florida", "New York", "Pennsylvania",
  "Illinois",   "Ohio",  "Georgia", "North Carolina", "Michigan"
)

# ---- load full 2010-2025 range, keep all weeks ----------------------
w <- read.csv("data/pni_mortality/deaths.csv") |>
  mutate(date = as.Date(date)) |>
  filter(as.integer(format(date, "%Y")) >= 2010 &
         as.integer(format(date, "%Y")) <= 2025) |>
  pivot_wider(names_from = state, values_from = deaths) |>
  arrange(date)

missing_states <- setdiff(states, names(w))
if (length(missing_states) > 0)
  stop("States not found in deaths.csv: ", paste(missing_states, collapse = ", "))

W <- w[, c("date", states)]
W <- W[complete.cases(W[, states]), ]

# ---- normalize to deaths per 100k ----------------------------------
pop_df <- read.csv("data/pni_mortality/populations.csv")
pop_df$date <- as.Date(pop_df$date)

interp_pop <- function(state_name, dates) {
  sub <- pop_df[pop_df$state == state_name, c("date", "population")]
  sub <- sub[order(sub$date), ]
  approx(x = as.numeric(sub$date),
         y = sub$population,
         xout = as.numeric(dates),
         rule = 2)$y
}
for (s in states) {
  W[[s]] <- W[[s]] / interp_pop(s, W$date) * 1e5
}

# ---- train / test masks --------------------------------------------
yr <- as.integer(format(W$date, "%Y"))
train_idx <- which(yr >= 2010 & yr <= 2019)
test_idx  <- which(yr >= 2023 & yr <= 2025)

Tn <- nrow(W)
cat(sprintf("Data: %d weekly observations (%s to %s), 10 states.\n",
            Tn, format(min(W$date)), format(max(W$date))))
cat(sprintf("Train: %d weeks (2010-2019); Test: %d weeks (2023-2025).\n\n",
            length(train_idx), length(test_idx)))

# ---- deterministic xreg: trend + 51 seasonal dummies ---------------
week_of_year <- ((seq_len(Tn) - 1) %% 52) + 1
seasonal_dummies <- sapply(2:52, function(w) as.numeric(week_of_year == w))
colnames(seasonal_dummies) <- paste0("sd", 2:52)
X_det <- cbind(trend = seq_len(Tn), seasonal_dummies)

# ---- lag matrix builder --------------------------------------------
make_lags <- function(y, p, prefix = "lag") {
  if (p <= 0) return(NULL)
  T <- length(y)
  L <- matrix(NA, T, p)
  for (k in 1:p) L[(k + 1):T, k] <- y[1:(T - k)]
  colnames(L) <- paste0(prefix, 1:p)
  L
}

# ---- univariate AR: fit on train, predict on test via lm -----------
fit_uni <- function(y, p) {
  L <- make_lags(y, p)
  df <- data.frame(y = y, (if (is.null(L)) X_det else cbind(L, X_det)))
  train_rows <- intersect(train_idx, which(complete.cases(df)))
  test_rows  <- intersect(test_idx,  which(complete.cases(df)))
  fit <- lm(y ~ ., data = df[train_rows, ])
  preds <- predict(fit, newdata = df[test_rows, ])
  list(fit = fit, preds = preds, actual = y[test_rows], p = p)
}

# ---- bivariate VAR: two equations fitted by lm ---------------------
fit_var <- function(y1, y2, p) {
  L1 <- make_lags(y1, p, prefix = "y1.l")
  L2 <- make_lags(y2, p, prefix = "y2.l")
  L  <- if (p == 0) NULL else cbind(L1, L2)
  X  <- if (is.null(L)) X_det else cbind(L, X_det)
  df1 <- data.frame(y = y1, X)
  df2 <- data.frame(y = y2, X)
  train_rows <- intersect(train_idx, which(complete.cases(df1)))
  test_rows  <- intersect(test_idx,  which(complete.cases(df1)))
  fit1 <- lm(y ~ ., data = df1[train_rows, ])
  fit2 <- lm(y ~ ., data = df2[train_rows, ])
  preds1 <- predict(fit1, newdata = df1[test_rows, ])
  preds2 <- predict(fit2, newdata = df2[test_rows, ])
  list(fit1 = fit1, fit2 = fit2,
       preds1 = preds1, preds2 = preds2,
       actual1 = y1[test_rows], actual2 = y2[test_rows],
       p = p)
}

# ---- AIC-based order selection (on training only) ------------------
pick_p_uni <- function(y, max_p = 8) {
  aics <- sapply(0:max_p, function(p)
    tryCatch(AIC(fit_uni(y, p)$fit), error = function(e) Inf))
  which.min(aics) - 1
}
pick_p_var <- function(y1, y2, max_p = 8) {
  aics <- sapply(0:max_p, function(p)
    tryCatch({ f <- fit_var(y1, y2, p); AIC(f$fit1) + AIC(f$fit2) },
             error = function(e) Inf))
  which.min(aics) - 1
}

rmse <- function(r) sqrt(mean(r^2, na.rm = TRUE))

# ---- cache univariate fits per state -------------------------------
cat("Picking p and fitting univariate AR per state (cached, 10 fits):\n")
uni_cache <- list()
for (s in states) {
  y  <- W[[s]]
  p  <- pick_p_uni(y)
  fu <- fit_uni(y, p)
  uni_cache[[s]] <- list(p = p, preds = fu$preds, actual = fu$actual)
  cat(sprintf("  %-16s  p = %d   train-AIC = %8.1f   test-RMSE = %.4f\n",
              s, p, AIC(fu$fit), rmse(fu$actual - fu$preds)))
}
cat("\n")

# ---- run all 45 pairs ----------------------------------------------
pairs <- combn(states, 2, simplify = FALSE)
cat(sprintf("Fitting %d VARs and computing test-period RMSE...\n", length(pairs)))

rows <- vector("list", length(pairs))
for (i in seq_along(pairs)) {
  s1 <- pairs[[i]][1]; s2 <- pairs[[i]][2]
  y1 <- W[[s1]]; y2 <- W[[s2]]
  pv <- pick_p_var(y1, y2)
  fv <- fit_var(y1, y2, pv)

  # Univariate test residuals (both states)
  u1 <- uni_cache[[s1]]; u2 <- uni_cache[[s2]]
  r_uni1 <- u1$actual - u1$preds
  r_uni2 <- u2$actual - u2$preds
  # VAR test residuals
  r_var1 <- fv$actual1 - fv$preds1
  r_var2 <- fv$actual2 - fv$preds2

  rmse_uni <- rmse(c(r_uni1, r_uni2))
  rmse_var <- rmse(c(r_var1, r_var2))

  rows[[i]] <- data.frame(
    pair      = paste(s1, s2, sep = " x "),
    p_uni     = sprintf("(%d, %d)", u1$p, u2$p),
    p_var     = pv,
    rmse_uni  = round(rmse_uni, 4),
    rmse_var  = round(rmse_var, 4),
    delta     = round(rmse_var - rmse_uni, 4),
    pct_change= round(100 * (rmse_var - rmse_uni) / rmse_uni, 2),
    prefers   = ifelse(rmse_var < rmse_uni, "multivariate", "independent")
  )
}
res <- do.call(rbind, rows)

# ---- print + save --------------------------------------------------
cat("\nOne-step-ahead test-period RMSE: univariate AR vs bivariate VAR.\n")
cat("Trained on 2010-2019, tested on 2023-2025. Deaths per 100k.\n")
cat("Negative delta / pct_change => multivariate forecasts better.\n\n")

res_sorted <- res[order(res$pct_change), ]
print(res_sorted, row.names = FALSE)

cat(sprintf("\nSummary: %d of %d pairs prefer multivariate (lower RMSE)\n",
            sum(res$prefers == "multivariate"), nrow(res)))
cat(sprintf("         pct_change: min = %+.2f%%, median = %+.2f%%, max = %+.2f%%\n",
            min(res$pct_change), median(res$pct_change), max(res$pct_change)))

dir.create("outputs", showWarnings = FALSE)
write.csv(res, "outputs/varima_cv.csv", row.names = FALSE)
cat(sprintf("\nSaved: outputs/varima_cv.csv\n"))
