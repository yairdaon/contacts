#!/usr/bin/env Rscript                                                                      # py: #!/usr/bin/env python3

# ar.R -- SARIMA forecast benchmark on 47 US states' weekly per-100k P&I mortality.
# Fit four SARIMAs and compare test-period RMSE for each ordered (target, neighbour) pair:
#   (null) no exogenous covariate            }  per target  (47 fits)
#   (nav)  national average y-bar_t-1 only   }
#   (biv)  neighbour state's lag-1 y_B,t-1                     }  per pair  (2162 fits)
#   (both) y_B,t-1 + y-bar_t-1 jointly                         }
# Headline comparison: (both) vs (nav) -- does adding neighbour state B reduce RMSE on top
# of a SARIMA that already includes the national average?
# Train: 2010-2019 (pre-COVID).  Test: 2023-2025 (post-COVID).  COVID years skipped.
# Each R line annotated with a Python-equivalent comment for readers coming from Python.

suppressPackageStartupMessages({                                                             # py: with warnings.catch_warnings(): warnings.simplefilter("ignore")
  library(forecast)                                                                          # py: import pmdarima.auto_arima as auto_arima (closest Python equivalent)
  library(dplyr)                                                                             # py: import pandas as pd
  library(tidyr)                                                                             # py: (pandas again; tidyr = pandas reshape/pivot)
  library(lubridate)                                                                         # py: from datetime import date; pd.to_datetime
  library(parallel)                                                                          # py: from multiprocessing import Pool
})
N_CORES <- max(1, detectCores() - 2)                                                         # py: n_cores = max(1, os.cpu_count() - 2)
SEASONAL <- list(order = c(1, 0, 0), period = 52)                                            # py: seasonal_order = (1, 0, 0, 52)    # SARIMA seasonal AR(1) at lag 52

# ---- data: load, filter to pre+post-COVID, normalize per-100k ------------------------------

pop <- read.csv("data/pni_mortality/populations.csv") |> mutate(date = as.Date(date))        # py: pop = pd.read_csv(...); pop['date'] = pd.to_datetime(pop['date'])

ipop <- function(s, d) approx(as.numeric(pop$date[pop$state == s]),                          # py: def ipop(s, d): return np.interp(d, pop.loc[pop.state==s, 'date'], pop.loc[pop.state==s, 'population'])
                              pop$population[pop$state == s], as.numeric(d), rule = 2)$y    #     (linear interp between July-1 Census estimates)

excluded <- c("Alaska", "Hawaii", "Arizona")                                                 # py: excluded = ["Alaska", "Hawaii", "Arizona"]

w0 <- read.csv("data/pni_mortality/deaths.csv") |>                                           # py: w0 = pd.read_csv("data/pni_mortality/deaths.csv")
  mutate(date = as.Date(date)) |>                                                            # py: w0['date'] = pd.to_datetime(w0['date'])
  filter(year(date) %in% c(2010:2019, 2023:2025),                                            # py: w0 = w0[(w0.date.dt.year.isin([*range(2010,2020),*range(2023,2026)]))
         !(state %in% excluded))                                                             # py:           & (~w0.state.isin(excluded))]
states <- unique(w0$state)                                                                   # py: states = w0['state'].unique()

w <- w0 |>                                                                                   # py: w = w0.copy()
  group_by(state) |>                                                                         # py: w = w.groupby("state", group_keys=False).apply(
  mutate(deaths = deaths / ipop(state[1], date) * 1e5) |> ungroup() |>                       # py:     lambda g: g.assign(deaths = g.deaths / ipop(g.name, g.date) * 1e5))
  pivot_wider(names_from = state, values_from = deaths) |> arrange(date)                     # py: w = w.pivot(index="date", columns="state", values="deaths").sort_index()

tr <- which(year(w$date) <= 2019)                                                            # py: tr = np.where(w.index.year <= 2019)[0]
te <- which(year(w$date) >= 2023)                                                            # py: te = np.where(w.index.year >= 2023)[0]

fmt <- function(o) sprintf("(%d,%d,%d)(%d,%d,%d)[%d]",                                       # py: def fmt(o): return f"({o[0]},{o[1]},{o[2]})({o[3]},{o[4]},{o[5]})[{o[6]}]"
                           o[1], o[2], o[3], o[4], o[5], o[6], o[7])

# ---- national average covariate (same for every pair) ------------------------------------

NAVG <- rowMeans(as.matrix(w[, states]), na.rm = TRUE)                                       # py: NAVG = w[states].mean(axis=1)   # national mean across all states

# ---- per-target fits: null + national-average-only (47 fits) ------------------------------

eval_target <- function(target) {                                                            # py: def eval_target(target):
  tryCatch({                                                                                 # py: try:
    y    <- ts(w[[target]], frequency = 52)                                                  # py:     y    = pd.Series(w[target].values)
    xn   <- c(NA, head(NAVG, -1))                                                            # py:     xn   = np.concatenate(([np.nan], NAVG[:-1]))
    o_n  <- forecast::arimaorder(auto.arima(y[tr]))[1:3]                                     # py:     o_n  = auto_arima(y[tr]).order
    o_v  <- forecast::arimaorder(auto.arima(y[tr], xreg = xn[tr]))[1:3]                      # py:     o_v  = auto_arima(y[tr], X=xn[tr]).order
    m_n  <- Arima(y[tr], order = o_n, seasonal = SEASONAL)                                   # py:     m_n  = SARIMAX(y[tr], order=o_n, seasonal_order=...).fit()
    m_v  <- Arima(y[tr], order = o_v, seasonal = SEASONAL, xreg = xn[tr])                    # py:     m_v  = SARIMAX(y[tr], order=o_v, seasonal_order=..., exog=xn[tr]).fit()
    p_n  <- fitted(Arima(y, model = m_n))[te]                                                # py:     p_n  = m_n.apply(y).fittedvalues[te]
    p_v  <- fitted(Arima(y, model = m_v, xreg = xn))[te]                                     # py:     p_v  = m_v.apply(y, exog=xn).fittedvalues[te]
    rn   <- sqrt(mean((y[te] - p_n)^2, na.rm = TRUE))                                        # py:     rn   = np.sqrt(np.nanmean((y[te] - p_n)**2))
    rv   <- sqrt(mean((y[te] - p_v)^2, na.rm = TRUE))                                        # py:     rv   = np.sqrt(np.nanmean((y[te] - p_v)**2))
    cat(sprintf("[target] %-14s  null=%.4f  nav=%.4f  pct_nav=%+6.2f%%\n",                   # py:     print(f"[target] {target} null={rn:.4f} nav={rv:.4f} pct={100*(rv-rn)/rn:+.2f}%")
                target, rn, rv, 100 * (rv - rn) / rn))
    data.frame(target, rmse_null = rn, rmse_nav = rv)                                        # py:     return dict(target=target, rmse_null=rn, rmse_nav=rv)
  }, error = function(e) {                                                                   # py: except Exception as e:
    cat(sprintf("[target] %-14s  [SKIP: %s]\n", target, e$message)); NULL                    # py:     print(f"[target] {target} [SKIP: {e}]"); return None
  })
}

# ---- per-pair fits: biv-only + biv+nav (2162 fits) ----------------------------------------

eval_pair <- function(target, cov) {                                                         # py: def eval_pair(target, cov):
  tryCatch({                                                                                 # py: try:
    y    <- ts(w[[target]], frequency = 52)                                                  # py:     y    = pd.Series(w[target].values)
    xb   <- c(NA, head(w[[cov]], -1))                                                        # py:     xb   = np.concatenate(([np.nan], w[cov].values[:-1]))
    xn   <- c(NA, head(NAVG, -1))                                                            # py:     xn   = np.concatenate(([np.nan], NAVG[:-1]))
    X    <- cbind(biv = xb, nav = xn)                                                        # py:     X    = np.column_stack([xb, xn])
    o_b  <- forecast::arimaorder(auto.arima(y[tr], xreg = xb[tr]))[1:3]                      # py:     o_b  = auto_arima(y[tr], X=xb[tr]).order
    o_bn <- forecast::arimaorder(auto.arima(y[tr], xreg = X[tr, ]))[1:3]                     # py:     o_bn = auto_arima(y[tr], X=X[tr]).order
    m_b  <- Arima(y[tr], order = o_b,  seasonal = SEASONAL, xreg = xb[tr])                   # py:     m_b  = SARIMAX(y[tr], order=o_b,  seasonal_order=..., exog=xb[tr]).fit()
    m_bn <- Arima(y[tr], order = o_bn, seasonal = SEASONAL, xreg = X[tr, ])                  # py:     m_bn = SARIMAX(y[tr], order=o_bn, seasonal_order=..., exog=X[tr]).fit()
    p_b  <- fitted(Arima(y, model = m_b,  xreg = xb))[te]                                    # py:     p_b  = m_b.apply(y,  exog=xb).fittedvalues[te]
    p_bn <- fitted(Arima(y, model = m_bn, xreg = X))[te]                                     # py:     p_bn = m_bn.apply(y, exog=X).fittedvalues[te]
    rb   <- sqrt(mean((y[te] - p_b)^2,  na.rm = TRUE))                                       # py:     rb   = np.sqrt(np.nanmean((y[te] - p_b)**2))
    rbn  <- sqrt(mean((y[te] - p_bn)^2, na.rm = TRUE))                                       # py:     rbn  = np.sqrt(np.nanmean((y[te] - p_bn)**2))
    cat(sprintf("[pair] %-14s | %-14s  biv=%.4f  both=%.4f\n", target, cov, rb, rbn))        # py:     print(f"[pair] {target} | {cov}  biv={rb:.4f}  both={rbn:.4f}")
    data.frame(target, cov, rmse_biv = rb, rmse_both = rbn)                                  # py:     return dict(target=target, cov=cov, rmse_biv=rb, rmse_both=rbn)
  }, error = function(e) {                                                                   # py: except Exception as e:
    cat(sprintf("[pair] %-14s | %-14s  [SKIP: %s]\n", target, cov, e$message)); NULL         # py:     print(f"[pair] {target} | {cov} [SKIP: {e}]"); return None
  })
}

# ---- run everything: per-target then per-pair, then merge & write CSV --------------------

run_all <- function() {                                                                      # py: def run_all():
  cat("--- per-target fits (null + national average) ---\n")                                 # py: print("--- per-target fits ...")
  t1 <- proc.time()["elapsed"]                                                               # py: t1 = time.time()
  tres <- do.call(rbind, Filter(Negate(is.null),                                             # py: tres = pd.DataFrame([r for r in pool.map(eval_target, states) if r])
                  mclapply(states, eval_target, mc.cores = N_CORES)))
  cat(sprintf("[target] %d/%d fitted in %.0fs\n",                                            # py: print(f"[target] {len(tres)}/{len(states)} in {time.time()-t1:.0f}s")
              nrow(tres), length(states), proc.time()["elapsed"] - t1))

  cat("--- per-pair fits (biv + biv+nav) ---\n")                                             # py: print("--- per-pair fits ...")
  pairs <- expand.grid(target = states, cov = states, stringsAsFactors = FALSE) |>           # py: pairs = [(a,b) for a,b in product(states,states) if a!=b]
    subset(target != cov)
  t2 <- proc.time()["elapsed"]                                                               # py: t2 = time.time()
  pres <- do.call(rbind, Filter(Negate(is.null),                                             # py: pres = pd.DataFrame([r for r in pool.starmap(eval_pair, pairs) if r])
                  mclapply(seq_len(nrow(pairs)),
                           function(i) eval_pair(pairs$target[i], pairs$cov[i]),
                           mc.cores = N_CORES)))
  cat(sprintf("[pair] %d/%d fitted in %.0fs\n",                                              # py: print(f"[pair] {len(pres)}/{len(pairs)} in {time.time()-t2:.0f}s")
              nrow(pres), nrow(pairs), proc.time()["elapsed"] - t2))

  res <- merge(pres, tres, by = "target")                                                    # py: res = pres.merge(tres, on="target")
  res$pct_biv  <- 100 * (res$rmse_biv  - res$rmse_null) / res$rmse_null                      # py: res['pct_biv']  = 100*(res.rmse_biv  - res.rmse_null)/res.rmse_null   # biv vs null
  res$pct_nav  <- 100 * (res$rmse_nav  - res$rmse_null) / res$rmse_null                      # py: res['pct_nav']  = 100*(res.rmse_nav  - res.rmse_null)/res.rmse_null   # nav vs null
  res$pct_both <- 100 * (res$rmse_both - res$rmse_null) / res$rmse_null                      # py: res['pct_both'] = 100*(res.rmse_both - res.rmse_null)/res.rmse_null   # both vs null
  res$pct_anti <- 100 * (res$rmse_both - res$rmse_nav)  / res$rmse_nav                       # py: res['pct_anti'] = 100*(res.rmse_both - res.rmse_nav) /res.rmse_nav    # adding biv on top of nav

  for (col in c("pct_biv", "pct_nav", "pct_both", "pct_anti")) {                             # py: for col in ['pct_biv','pct_nav','pct_both','pct_anti']:
    v <- res[[col]]                                                                          # py:     v = res[col]
    cat(sprintf("%-9s helps %4d/%d (%5.1f%%)  median=%+6.2f%%\n",                            # py:     print(f"{col} helps {sum(v<0)}/{len(v)} ({100*np.mean(v<0):.1f}%) median={np.nanmedian(v):+.2f}%")
                col, sum(v < 0, na.rm = TRUE), sum(!is.na(v)),
                100 * mean(v < 0, na.rm = TRUE), median(v, na.rm = TRUE)))
  }
  write.csv(res, "outputs/sarima_cv.csv", row.names = FALSE)                                 # py: res.to_csv("outputs/sarima_cv.csv", index=False)
  res
}

# ---- main: run what hasn't been cached --------------------------------------------------

dir.create("outputs", showWarnings = FALSE)                                                  # py: os.makedirs("outputs", exist_ok=True)
if (!file.exists("outputs/sarima_cv.csv")) run_all()                                         # py: if not os.path.exists(...): run_all()
# Visualization lives in viz.R; run `Rscript viz.R` to regenerate figures.                   # py: # run `python viz.py` to regenerate figures
