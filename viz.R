#!/usr/bin/env Rscript                                                                      # py: #!/usr/bin/env python3

# viz.R -- Generate figures from outputs/ CSVs produced by ar.R.
# Currently produces:
#   pix/sarima_heatmap.pdf / .png   -- bilateral SARIMA RMSE-change heatmap (Fig. 1)

suppressPackageStartupMessages({                                                             # py: with warnings.catch_warnings(): warnings.simplefilter("ignore")
  library(ggplot2)                                                                           # py: import matplotlib.pyplot as plt  (or seaborn)
})

# ---- Fig. 1: bilateral heatmap (target x cov -> relative RMSE change) ---------------------

heatmap_bilateral <- function(bilateral_csv = "outputs/sarima_cv.csv",                       # py: def heatmap_bilateral(csv="outputs/sarima_cv.csv", out_pdf="pix/sarima_heatmap.pdf"):
                              out_pdf       = "pix/sarima_heatmap.pdf") {
  if (!file.exists(bilateral_csv)) {                                                         # py: if not os.path.exists(csv):
    cat(sprintf("Missing %s; run ar.R first.\n", bilateral_csv)); return(invisible(NULL))    # py:     print(f"Missing {csv}; run ar.R first."); return
  }
  d <- read.csv(bilateral_csv)                                                               # py: d = pd.read_csv(csv)
  d <- d[!is.na(d$pct_biv) & d$cov %in% unique(d$target), ]                                  # py: d = d.dropna(subset=['pct_biv']); d = d[d.cov.isin(d.target.unique())]
  d$pct_clip <- pmax(pmin(d$pct_biv, 10), -10)                                               # py: d['pct_clip'] = d['pct_biv'].clip(-10, 10)
  rank <- aggregate(pct_biv ~ target, data = d, FUN = median)                                # py: rank = d.groupby('target').pct_biv.median().reset_index()
  rank <- rank[order(rank$pct_biv), "target"]                                                # py: rank = rank.sort_values('pct_biv').target.tolist()
  d$target <- factor(d$target, levels = rank)                                                # py: d['target'] = pd.Categorical(d.target, categories=rank, ordered=True)
  d$cov    <- factor(d$cov,    levels = rank)                                                # py: d['cov']    = pd.Categorical(d.cov,    categories=rank, ordered=True)
  g <- ggplot(d, aes(x = cov, y = target, fill = pct_clip)) +                                # py: pivot = d.pivot("target","cov","pct_clip")
    geom_tile() +                                                                            # py: sns.heatmap(pivot, cmap="RdBu_r", center=0, vmin=-10, vmax=10)
    scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B",                   # py: cmap="RdBu_r"
                         midpoint = 0, limits = c(-10, 10), name = "Rel RMSE %") +
    theme_minimal(base_size = 18) +                                                          # py: plt.style.use("default")
    theme(axis.text.x  = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 11),        # py: plt.xticks(rotation=90, fontsize=11); plt.yticks(fontsize=11)
          axis.text.y  = element_text(size = 11),
          axis.title   = element_text(size = 18),                                            # py: ax.set_xlabel(..., fontsize=18); ax.set_ylabel(..., fontsize=18)
          legend.title = element_text(size = 14),                                            # py: cbar.set_label(fontsize=14)
          legend.text  = element_text(size = 12)) +                                          # py: cbar.ax.tick_params(labelsize=12)
    labs(x = "covariate state", y = "target state")                                          # py: ax.set_xlabel("covariate state"); ax.set_ylabel("target state")
  dir.create("pix", showWarnings = FALSE)                                                    # py: os.makedirs("pix", exist_ok=True)
  ggsave(out_pdf, g, width = 9, height = 9)                                                  # py: plt.savefig(out_pdf, bbox_inches="tight")
  ggsave(sub("\\.pdf$", ".png", out_pdf), g, width = 9, height = 9, dpi = 150)               # py: plt.savefig(out_pdf.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
  cat(sprintf("Saved heatmap: %s\n", out_pdf))                                               # py: print(f"Saved heatmap: {out_pdf}")
}

heatmap_bilateral()                                                                          # py: heatmap_bilateral()

# ---- Fig. 2: distribution of pct_anti -- adding neighbour state on top of national average -

density_anti <- function(bilateral_csv = "outputs/sarima_cv.csv",                            # py: def density_anti(bilateral_csv="outputs/sarima_cv.csv",
                         out_pdf       = "pix/sarima_density.pdf") {                         #                    out_pdf="pix/sarima_density.pdf"):
  if (!file.exists(bilateral_csv)) {                                                         # py: if not os.path.exists(bilateral_csv):
    cat("Missing CSV; run ar.R first.\n"); return(invisible(NULL))                           # py:     print("Missing CSV; run ar.R first."); return
  }
  d   <- read.csv(bilateral_csv)                                                             # py: d   = pd.read_csv(bilateral_csv)
  v   <- pmax(pmin(d$pct_anti, 15), -15)                                                     # py: v   = d['pct_anti'].clip(-15, 15).values
  med <- median(d$pct_anti, na.rm = TRUE)                                                    # py: med = np.nanmedian(d['pct_anti'])
  g <- ggplot(data.frame(pct = v), aes(x = pct)) +                                           # py: fig, ax = plt.subplots()
    geom_histogram(aes(y = after_stat(density)), bins = 50,                                  # py: ax.hist(v, bins=50, density=True, alpha=0.35, color="#4575B4")
                   fill = "#4575B4", colour = "#4575B4", alpha = 0.35) +
    geom_density(colour = "#4575B4", linewidth = 1) +                                        # py: # KDE overlay -- use scipy.stats.gaussian_kde
    geom_vline(xintercept = 0,   colour = "grey40", linewidth = 0.4) +                       # py: ax.axvline(0, color='grey', lw=0.4)
    geom_vline(xintercept = med, colour = "#B2182B", linetype = "dashed", linewidth = 0.8) + # py: ax.axvline(med, ls='--', color='#B2182B')
    labs(x = "Change in RMSE (%)", y = NULL) +                                               # py: ax.set_xlabel("Change in RMSE (%)"); ax.set_ylabel("")
    theme_minimal(base_size = 16) +                                                          # py: plt.style.use("default")
    theme(axis.text.y  = element_blank(),                                                    # py: ax.set_yticklabels([])
          axis.ticks.y = element_blank(),                                                    # py: ax.tick_params(axis='y', length=0)
          panel.grid   = element_blank())                                                    # py: ax.grid(False)
  dir.create("pix", showWarnings = FALSE)                                                    # py: os.makedirs("pix", exist_ok=True)
  ggsave(out_pdf, g, width = 8, height = 5)                                                  # py: plt.savefig(out_pdf, bbox_inches='tight')
  ggsave(sub("\\.pdf$", ".png", out_pdf), g, width = 8, height = 5, dpi = 150)               # py: plt.savefig(out_pdf.replace('.pdf','.png'), dpi=150, bbox_inches='tight')
  cat(sprintf("Saved pct_anti distribution: %s\n", out_pdf))                                 # py: print(f"Saved pct_anti distribution: {out_pdf}")
}

density_anti()                                                                               # py: density_anti()
