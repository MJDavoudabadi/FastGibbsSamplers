# ============================================================
# High-dimensional simulation study for LassoHiDFastGibbs
# Revised to match the actual function signatures you provided
# ============================================================

library(LassoHiDFastGibbs)
library(posterior)

# ------------------------------------------------------------
# 1) Settings
# ------------------------------------------------------------
sim_cfg <- list(
  n_train = 100,
  n_test  = 1000,
  p       = 500,      # high-dimensional setting: p >> n
  s0      = 10,       # number of true nonzero coefficients
  beta_signal = 2.5,
  sigma2_true = 1.0,
  rho = 0.3           # AR(1)-type correlation in predictors
)

mcmc_cfg <- list(
  a = 1, b = 1,
  u = 1, v = 1,
  nsamples = 3000,
  burnin   = 1000,
  lambda_init = 1,
  sigma2_init = 1,
  lower = 1e-12,
  upper = 5000,
  s_beta   = 5L,
  s_siglam = 2L
)

# Choose methods you want to compare
method_names <- c(
  "2BG_bl",
  "2BG_bs",
  "PCG_lambda2_va",
  "PCG_sigma2_va",
  "PCG_beta_sigma2",
  "PCG_lambda2_sigma2",
  "PCG_sigma2_beta",
  "PCG_sigma2_lambda2",
  "NG"
)

# ------------------------------------------------------------
# 2) Data generation
# ------------------------------------------------------------
generate_design <- function(n, p, rho = 0.3) {
  idx <- seq_len(p)
  Sigma <- rho ^ abs(outer(idx, idx, "-"))
  Z <- matrix(rnorm(n * p), n, p)
  Z %*% chol(Sigma)
}

generate_beta <- function(p, s0, beta_signal = 2.5) {
  beta <- rep(0, p)
  vals <- rep(c(beta_signal, -beta_signal), length.out = s0)
  beta[seq_len(s0)] <- vals
  beta
}

generate_dataset <- function(cfg) {
  beta_true <- generate_beta(cfg$p, cfg$s0, cfg$beta_signal)
  
  X_train <- generate_design(cfg$n_train, cfg$p, cfg$rho)
  eps_train <- rnorm(cfg$n_train, sd = sqrt(cfg$sigma2_true))
  y_train <- as.vector(X_train %*% beta_true + eps_train)
  
  X_test <- generate_design(cfg$n_test, cfg$p, cfg$rho)
  eps_test <- rnorm(cfg$n_test, sd = sqrt(cfg$sigma2_true))
  y_test <- as.vector(X_test %*% beta_true + eps_test)
  
  list(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    y_test = y_test,
    beta_true = beta_true,
    sigma2_true = cfg$sigma2_true
  )
}

# ------------------------------------------------------------
# 3) Standardization helpers
#    (self-contained, avoids relying on package normalize())
# ------------------------------------------------------------
standardize_xy <- function(y, X) {
  y_mean <- mean(y)
  y_sd   <- sd(y)
  
  X_means <- colMeans(X)
  X_sds   <- apply(X, 2, sd)
  
  # avoid division by zero if any constant column appears
  X_sds[X_sds == 0] <- 1
  
  vy <- as.numeric((y - y_mean) / y_sd)
  mX <- scale(X, center = X_means, scale = X_sds)
  
  list(
    vy = vy,
    mX = mX,
    y_mean = y_mean,
    y_sd = y_sd,
    X_means = X_means,
    X_sds = X_sds
  )
}

beta_backtransform <- function(beta_std_draws, std_obj) {
  sweep(beta_std_draws, 2, std_obj$y_sd / std_obj$X_sds, "*")
}

sigma2_backtransform <- function(sigma2_std_draws, std_obj) {
  sigma2_std_draws * (std_obj$y_sd^2)
}

# ------------------------------------------------------------
# 4) Fit wrapper aligned to your actual function signatures
# ------------------------------------------------------------
fit_one_method <- function(method, y, X, mcmc_cfg) {
  std_obj <- standardize_xy(y, X)
  
  vy <- std_obj$vy
  mX <- std_obj$mX
  p  <- ncol(mX)
  va_init <- rep(1, p)
  
  t0 <- proc.time()[3]
  
  fit <- switch(
    method,
    
    "2BG_bl" = blasso_gibbs_2block_bl(
      vy = vy,
      mX = mX,
      a = mcmc_cfg$a,
      b = mcmc_cfg$b,
      u = mcmc_cfg$u,
      v = mcmc_cfg$v,
      nsamples = mcmc_cfg$nsamples,
      lambda_init = mcmc_cfg$lambda_init,
      sigma2_init = mcmc_cfg$sigma2_init,
      va_init = va_init,
      verbose = 0,
      lower = mcmc_cfg$lower,
      upper = mcmc_cfg$upper
    ),
    
    "2BG_bs" = blasso_gibbs_2block_bs(
      vy = vy,
      mX = mX,
      a = mcmc_cfg$a,
      b = mcmc_cfg$b,
      u = mcmc_cfg$u,
      v = mcmc_cfg$v,
      nsamples = mcmc_cfg$nsamples,
      lambda_init = mcmc_cfg$lambda_init,
      sigma2_init = mcmc_cfg$sigma2_init,
      verbose = 0
    ),
    
    "PCG_lambda2_va" = blasso_pcg_lambda2_va(
      vy = vy,
      mX = mX,
      a = mcmc_cfg$a,
      b = mcmc_cfg$b,
      u = mcmc_cfg$u,
      v = mcmc_cfg$v,
      nsamples = mcmc_cfg$nsamples,
      lambda_init = mcmc_cfg$lambda_init,
      sigma2_init = mcmc_cfg$sigma2_init,
      verbose = 0
    ),
    
    "PCG_sigma2_va" = blasso_pcg_sigma2_va(
      vy = vy,
      mX = mX,
      a = mcmc_cfg$a,
      b = mcmc_cfg$b,
      u = mcmc_cfg$u,
      v = mcmc_cfg$v,
      nsamples = mcmc_cfg$nsamples,
      lambda_init = mcmc_cfg$lambda_init,
      sigma2_init = mcmc_cfg$sigma2_init,
      va_init = va_init,
      verbose = 0,
      lower = mcmc_cfg$lower,
      upper = mcmc_cfg$upper
    ),
    
    "PCG_beta_sigma2" = penalized_pcg_beta_sigma2(
      vy = vy,
      mX = mX,
      penalty_type = "lasso",
      a = mcmc_cfg$a,
      b = mcmc_cfg$b,
      u = mcmc_cfg$u,
      v = mcmc_cfg$v,
      nsamples = mcmc_cfg$nsamples,
      lambda_init = mcmc_cfg$lambda_init,
      sigma2_init = mcmc_cfg$sigma2_init,
      verbose = 0
    ),
    
    "PCG_lambda2_sigma2" = penalized_pcg_lambda2_sigma2(
      vy = vy,
      mX = mX,
      penalty_type = "lasso",
      a = mcmc_cfg$a,
      b = mcmc_cfg$b,
      u = mcmc_cfg$u,
      v = mcmc_cfg$v,
      nsamples = mcmc_cfg$nsamples,
      lambda_init = mcmc_cfg$lambda_init,
      sigma2_init = mcmc_cfg$sigma2_init,
      verbose = 0
    ),
    
    "PCG_sigma2_beta" = penalized_pcg_sigma2_beta(
      vy = vy,
      mX = mX,
      penalty_type = "lasso",
      a = mcmc_cfg$a,
      b = mcmc_cfg$b,
      u = mcmc_cfg$u,
      v = mcmc_cfg$v,
      nsamples = mcmc_cfg$nsamples,
      lambda_init = mcmc_cfg$lambda_init,
      sigma2_init = mcmc_cfg$sigma2_init,
      va_init = va_init,
      verbose = 0,
      lower = mcmc_cfg$lower,
      upper = mcmc_cfg$upper
    ),
    
    "PCG_sigma2_lambda2" = penalized_pcg_sigma2_lambda2(
      vy = vy,
      mX = mX,
      penalty_type = "lasso",
      a = mcmc_cfg$a,
      b = mcmc_cfg$b,
      u = mcmc_cfg$u,
      v = mcmc_cfg$v,
      nsamples = mcmc_cfg$nsamples,
      lambda_init = mcmc_cfg$lambda_init,
      sigma2_init = mcmc_cfg$sigma2_init,
      va_init = va_init,
      verbose = 0,
      lower = mcmc_cfg$lower,
      upper = mcmc_cfg$upper
    ),
    
    "NG" = penalized_nested_Gibbs(
      vy = vy,
      mX = mX,
      penalty_type = "lasso",
      a = mcmc_cfg$a,
      b = mcmc_cfg$b,
      u = mcmc_cfg$u,
      v = mcmc_cfg$v,
      nsamples = mcmc_cfg$nsamples,
      lambda_init = mcmc_cfg$lambda_init,
      va_init = va_init,
      verbose = 0,
      lower = mcmc_cfg$lower,
      upper = mcmc_cfg$upper,
      s_beta = mcmc_cfg$s_beta,
      s_siglam = mcmc_cfg$s_siglam
    )
  )
  
  elapsed <- proc.time()[3] - t0
  
  fit$time_sec <- elapsed
  fit$std_obj <- std_obj
  fit
}

# ------------------------------------------------------------
# 5) Post-processing helpers
# ------------------------------------------------------------
post_burn <- function(fit, burnin_driver, nsamples_driver = NULL) {
  n_beta <- nrow(fit$mBeta)
  n_sigma2 <- length(fit$vsigma2)
  n_lambda2 <- length(fit$vlambda2)
  
  # If nsamples_driver is supplied, burn-in is interpreted on the driver-iteration scale
  # and converted proportionally for each saved chain length.
  if (!is.null(nsamples_driver)) {
    burn_prop <- burnin_driver / nsamples_driver
    if (burn_prop < 0 || burn_prop >= 1) {
      stop("burnin_driver / nsamples_driver must be in [0,1).")
    }
    
    burn_beta <- floor(burn_prop * n_beta)
    burn_sigma2 <- floor(burn_prop * n_sigma2)
    burn_lambda2 <- floor(burn_prop * n_lambda2)
  } else {
    # Otherwise use the same burn-in count directly for all chains
    burn_beta <- burnin_driver
    burn_sigma2 <- burnin_driver
    burn_lambda2 <- burnin_driver
  }
  
  if (burn_beta >= n_beta) stop("Burn-in for beta is too large.")
  if (burn_sigma2 >= n_sigma2) stop("Burn-in for sigma2 is too large.")
  if (burn_lambda2 >= n_lambda2) stop("Burn-in for lambda2 is too large.")
  
  idx_beta <- seq.int(burn_beta + 1, n_beta)
  idx_sigma2 <- seq.int(burn_sigma2 + 1, n_sigma2)
  idx_lambda2 <- seq.int(burn_lambda2 + 1, n_lambda2)
  
  list(
    Beta = fit$mBeta[idx_beta, , drop = FALSE],
    sigma2 = fit$vsigma2[idx_sigma2],
    lambda2 = fit$vlambda2[idx_lambda2]
  )
}



compute_ess <- function(x) {
  posterior::ess_basic(as.matrix(x))
}

credible_interval <- function(x, prob = 0.95) {
  alpha <- (1 - prob) / 2
  unname(quantile(x, probs = c(alpha, 1 - alpha)))
}

safe_div <- function(a, b) ifelse(b == 0, NA_real_, a / b)

# ------------------------------------------------------------
# 6) Table 2 analogue:
#    mixing %, ESS/time, runtime
# ------------------------------------------------------------
table2_metrics <- function(fit, burnin, nsamples_driver) {
  dr <- post_burn(
    fit,
    burnin_driver = burnin,
    nsamples_driver = nsamples_driver
  )
  
  ess_beta_vec <- apply(dr$Beta, 2, compute_ess)
  ess_sigma2   <- compute_ess(dr$sigma2)
  ess_lambda2  <- compute_ess(dr$lambda2)
  
  N_beta <- nrow(dr$Beta)
  N_sigma2 <- length(dr$sigma2)
  N_lambda2 <- length(dr$lambda2)
  
  ess_beta <- median(ess_beta_vec, na.rm = TRUE)
  
  data.frame(
    Method = NA_character_,
    beta_MixPct = 100 * ess_beta / N_beta,
    beta_Eff = ess_beta / fit$time_sec,
    sigma2_MixPct = 100 * ess_sigma2 / N_sigma2,
    sigma2_Eff = ess_sigma2 / fit$time_sec,
    lambda2_MixPct = 100 * ess_lambda2 / N_lambda2,
    lambda2_Eff = ess_lambda2 / fit$time_sec,
    Time_sec = fit$time_sec
  )
}

# ------------------------------------------------------------
# 7) Table 3 analogue:
#    95% credible intervals and coverage
# ------------------------------------------------------------
table3_metrics <- function(fit, burnin, nsamples_driver, beta_true, sigma2_true) {
  dr <- post_burn(
    fit,
    burnin_driver = burnin,
    nsamples_driver = nsamples_driver
  )
  
  Beta_orig   <- beta_backtransform(dr$Beta, fit$std_obj)
  sigma2_orig <- sigma2_backtransform(dr$sigma2, fit$std_obj)
  
  beta_ci <- t(apply(Beta_orig, 2, credible_interval))
  sigma2_ci <- credible_interval(sigma2_orig)
  lambda2_ci <- credible_interval(dr$lambda2)
  
  beta_cover <- (beta_true >= beta_ci[, 1]) & (beta_true <= beta_ci[, 2])
  sigma2_cover <- (sigma2_true >= sigma2_ci[1]) & (sigma2_true <= sigma2_ci[2])
  
  tab_beta <- data.frame(
    Parameter = paste0("beta", seq_along(beta_true)),
    True = beta_true,
    CI_low = beta_ci[, 1],
    CI_high = beta_ci[, 2],
    Covered = beta_cover
  )
  
  tab_hyp <- data.frame(
    Parameter = c("sigma2", "lambda2"),
    True = c(sigma2_true, NA_real_),
    CI_low = c(sigma2_ci[1], lambda2_ci[1]),
    CI_high = c(sigma2_ci[2], lambda2_ci[2]),
    Covered = c(sigma2_cover, NA)
  )
  
  coverage_summary <- data.frame(
    Coverage_nonzero_beta = mean(beta_cover[beta_true != 0]),
    Coverage_zero_beta = mean(beta_cover[beta_true == 0]),
    Coverage_overall_beta = mean(beta_cover),
    Coverage_sigma2 = sigma2_cover
  )
  
  list(
    beta_table = tab_beta,
    hyper_table = tab_hyp,
    coverage_summary = coverage_summary
  )
}

# ------------------------------------------------------------
# 8) Variable selection from 95% credible intervals
#    Table 4 analogue
# ------------------------------------------------------------
selection_from_ci <- function(beta_draws_orig, level = 0.95) {
  ci <- t(apply(beta_draws_orig, 2, credible_interval, prob = level))
  as.integer(ci[, 1] > 0 | ci[, 2] < 0)
}

selection_metrics <- function(selected, truth_nonzero) {
  truth <- as.integer(truth_nonzero)
  
  TP <- sum(selected == 1 & truth == 1)
  TN <- sum(selected == 0 & truth == 0)
  FP <- sum(selected == 1 & truth == 0)
  FN <- sum(selected == 0 & truth == 1)
  
  data.frame(
    Accuracy = safe_div(TP + TN, TP + TN + FP + FN),
    Sensitivity = safe_div(TP, TP + FN),
    Specificity = safe_div(TN, TN + FP),
    FDR = safe_div(FP, TP + FP),
    FNR = safe_div(FN, TP + FN)
  )
}

# ------------------------------------------------------------
# 9) Optional: predictive performance
# ------------------------------------------------------------
posterior_mean_beta_original <- function(fit, burnin, nsamples_driver) {
  dr <- post_burn(
    fit,
    burnin_driver = burnin,
    nsamples_driver = nsamples_driver
  )
  Beta_orig <- beta_backtransform(dr$Beta, fit$std_obj)
  colMeans(Beta_orig)
}

test_mse <- function(beta_hat, X_test, y_test) {
  mean((y_test - X_test %*% beta_hat)^2)
}

# ------------------------------------------------------------
# 10) Single dataset study
# ------------------------------------------------------------
run_single_dataset_study <- function(sim_cfg, mcmc_cfg, method_names) {
  dat <- generate_dataset(sim_cfg)
  
  fits <- setNames(vector("list", length(method_names)), method_names)
  tab2_list <- vector("list", length(method_names))
  tab3_list <- vector("list", length(method_names))
  
  for (i in seq_along(method_names)) {
    meth <- method_names[i]
    cat("Running:", meth, "\n")
    
    fit <- fit_one_method(
      method = meth,
      y = dat$y_train,
      X = dat$X_train,
      mcmc_cfg = mcmc_cfg
    )
    
    fits[[meth]] <- fit
    
    tmp2 <- table2_metrics(fit, mcmc_cfg$burnin, mcmc_cfg$nsamples)
    tmp2$Method <- meth
    tab2_list[[i]] <- tmp2
    
    tab3_list[[i]] <- table3_metrics(
      fit = fit,
      burnin = mcmc_cfg$burnin,
      nsamples_driver = mcmc_cfg$nsamples,
      beta_true = dat$beta_true,
      sigma2_true = dat$sigma2_true
    )
  }
  
  tab2 <- do.call(rbind, tab2_list)
  rownames(tab2) <- NULL
  
  pred_tab <- data.frame(
    Method = method_names,
    Test_MSE = NA_real_
  )
  
  for (i in seq_along(method_names)) {
    meth <- method_names[i]
    bhat <- posterior_mean_beta_original(
      fits[[meth]],
      mcmc_cfg$burnin,
      mcmc_cfg$nsamples
    )
    pred_tab$Test_MSE[i] <- test_mse(bhat, dat$X_test, dat$y_test)
  }
  
  list(
    data = dat,
    fits = fits,
    table2 = tab2,
    table3 = tab3_list,
    pred_table = pred_tab
  )
}

# ------------------------------------------------------------
# 11) Repeated simulation study for Table 4 analogue
# ------------------------------------------------------------
run_replicated_selection_study <- function(
    nrep = 100,
    sim_cfg,
    mcmc_cfg,
    method_names) {
  
  res <- vector("list", length(method_names))
  names(res) <- method_names
  
  for (meth in method_names) {
    res[[meth]] <- matrix(NA_real_, nrep, 5)
    colnames(res[[meth]]) <- c("Accuracy", "Sensitivity", "Specificity", "FDR", "FNR")
  }
  
  for (r in seq_len(nrep)) {
    cat("Replication:", r, "of", nrep, "\n")
    
    dat <- generate_dataset(sim_cfg)
    truth_nonzero <- dat$beta_true != 0
    
    for (meth in method_names) {
      fit <- fit_one_method(
        method = meth,
        y = dat$y_train,
        X = dat$X_train,
        mcmc_cfg = mcmc_cfg
      )
      

      dr <- post_burn(
        fit,
        burnin_driver = mcmc_cfg$burnin,
        nsamples_driver = mcmc_cfg$nsamples
      )
      Beta_orig <- beta_backtransform(dr$Beta, fit$std_obj)
      
      selected <- selection_from_ci(Beta_orig, level = 0.95)
      met <- selection_metrics(selected, truth_nonzero)
      
      res[[meth]][r, ] <- as.numeric(met[1, ])
    }
  }
  
  out <- do.call(
    rbind,
    lapply(names(res), function(meth) {
      vals <- res[[meth]]
      data.frame(
        Method = meth,
        Accuracy = sprintf("%.3f ± %.3f",
                           mean(vals[, "Accuracy"], na.rm = TRUE),
                           sd(vals[, "Accuracy"], na.rm = TRUE)),
        Sensitivity = sprintf("%.3f ± %.3f",
                              mean(vals[, "Sensitivity"], na.rm = TRUE),
                              sd(vals[, "Sensitivity"], na.rm = TRUE)),
        Specificity = sprintf("%.3f ± %.3f",
                              mean(vals[, "Specificity"], na.rm = TRUE),
                              sd(vals[, "Specificity"], na.rm = TRUE)),
        FDR = sprintf("%.3f ± %.3f",
                      mean(vals[, "FDR"], na.rm = TRUE),
                      sd(vals[, "FDR"], na.rm = TRUE)),
        FNR = sprintf("%.3f ± %.3f",
                      mean(vals[, "FNR"], na.rm = TRUE),
                      sd(vals[, "FNR"], na.rm = TRUE))
      )
    })
  )
  
  rownames(out) <- NULL
  out
}

# ------------------------------------------------------------
# 12) Example run
# ------------------------------------------------------------
set.seed(123)

# For initial testing, reduce workload:
# sim_cfg$p <- 200
# mcmc_cfg$nsamples <- 1500
# mcmc_cfg$burnin <- 500

one_run <- run_single_dataset_study(
  sim_cfg = sim_cfg,
  mcmc_cfg = mcmc_cfg,
  method_names = method_names
)

cat("\n=== Table 2 analogue ===\n")
print(one_run$table2)

cat("\n=== Prediction MSE summary ===\n")
print(one_run$pred_table)

cat("\n=== Table 3 analogue for NG ===\n")
print(one_run$table3[["NG"]]$beta_table)
print(one_run$table3[["NG"]]$hyper_table)
print(one_run$table3[["NG"]]$coverage_summary)

# Start with nrep = 10 for testing; then use 100 in the paper
table4 <- run_replicated_selection_study(
  nrep = 10,
  sim_cfg = sim_cfg,
  mcmc_cfg = mcmc_cfg,
  method_names = method_names
)

cat("\n=== Table 4 analogue ===\n")
print(table4)
