import numpy as np
import cma
import dfols


class Chi2Minimizer:
    def __init__(
        self,
        objective_func,
        bounds,
        use_scaling=False,
        verbose=True,
    ):
        """
        Parameters
        ----------
        objective_func : callable
            Function f(x) -> scalar or array-like (chi2 or likelihood)
        bounds : tuple (lower, upper)
            Arrays defining parameter bounds
        use_scaling : bool
            Whether to scale parameters to [0,1]
        verbose : bool
        """
        self.objective_func = objective_func
        self.lower, self.upper = np.array(bounds[0]), np.array(bounds[1])
        self.use_scaling = use_scaling
        self.verbose = verbose

        self.results_to_save = []

    # -------------------------
    # Scaling utilities
    # -------------------------
    def scale(self, x):
        return (x - self.lower) / (self.upper - self.lower)

    def unscale(self, x_scaled):
        return self.lower + x_scaled * (self.upper - self.lower)

    # -------------------------
    # Objective wrappers
    # -------------------------
    def objective(self, x):
        val = self.objective_func(x)
        # return val[0] if isinstance(val, (list, np.ndarray)) else val
        return float(val)

    def objective_safe(self, x):
        val = self.objective_func(x)

        if val is None:
            print("Objective returned None at:", x)
            return 1e30

        val = float(np.asarray(val))

        if not np.isfinite(val):
            print("Objective returned non-finite:", val, "at", x)
            return 1e30

        return val

    def scaled_objective(self, x_scaled):
        x_real = self.unscale(np.array(x_scaled))
        return self.objective(x_real)

    # -------------------------
    # CMA-ES optimizer
    # -------------------------
    def run_cma(self, x0, config):
        x0_scaled = self.scale(x0) if self.use_scaling else x0
        popsize_min = int(4 + 3 * np.log(len(x0_scaled)) + 1)
        opts = {
            'bounds': [0.0, 1.0] if self.use_scaling else [self.lower,self.upper],
            'tolfunrel': config.get("tolfunrel", 5e-5),
            'popsize': config.get("popsize", popsize_min),
            'maxiter': config.get("maxiter", 250),
            'maxfevals': config.get("maxfevals", 2500),
            'tolstagnation': config.get("tolstagnation", 75),
            'verb_disp': 50 if self.verbose else 0,
        }

        es = cma.CMAEvolutionStrategy(
            x0_scaled,
            config.get("sigma0", 0.3),
            opts
        )

        es.optimize(self.scaled_objective if self.use_scaling else self.objective)

        result = es.result
        x_best = result.xbest

        if self.use_scaling:
            x_best = self.unscale(x_best)

        return x_best, result.fbest

    # -------------------------
    # DFOLS optimizer
    # -------------------------
    def run_dfols(self, x0, config):

        min_npt = 2*len(x0)+1
        default_npt = 2*len(x0)+1

        def objc(x):
            # return self.objective_safe(x)
            val = self.objective_safe(x)
            return np.array([val])

        result = dfols.solve(
            objc,
            x0,
            bounds=(self.lower, self.upper),
            scaling_within_bounds=True,
            rhobeg=config.get("rhobeg", 0.2),
            rhoend=config.get("rhoend", 5e-5),
            npt=config.get("npt", min_npt),
            maxfun=config.get("maxfun", 1500),
            print_progress=self.verbose,
            do_logging=False,
        )
        return  result.x, float(self.objective_safe(result.x))

    # -------------------------
    # Full optimization pipeline
    # -------------------------
    def fit(self, x0, n_stages=3, cma_configs=None, dfols_config=None):
        """
        Run combined CMA-ES + DFOLS optimization
        """

        x_current = np.array(x0)
        f_current = self.objective(x_current)

        if self.verbose:
            print(f"Initial f(x): {f_current}")

        for k in range(n_stages):
            if self.verbose:
                print(f"\n--- Stage {k} ---")

            cma_config = cma_configs[k] if cma_configs else {}

            if self.verbose:
                print(f"--- CMA ---")
            # ---- CMA step ----
            x_cma, f_cma = self.run_cma(x_current, cma_config)

            if x_cma is None or f_cma is None:
                raise ValueError("CMA returned invalid result")

            if f_cma < f_current:
                x_best, f_best = x_cma, f_cma
            else:
                x_best, f_best = x_current, f_current

            if self.verbose:
                print("After CMA:", f_best)

            self.results_to_save.append(("CMA", k, f_best, x_best.copy()))

            if self.verbose:
                print(f"--- DFOLS ---")
            # ---- DFOLS refinement ----
            x_dfols, f_dfols = self.run_dfols(
                x_best,
                dfols_config or {}
            )

            if f_dfols < f_best:
                x_best, f_best = x_dfols, f_dfols

            if self.verbose:
                print("After DFOLS:", f_best)

            self.results_to_save.append(("DFOLS", k, f_best, x_best.copy()))

            x_current, f_current = x_best, f_best

        return x_current, f_current

    # -------------------------
    # Save results
    # -------------------------
    def save_results(self, filename, inv_affine_transform, fmt=None):
        data = []
        for method, k, fval, x in self.results_to_save:
            row = [k, fval] + list(inv_affine_transform(np.array(x)))
            data.append(row)

        np.savetxt(filename, np.array(data), fmt=fmt or "%.6g")