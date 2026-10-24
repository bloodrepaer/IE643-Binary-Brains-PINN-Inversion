"""CN discrete adjoint with a neural coefficient; no dense network Jacobian."""
import time
import numpy as np
import torch
from scipy.optimize import minimize
from adjoint_inversion import AdjointSolver
from neural_diffusivity import NeuralDiffusivity


def grid_coordinates(N):
    X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N), indexing="ij")
    return torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float64)


class NeuralAdjointSolver(AdjointSolver):
    def __init__(self, ds, coefficient=None, **kwargs):
        super().__init__(ds, **kwargs)
        self.coefficient = coefficient if coefficient is not None else NeuralDiffusivity()
        self.xy = grid_coordinates(ds.N)

    def weights(self):
        return torch.nn.utils.parameters_to_vector(self.coefficient.parameters()).detach().numpy().copy()

    def set_weights(self, theta):
        torch.nn.utils.vector_to_parameters(torch.tensor(theta, dtype=torch.float64), self.coefficient.parameters())

    def m_of(self, theta):
        self.set_weights(theta)
        with torch.no_grad():
            return self.coefficient(self.xy).numpy().ravel()

    def objective(self, theta, alpha, want_grad=True):
        self.set_weights(theta)
        m = self.coefficient(self.xy).ravel()
        value = self.field_objective(m.detach().numpy(), alpha, want_grad)
        if not want_grad:
            return value
        J, g_m = value
        # Discrete adjoint supplies dJ/dm, autograd supplies (dm/dphi)^T g_m.
        gradients = torch.autograd.grad(m, tuple(self.coefficient.parameters()),
                                        grad_outputs=torch.from_numpy(g_m))
        return J, torch.cat([g.ravel() for g in gradients]).detach().numpy()

    def best_constant(self):
        # The parent method assumes K*K optimisation variables. Use a separate
        # grid solver to fit the scalar, without touching neural weights.
        solver = AdjointSolver(self.ds, n_steps=self.disc.n_steps, control=2)
        value = solver.best_constant()
        self.n_evals += solver.n_evals
        return value

    def invert(self, alphas=None, maxiter=150, tau=1.0, verbose=True):
        tic = time.perf_counter()
        m_c = self.best_constant()
        # Preserve the exact shared random features, reset only the output.
        last = self.coefficient.net[-1]
        c = self.coefficient.config
        p = (m_c - np.log(c.d_min)) / np.log(c.d_max / c.d_min)
        if not 0 < p < 1:
            raise ValueError("Fitted constant lies outside coefficient bounds")
        with torch.no_grad():
            last.weight.zero_(); last.bias.fill_(np.log(p / (1 - p)))
        theta = self.weights()
        self.objective(theta, 0, False)
        rms_const = self._last["rms"]
        if alphas is None:
            alphas = len(self.d) * np.logspace(0, -6, 13)
        path, states, history = [], [], []
        flag, chosen = "noise_free" if self.ds.sigma == 0 else "floor_above_sigma", None
        for alpha in alphas:
            hist = []
            res = minimize(self.objective, theta, args=(alpha,), jac=True, method="L-BFGS-B",
                           callback=lambda _: hist.append(float(self._last["misfit"])),
                           options={"maxiter": maxiter, "ftol": 1e-8, "gtol": 1e-6, "maxls": 30})
            theta = res.x
            self.objective(theta, alpha, False)
            rms = float(self._last["rms"])
            path.append(dict(alpha=float(alpha), rms=rms, rms_over_sigma=rms / self.sigma_w,
                             iters=int(res.nit), success=bool(res.success), message=str(res.message)))
            states.append(theta.copy()); history.append(hist)
            if verbose:
                print(f"  adjoint-NN alpha={alpha:.3g} RMS/sigma={rms/self.sigma_w:.3f} iterations={res.nit}", flush=True)
            if self.ds.sigma > 0 and rms <= tau * self.sigma_w:
                chosen, flag = len(path) - 1, "reached_sigma"
                break
            if len(path) >= 3 and path[-1]["rms"] > .99 * path[-2]["rms"] and path[-2]["rms"] > .99 * path[-3]["rms"]:
                break
        if not path:
            raise ValueError("At least one regularisation alpha is required")
        if chosen is None:
            best = min(p["rms"] for p in path)
            chosen = next(i for i, p in enumerate(path) if p["rms"] <= 1.02 * best)
        theta, sel = states[chosen], path[chosen]
        self.objective(theta, sel["alpha"], False)
        return dict(theta=theta, m=self.m_of(theta), D=self._last["D"], W=self._last["W"],
                    alpha=sel["alpha"], rms=self._last["rms"], flag=flag, path=path, history=history,
                    D_const=float(np.exp(m_c)), rms_const=rms_const, n_evals=self.n_evals,
                    wall_s=time.perf_counter()-tic, n_steps=self.disc.n_steps)
