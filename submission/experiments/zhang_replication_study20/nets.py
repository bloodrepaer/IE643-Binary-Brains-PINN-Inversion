"""Shared JAX pieces: float64 MLPs with Xavier-uniform init, and the two
representations of m = log D (paper Sec. 2.1)."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def init_mlp(key, sizes):
    params = []
    for din, dout in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        lim = np.sqrt(6.0 / (din + dout))
        params.append((jax.random.uniform(k, (din, dout), jnp.float64, -lim, lim), jnp.zeros(dout)))
    return params


def mlp(params, x):
    for W, b in params[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    return x @ W + b


class NNField:
    """m(x, y) = m0 + MLP([2x-1, 2y-1]); default 2 hidden layers of width 32."""
    kind = "nn"

    def __init__(self, m0, seed, width=32, depth=2):
        p = init_mlp(jax.random.PRNGKey(seed), [2] + [width] * depth + [1])
        self.theta0, self.unravel = ravel_pytree(p)
        self.theta0 = np.asarray(self.theta0)
        self.m0 = float(m0)

    def m_pt(self, theta, x, y):
        return self.m0 + mlp(self.unravel(theta), jnp.stack([2 * x - 1, 2 * y - 1]))[0]


class GridField:
    """m(x, y) = bilinear interpolation of theta on a K x K node grid over [0,1]^2
    (the project's control grid; identical to P theta at the 65^2 nodes)."""
    kind = "grid"

    def __init__(self, m0, K=16):
        self.K = K
        self.theta0 = np.full(K * K, float(m0))
        self.m0 = float(m0)

    def m_pt(self, theta, x, y):
        K = self.K
        hx = 1.0 / (K - 1)
        fx, fy = x / hx, y / hx
        i = jnp.clip(jnp.floor(fx), 0, K - 2).astype(jnp.int32)
        j = jnp.clip(jnp.floor(fy), 0, K - 2).astype(jnp.int32)
        a, b = fx - i, fy - j
        T = theta.reshape(K, K)
        return ((1 - a) * (1 - b) * T[i, j] + a * (1 - b) * T[i + 1, j]
                + (1 - a) * b * T[i, j + 1] + a * b * T[i + 1, j + 1])


def field_on_grid(field, theta, N):
    xs = jnp.linspace(0.0, 1.0, N)
    X, Y = jnp.meshgrid(xs, xs, indexing="ij")
    f = jax.vmap(lambda x, y: field.m_pt(theta, x, y))
    return f(X.ravel(), Y.ravel())
