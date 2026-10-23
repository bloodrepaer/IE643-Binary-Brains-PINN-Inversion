"""Case I on a 251 x 251 mesh with output every 1e-3 (1000 snapshots) for the dt-bias study."""
import numpy as np, gen
t, X, Y, D, C = gen.simulate("I", N=251, dt=1e-3, n_out=1000, every=1)
C = C / C.max()
np.savez("/home/claude/thakur/data_I_N251_fine.npz", t=t, x=X[:, 0], D=D, c=C.astype(np.float32))
print("done", C.shape)
