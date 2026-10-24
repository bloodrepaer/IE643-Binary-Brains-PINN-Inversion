# archive

Code kept for completeness. No study in this folder imports or runs it.

| File | Why it is here |
|---|---|
| `compare_integrators.py` | Byte-identical copy of `../../submission/compare_integrators.py` (measured orders of accuracy of RK4, CN and forward Euler). It verifies the forward solver of the main code; the studies here do not depend on it. |
| `check_problem_spec.py` | Byte-identical copy of `../../submission/check_problem_spec.py`. It rebuilds each forward problem from the `problem/` files alone and replays the true D, checking that a collection carries everything an inversion needs. By default it checks a random sample of 20 datasets (`--n`). |

Both run from `extended_studies/`, for example `python run.py archive/check_problem_spec.py datasets_v2_rk4_zero --n 389`.
