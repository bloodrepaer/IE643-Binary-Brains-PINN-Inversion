"""Numerical checks of the neural coefficient, discrete chain rule and PINN flux."""
import io
import unittest
import numpy as np
import torch
from adjoint_inversion import AdjointSolver
from adjoint_nn import NeuralAdjointSolver, grid_coordinates
from neural_diffusivity import CoefficientConfig, NeuralDiffusivity
from compare_neural import field_regularizer, residual
from test_adjoint import SyntheticDataset


class NeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shared_initialisation_and_checkpoint(self):
        cfg = CoefficientConfig(width=8, depth=2)
        a, b = NeuralDiffusivity(cfg, .3, 9), NeuralDiffusivity(cfg, .3, 9)
        xy = grid_coordinates(11)
        torch.testing.assert_close(a(xy), b(xy), rtol=0, atol=0)
        torch.testing.assert_close(a.diffusivity(xy), torch.full((121, 1), .3, dtype=torch.float64))
        with io.BytesIO() as stream:
            torch.save(a.checkpoint(), stream); stream.seek(0)
            restored = NeuralDiffusivity.from_checkpoint(torch.load(stream, weights_only=True))
            torch.testing.assert_close(restored(xy), a(xy), rtol=0, atol=0)
        with torch.no_grad():
            a.net[-1].weight.fill_(1e4)
        D = a.diffusivity(torch.rand(100, 2, dtype=torch.float64))
        self.assertTrue(torch.isfinite(D).all())
        self.assertTrue((D >= cfg.d_min*(1-1e-12)).all() and (D <= cfg.d_max*(1+1e-12)).all())

    def test_neural_adjoint_gradient_and_regularisation(self):
        ds = SyntheticDataset(N=13)
        rng = np.random.default_rng(7)
        for reg in ("tikhonov", "tv"):
            net = NeuralDiffusivity(CoefficientConfig(6, 2))
            solver = NeuralAdjointSolver(ds, net, n_steps=12, reg=reg)
            theta = solver.weights() + rng.normal(0, .08, solver.weights().shape)
            direction = rng.normal(size=len(theta)); direction /= np.linalg.norm(direction)
            J, g = solver.objective(theta, 3.)
            eps = 1e-5
            fd = (solver.objective(theta+eps*direction, 3., False)-solver.objective(theta-eps*direction, 3., False))/(2*eps)
            self.assertAlmostEqual(fd/(g@direction), 1., places=5)
            errors = [abs(solver.objective(theta+e*direction, 3., False)-J-e*(g@direction)) for e in (1e-2, 5e-3, 2.5e-3)]
            self.assertTrue(3.8 < errors[0]/errors[1] < 4.2)
            self.assertTrue(3.8 < errors[1]/errors[2] < 4.2)
            m = torch.tensor(solver.m_of(theta), requires_grad=True)
            R = field_regularizer(m, solver); R.backward()
            expected, gradient = solver.regulariser(m.detach().numpy())
            self.assertAlmostEqual(R.item(), expected, places=10)
            np.testing.assert_allclose(m.grad.numpy(), gradient, atol=1e-10)

    def test_forward_matches_grid_representation(self):
        ds = SyntheticDataset(N=11)
        net = NeuralDiffusivity(CoefficientConfig(5, 2))
        nn_solver = NeuralAdjointSolver(ds, net, n_steps=10)
        grid_solver = AdjointSolver(ds, n_steps=10, control=3)
        theta = nn_solver.weights()
        m = nn_solver.m_of(theta)
        J_nn, _ = nn_solver.objective(theta, 2.)
        J_grid, _ = grid_solver.field_objective(m, 2.)
        self.assertAlmostEqual(J_nn, J_grid, places=10)
        np.testing.assert_array_equal(nn_solver._last["W"], grid_solver._last["W"])

    def test_pinn_variable_coefficient_flux(self):
        # u=x^2+y^2+t^2, D=exp(x+2y): div(D grad u)=D(4+2x+4y).
        class U(torch.nn.Module):
            def forward(self, x): return (x*x).sum(1)
        class D(torch.nn.Module):
            def diffusivity(self, xy): return torch.exp(xy[:, :1]+2*xy[:, 1:2])
        x = torch.rand(15, 3, dtype=torch.float64, requires_grad=True)
        T = .7
        exact = 2*x[:, 2]-T*torch.exp(x[:, 0]+2*x[:, 1])*(4+2*x[:, 0]+4*x[:, 1])
        torch.testing.assert_close(residual(U(), D(), x, T), exact)


if __name__ == "__main__":
    unittest.main()
