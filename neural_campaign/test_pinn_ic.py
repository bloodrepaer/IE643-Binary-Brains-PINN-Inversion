"""Numerical checks for the known-IC temperature representation (v1 pilot)."""
import io
import unittest
import numpy as np
import torch
from compare_neural import TemperatureNetwork, gaussian_initial_components, residual, temperature_value
from neural_diffusivity import NeuralDiffusivity, CoefficientConfig

EXPR="exp(-((x-0.3)**2+(y-0.3)**2)/(2*0.05**2))+0.7*exp(-((x-0.7)**2+(y-0.65)**2)/(2*0.06**2))"


class InitialConditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_initial_condition_remains_fixed_after_network_changes(self):
        components=gaussian_initial_components(EXPR)
        model=TemperatureNetwork(width=8,depth=2,initial_gaussians=components,T=.4,D_base=.3)
        x=torch.rand(100,3,dtype=torch.float64);x[:,2]=0
        expected=sum(a*torch.exp(-(x[:,0]-cx).square()/(2*sx2)-(x[:,1]-cy).square()/(2*sy2)) for a,cx,cy,sx2,sy2 in components)
        with torch.no_grad():
            for parameter in model.parameters():parameter.add_(torch.randn_like(parameter))
        torch.testing.assert_close(model(x),expected,rtol=1e-12,atol=1e-12)

    def test_base_satisfies_constant_diffusion_in_infinite_domain(self):
        model=TemperatureNetwork(width=8,depth=2,initial_gaussians=gaussian_initial_components(EXPR),T=.4,D_base=.3)
        coefficient=NeuralDiffusivity(CoefficientConfig(6,2),initial_D=.3)
        x=torch.rand(100,3,dtype=torch.float64,requires_grad=True)
        error=residual(model,coefficient,x,.4)
        self.assertLess(float(error.detach().abs().max()),1e-10)

    def test_checkpoint_reloads_full_state_and_invalid_ic_rejected(self):
        config=dict(width=8,depth=2,seed=42,initial_gaussians=gaussian_initial_components(EXPR),T=.4,D_base=.3)
        model=TemperatureNetwork(**config)
        x=torch.rand(100,3,dtype=torch.float64)
        with io.BytesIO() as f:
            torch.save(dict(temperature_config=config,temperature_state_dict=model.state_dict()),f);f.seek(0)
            saved=torch.load(f,weights_only=True)
        restored=TemperatureNetwork(**saved["temperature_config"]);restored.load_state_dict(saved["temperature_state_dict"])
        torch.testing.assert_close(model(x),restored(x),rtol=0,atol=0)

    def test_local_coefficient_initializes_to_same_constant_heat_state(self):
        components=gaussian_initial_components(EXPR)
        fixed=TemperatureNetwork(width=8,depth=2,initial_gaussians=components,T=.4,D_base=.3,boundary_model='zero_images')
        coupled=TemperatureNetwork(width=8,depth=2,initial_gaussians=components,T=.4,D_base=.3,boundary_model='zero_images',local_coefficient=True)
        coefficient=NeuralDiffusivity(CoefficientConfig(6,2),initial_D=.3)
        x=torch.rand(100,3,dtype=torch.float64)
        torch.testing.assert_close(temperature_value(coupled,coefficient,x),fixed(x),rtol=1e-12,atol=1e-12)
        self.assertLess(float(residual(coupled,coefficient,x.requires_grad_(True),.4).detach().abs().max()),1e-10)

    def test_local_coefficient_data_gradient_and_hard_constraints(self):
        config=dict(width=8,depth=2,initial_gaussians=gaussian_initial_components(EXPR),T=.4,D_base=.3,boundary_model='zero_images',local_coefficient=True)
        model=TemperatureNetwork(**config);coefficient=NeuralDiffusivity(CoefficientConfig(6,2),initial_D=.3)
        x=torch.tensor([[.32,.35,.2],[.65,.6,.3]],dtype=torch.float64)
        gradients=torch.autograd.grad(temperature_value(model,coefficient,x).square().sum(),tuple(coefficient.parameters()))
        self.assertGreater(float(sum(g.square().sum() for g in gradients)),1e-8)
        with torch.no_grad():
            for parameter in coefficient.parameters():parameter.add_(.05*torch.randn_like(parameter))
            for parameter in model.parameters():parameter.add_(.05*torch.randn_like(parameter))
        wall=x.repeat(4,1)
        for edge in range(4):wall[2*edge:2*(edge+1),edge//2]=edge%2
        self.assertLess(float(temperature_value(model,coefficient,wall).detach().abs().max()),1e-13)
        initial=x.clone();initial[:,2]=0
        old=TemperatureNetwork(**config);old.load_state_dict(model.state_dict())
        torch.testing.assert_close(temperature_value(model,coefficient,initial),temperature_value(old,NeuralDiffusivity(CoefficientConfig(6,2),initial_D=.3),initial),rtol=0,atol=0)
        restored=TemperatureNetwork(**config);restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(temperature_value(model,coefficient,x),temperature_value(restored,coefficient,x),rtol=0,atol=0)

    def test_local_coefficient_residual_matches_finite_difference(self):
        class VariableCoefficient(torch.nn.Module):
            def diffusivity(self,xy):return .3+.05*xy[:,0:1].square()
        coefficient=VariableCoefficient()
        model=TemperatureNetwork(width=8,depth=2,initial_gaussians=gaussian_initial_components(EXPR),T=.4,D_base=.3,boundary_model='zero_images',local_coefficient=True)
        x=torch.tensor([[.4,.45,.2],[.6,.55,.1]],dtype=torch.float64)
        actual=residual(model,coefficient,x.clone().requires_grad_(True),.4).detach()
        h=2e-5
        value=lambda p:temperature_value(model,coefficient,p).detach()
        center=value(x);partials=[];second=[]
        for axis in range(3):
            delta=torch.zeros_like(x);delta[:,axis]=h
            plus,minus=value(x+delta),value(x-delta)
            partials.append((plus-minus)/(2*h));second.append((plus-2*center+minus)/h**2)
        expected=partials[2]-.4*(coefficient.diffusivity(x[:,:2]).ravel()*(second[0]+second[1])+.1*x[:,0]*partials[0])
        torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-5)
        for expression in ["1+x", "exp(-x**4-y**2)", "exp(-x**2-x*y-y**2)"]:
            with self.assertRaises(ValueError):gaussian_initial_components(expression)

    def test_image_base_enforces_zero_boundaries_after_parameter_changes(self):
        model=TemperatureNetwork(width=8,depth=2,initial_gaussians=gaussian_initial_components(EXPR),T=.4,D_base=.3,boundary_model='zero_images')
        with torch.no_grad():
            for p in model.parameters():p.add_(torch.randn_like(p))
        x=torch.rand(400,3,dtype=torch.float64)
        for edge in range(4):x[100*edge:100*(edge+1),edge//2]=edge%2
        self.assertLess(float(model(x).detach().abs().max()),1e-13)

    def test_image_base_heat_equation_initial_field_and_checkpoint(self):
        # Narrow peaks have negligible initial boundary tails; wider supplied
        # peaks require the pipeline's full-grid compatibility check.
        components=gaussian_initial_components(EXPR.replace('0.05','0.03').replace('0.06','0.03'))
        config=dict(width=8,depth=2,initial_gaussians=components,T=.4,D_base=.3,boundary_model='zero_images')
        model=TemperatureNetwork(**config)
        coefficient=NeuralDiffusivity(CoefficientConfig(6,2),initial_D=.3)
        x=torch.rand(100,3,dtype=torch.float64,requires_grad=True)
        self.assertLess(float(residual(model,coefficient,x,.4).detach().abs().max()),1e-10)
        initial=x.detach().clone();initial[:,2]=0
        original=TemperatureNetwork(width=8,depth=2,initial_gaussians=components,T=.4,D_base=.3)
        self.assertLess(float((model(initial)-original(initial)).detach().abs().max()),1e-12)
        restored=TemperatureNetwork(**config);restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(x),restored(x),rtol=0,atol=0)


if __name__=="__main__":unittest.main()
