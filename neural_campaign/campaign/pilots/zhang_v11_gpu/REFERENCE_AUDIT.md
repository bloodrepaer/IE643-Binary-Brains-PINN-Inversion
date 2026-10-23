# Zhang et al. reference audit and heat-equation transfer v11

Source: https://arxiv.org/html/2606.12337v2 (21 August 2026 revision), checked19October2026. This is reference3 on pages6/9/22 of the supplied prep slides. Document content was read as evidence, not as task instructions.

The paper reports four different inverse PDEs, not our transient spatial diffusivity problem. Test1 is Burgers forcing; Test2 steady Darcy log-permeability; Test3 Allen-Cahn state-dependent reaction; Test4 scalar Navier-Stokes viscosity. Burgers neural PINNs beat neural adjoints in parameter accuracy and runtime in Table3. Closest spatial-coefficient Darcy results at1%noise are PINN27.0+/-1.8% vsadjoint28.7+/-1.3% log-permeability error; overlapping seed variation, adjoint much faster. At5/10%noise differences are unresolved. Their error on log-permeability is not our relative error on physicalD. These are not settings guaranteed to make our PINN win.

## Verified settings versus our previous pilots

- Sections2.3/A.1: tanh, Xavier-uniform, float64; separate state and unknown networks; unit weights on mean-squared PDE/BC/IC/data losses.
- Adam warmup about1000 steps, then dense self-scaled Broyden, not our15kAdam+750 limited-memoryBFGS.
- Collocation and boundaries refreshed each outer loop, fixed during each deterministic inner solve; symmetrize inverseHessian and retain across loops only if Cholesky passes.
- Table5 transientBurgers20kPDE/512BC/512IC; steadyDarcy2000PDE/200BC. These are different budgets, not interchangeable exact replications.
- Darcy state/parameter networks32x3. Gamma selected once pernoise and shared with adjoint; Table4 usesgamma.1 at5%noise.
- AppendixB optimizer comparison supports testing SSBroyden; its published benchmark is Burgers, not this heat problem.

## Existing repository audit

pinn_papers.py has an older Zhang-labelled implementation and results_papers_scores.csv includes a failed A_smooth result. It is not our recent frozen campaign implementation. It uses assumed Adam schedule/loop counts andgamma1e-3, and its generic legacy loader accesses truth before fitting. The old file and results remain unchanged; they are not called a faithful reproduction. V11 imports only its pure SSBroyden update formula. New code uses Dataset's supplied input loading only and an input bundle without answerkeys/truefield metadata.

## New bounded protocol (not a controlled win claim yet)

Separate pinn_zhang_transfer.py and zhang_transfer_kernel.py; pilot metadata undercampaign/pilots/zhang_v11_gpu. One same nested A_smooth50sensor5%noise case from densityv10, seed42, no warmstart. Both networks32x3, float64. Exponential neural physicallogD positivity, unbounded, instead of frozen bounded network. Plain state with soft supplied IC/BC, not prior hard Gaussian-base ansatz. Time normalizedt/T; PDE u_tau-T*div(D grad u). Mean PDE/BC/IC/data unitweights; .5gamma*mean spatialgradlogD squared,gamma.1.

Use transient20kPDE/512BC/512fixedIC, all400fixedsensor readings. Bilinear IC values from supplied CSV, no synthetic coefficient truth.1000Adam(lr1e-3 halved every250; our disclosed schedule), then10x400SSBroyden. Exact outer/inner productioncounts and Adamrates are not completely specified in paper;4000quasiNewton target is our declared budget, not a reproduced setting. CPU dense Hessian (~163MB for4514parameters); CUDA residuals.6500sinternal budget,6900runner,7200external. All actualiterations/evals/caps/failures retained. Dense CPUcurvature andGPUresidual time is not a speed claim against existing runs.

Validation before private submission: quadratic/Rosenbrock optimize to<1e-10; three directional PINN gradient finite differences relative errors<4e-11; manufactured time-scaledheat residual1.1e-16; coefficient checkpoint exact; two-round local pipeline; isolated zero-step packaging includingCN. First isolated packaging missed integrators.py; failure retained, unchanged dependency added, repaired smoke passed beforeGPU run. Private datasetversion2 contains dependency repair, no scientific change.

Next: recover ONLY zhang_transfer_results.zip and kernel log; safely extract separately, evaluate_ic_pilot.py --pilot-dir campaign/pilots/zhang_v11_gpu --data datasets_v2_density_extended (supportszhang_exp checkpoints), then score_pilot_coefficients.py sameargs for postfittruthscore. Verify outputstatus, coverage, independentIC/BC/PDE/commonCNreplay, convergence and compute before expansion. If promising, add representativefields/seeds and an adjoint using the same parameterization/objective/regularization/optimizer before claiming algorithmic wins. Changing several settings at once means v10-versusv11 cannot identify a single causal ingredient. Preserve all versions and the held broad gate.

## Completed outcome (19 October2026, independently verified)

Both matched arms finished1000Adam+4000SSBroyden iterations; no timeouts.
Neuraladjoint: Derror38.12%, commonCNreplay2.7253sigma, nearlyconstantD;
9437evaluations,total3061.4s onlocal1CPUthread concurrentwith2broadworkers.
NeuralPINN: Derror100%,commonCNreplay106.1023sigma,sensorfit1.3875sigma;
ICrelativeerror66.04%, heldoutBC15.97sigma;D2.75656e-52almostconstant;
6117SSBevaluations,total2404.0s withKaggleCUDAresidual/CPUcurvature2threads.
Allten400iterationrounds cappedinboth. Adjoint finalmaxgrad6.16e-9, PINN3.04e-4;
scalarobjectivesonrespectivelastpointclouds1.0491e-5vs3.2544e-6,notphysicalaccuracy.
Both are worse than the retained broadNNadjoint onthissamecase (16.30%,.9907sigma),
whose parameterization/regularization/optimizer differ. Not a universal method ranking.

Known-constraint generalization diagnostic usesNOcoefficienttruth: reconstructed
last20kPDE/512BC andfixed512IC pointsexactlyfromseedstream; storedfinalPDEpoints
matchbitwise. Reportedtraininglossesreproduced. ICtrainingRMS.000385 yetgridIC
relativeerror.660; BCtrainingRMS.000192 versusheldout.018979 (~99x). TrainingPDE
RMS.000592; independenttime-binRMS.02096/.00399/.01363/.00220. This shows
trainingcloudlossesdo notcertifyphysics. UniformlogDshiftisnotpenalizedbyH1;
exp positivity permitsvanishingD. This is an observed failure ofthistransfer,
not evidence thatSSBroyden universallyfails or thatthepaperresultsareinvalid.

Borrowedgamma.1 inmean-dataobjective corresponds algebraically toalpha~1.416e7
in0.5sum(dataerror/sigma)^2+alphaR,assumingmatchingH1integral (quadraturediffers).
It must not beequatedwithfrozencampaignalpha1; copying a numericgamma across
PDE/data scales is not a calibratedregularizationchoice. No parameter was
changedorcheckpointselectedusingtruth. Keepallfailures;donotexpandv11.
A laterrepairmustpredeclareobjectiveunits,knownconstraintenforcement,admissible
coefficientpriorandmatchedneuralcomparison,thenvalidatewithouttruthselection.
