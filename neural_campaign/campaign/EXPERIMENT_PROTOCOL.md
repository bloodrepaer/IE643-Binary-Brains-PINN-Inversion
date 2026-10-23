# Neural coefficient campaign: stage-wise protocol

This is a methodology record for the ongoing campaign, not a completed-results
report. Live counts are in state.json; numerical findings are regenerated in
FINDINGS.md. CAMPAIGN.md records dated operational decisions and failures.

Primary comparison (explicit user priority, 19 October 2026): neural adjoint
versus neural PINN. Both methods represent D with a neural network. Grid
adjoint is a secondary baseline. Report the neural pair first, using identical
observations/known physics and common CN replay, coefficient error, convergence
and hardware-qualified cost. Incomplete or failed consistency checks remain
visible. Do not substitute neural-adjoint-versus-grid wins for PINN wins.

The paper-transfer v11 changes coefficient parameterization, objective and
optimizer. Its retained broad adjoint comparator therefore cannot establish a
controlled method advantage. A corresponding neural-adjoint experiment must
share v11's coefficient network, initialization, observation objective and
regularization, with the optimizer matched where applicable, before drawing
that conclusion. All protocol changes remain versioned and separately counted.

## 1. Preserve the observation problem

The inverse problem is to estimate a positive, spatially varying diffusion
coefficient from temperature observations. Existing synthetic problem files,
initial conditions, sensor readings and ground-truth files are retained.
Replacing the coefficient parameterization does not create a new dataset.
Comparisons pair the same collection, observation key and physics assumption.

The historical grid baseline and the neural adjoint both eliminate the state
through a finite-difference solve. The PINN instead represents the temperature
as a neural function and penalizes the differential equation. Both neural
inverse methods use the same coefficient network. They do not have identical
state representations, optimization algorithms or regularization selection.

Answer keys are used for evaluation after fitting. They are not permitted for
choosing penalties, network versions, training checkpoints or convergence
diagnostics. Unknown boundary/source cases under the assumed-zero model are
reported separately from matched-physics cases.

## 2. Replace the coefficient grid with a neural function

The shared coefficient has three hidden tanh layers of width32,2241 parameters,
float64 arithmetic, and a sigmoid-bounded log-diffusivity between log(.001)
and log(10). Initialization gives a spatially constant field fitted to the
observations, with reproducible hidden features. The grid comparator retains
its original discretization and interpolation controls.

The neural adjoint uses the existing manual discrete CN/Rannacher adjoint to
compute the objective gradient with respect to logD at every inversion-grid
node. A PyTorch vector-Jacobian product then propagates that gradient into the
coefficient network. The finite-difference state constraint is enforced by
the forward solve at each objective evaluation.

The broad neural-adjoint protocol uses the retained discrepancy path and up
to150 L-BFGS-B iterations at each alpha. A discrepancy selection flag is not
proof that an optimizer converged. The selected alpha, all path stages and
iteration-cap diagnostics must accompany comparisons.

## 3. Evaluate physical consistency before expanding PINNs

The original PINN represents temperature with a separate4x64 tanh network,
uses normalized time t/T, and combines sensor, PDE, initial/boundary and
coefficient regularization losses. The initial budget is15000 Adam and750
L-BFGS steps with2048 collocation points, seed42 and fixed alpha1.

Its direct temperature-network sensor error and its coefficient replay error
are different measurements. Common replay means discarding the PINN state,
solving the supplied PDE numerically with the recovered coefficient, then
comparing those predictions with exactly the same sensor observations.
Near-noise temperature interpolation without good coefficient replay is not
evidence of a physically consistent inverse solution.

The completed plain-state models substantially violate the supplied initial
condition. New plain-state batches are therefore held by pinn_review_gate.json.
Previously completed models and their failures remain part of the record.

Separate pilots test supplied-Gaussian initial-condition bases, hard zero
boundary conditions, residual-focused sampling, additional optimization and
a coefficient-coupled local Gaussian base. The last base is an ansatz, not
an exact variable-diffusion solution. All spatial coefficient derivative
paths are included in the strong-form PDE residual. Each version has its
own source hashes, protocol, models, logs and independent checks.

Known-IC agreement, held-out boundaries/PDE residuals and common CN replay
are necessary consistency checks. The screening threshold of1.1 sensor sigma
does not prove coefficient recovery or authorize automatic broad migration.
Larger-budget and warm-start pilots are disclosed as such; they do not replace
equal-budget original results. No tested inverse pilot has yet passed the
full review required for scaling.

## 4. Separate optimization and sampling questions

The independent16384-point mixed-time probe is shared across retained pilots.
Where checkpoints include final training points, training residuals are also
recomputed. Adaptive training and independent mixed sampling have different
measures; their RMS ratio is descriptive. A significant training residual
and an iteration-capped optimizer prevent attributing failure exclusively to
unseen collocation points.

The forward-only diagnostic holds a retained fitted neural-adjoint coefficient
fixed and trains only a fresh temperature network against the PDE with hard
IC/BC. Sensors and CN are evaluation only, and the upload excludes answer keys.
It isolates a forward-state optimization problem, but its initial spatially
varying coefficient differs from the constant initialization of inverse pilots.
Success would not prove inverse identifiability; failure would not prove that
the network cannot express the solution.

A separate dense-adjoint continuation holds the selected alpha fixed and
adds up to750 optimizer iterations from a retained checkpoint. It addresses
iteration caps at3969 sensors and1% noise on the first completed field.
This is additional compute at a preselected case, not a repeated discrepancy
path, a replacement result, or a case chosen by coefficient truth.

## 5. Extend sensor density while retaining paired observations

The added collection contains180 configurations: six fields, ten nested
counts5/20/50/100/200/400/800/1600/3200/3969 and noise1/5/10%, with Fo=.1
and eight uniform observation times. Locations and retained noise samples
are nested within each field/noise group. These are grid-node measurements;
historical off-grid layouts are analyzed separately.

All180 new grid fits are complete. Neural-adjoint density fits are in progress.
Compare complete curves at a fixed field/noise and use identical observations
for method comparisons. Discrete alpha-path selection can produce reversals
between adjacent densities. The flag floor_above_sigma does not establish a
physical floor, especially when every path stage reaches its iteration cap.

## 6. Measure more than sensor fit

The saved analysis includes global relative coefficient error, log-coefficient
error, common replay, temperature-network sensor fit, regularization paths,
optimizer caps and hardware-qualified timing. Defect cases additionally report
Dice, detected count, contrast, localization and background error. Detection
counts alone do not penalize false-positive area.

Forward checks with the answer key are explicitly post-fit evaluation. The
layered-field mismatch on the65-grid decreases substantially on the original
129-grid, whereas increasing CN time steps on the65-grid changes little.
These checks diagnose spatial discretization mismatch; they do not supply
training settings or replace the observations.

Three initialization seeds on fixed canonical A/B/F observations test
initialization variability. They do not estimate uncertainty across sensor
locations or new noise draws. All seeds are retained, with no preferred seed
selected. Their six additional fits are separate from the1510 broad fits.

## 7. Reproduce and retain the record

Frozen broad source is under code/48b49b1f17a0. jobs.json and settings.json
define the broad plan. Historical grid results remain separate from new fits.
Separate diagnostic folders and pilot protocols identify additional compute.
Original observations, trained results, failed attempts and scientific versions
are preserved. Resuming workers checks existing completion files and OS locks.
Bounded coordinators stop dispatch below128MiB free disk space.

Run `python analyze_campaign.py --true-physics` to regenerate saved tables,
figures and FINDINGS.md. The true-physics option evaluates hidden physics only
after fitting; it does not make that information available to inversion.
Run `python evaluate_ic_pilot.py --pilot-dir <inverse-pilot>` for inverse
pilot checks, and `python evaluate_forward_diagnostic.py --pilot-dir
campaign/pilots/fixed_d_forward_v7_gpu` for the different forward-only schema.

Final reporting must state incomplete coverage, failed consistency checks,
different optimization budgets, warm starts, physics misspecification and
hardware differences. Pending or failed fits are not counted as method losses.

## 8. Matched PINN density diagnostic v10 (completed 2026-10-19)

Use pilots/density_v10_gpu/paired_method_comparison.csv for the matched nested
50 versus3969 sensor A_smooth comparison at5%noise,eighttimes. Private inputs
contained no answerkeys. IC/BC/PDE/common replay checked before separate local
coefficient scoring. Both cases failed replay, so this version remains outside
broad campaign completions. Fresh identical architecture/seed,15kAdam750LBFGS,
weights1000,alpha1; per-input sensor-fitted constant initialization. Fixedalpha
and batching rule4096 imply changing relative regularization and Adam sampling
with density. Both optimizers capped; the result does not establish universal
harm from dense sensors or a converged PINN optimum. Retain negative results.

## 9. Calibrated hard-physics paired protocol v12 (19 October2026)

See pilots/calibrated_v12_gpu/pilot.json and its validation reports. This is a
separate matched neural-adjoint/neural-PINN research pair, outside1510 broad
fits. It restores the campaign's existing bounded logD prior and uses supplied
Gaussian hardIC/zeroBC construction, instead of v11's soft sampled constraints.
Both primary methods share coefficient initialization, discrete regularizer,
mean observation objective and SSBroyden schedule. Conversion from the original
fixedalpha1 likelihood gives gamma=2sigma_w^2/(N U^2), not a borrowed gamma.
PINN's unit normalized-time residual penalty remains a declared modeling choice.
Sampling is intentionally early-time weighted. Multiple choices changed from
v11, so a positive result would not isolate a single causal ingredient. No trueD
is used for setup/model selection. Both arms run sequentially on one Kagglehost;
record actual CPU/GPU resource use, budgets and independent constraints/replay.
Preserve v11 failure and all v12 outcomes; no broad migration without validation
and field/seed confirmation. One seed is exploratory. On internalbudget expiry,
restore last completed step/outerround and disclose partialstatus/evaluations.


Post-fit v12 physics probe: diagnose_collocation_gap.py uses16384 independent points with the same declared half-uniform/half-log-time measure as v12, preserves earlier pilot aggregates, and checks the retained4096-point training cloud. state_consistency_diagnostic.json additionally records reproduction of normalized training loss, unchanged checkpoint SHA and state/common-CN sensor disagreement. These use supplied physics and observations only, without answer keys, refitting or selection. A large held-out/training ratio is a generalization diagnostic, not proof of a unique cause or converged optimum.


## 10. Controlled collocation pilot v13

Separate single-case/seed research pilot: two fresh PINNs differ only in4096-point collocation generation (original pseudorandom versus independently scrambled Sobol blocks, same half-uniform/half-log-time measure). Both keep all v12 representation, initialization, data objective, penalty, regularizer and optimizer settings. A fresh matched neural-adjoint control runs sequentially on the same host. All arms retained, no truth-based settings or best-seed selection. The frozen-checkpoint quadrature audit was inconclusive; this tests a hypothesis, not an established repair. Count total additional research compute and per-arm limits separately from the broad campaign. Report numerical control equality/gradient/packaging validation and retained initial OpenMP packaging failure. Independent constraints and all-three-arm checkpoint/replay/protocol checks precede local answer-key scoring; incomplete arms are not losses. No broad migration without fields/seeds and valid consistency.


## 11. Explicitly requested higher-density matched pairs v14

Compare fresh neural PINN and neural adjoint at400and3969sensors,3969first,using the same nestedA_smooth5%noise/eight-time observations. The400-point subset shares its exact noise realization with the3969input. Keep the calibratedv12 architecture/optimizer/randomPDE protocol and full-data objective. Fixed likelihoodalpha1 converts to a density-dependentgamma; constant initialization is sensor-fitted per input, shared within each pair. Therefore density effects are conditional on this declared regularization/initialization protocol. Full-data gradient, packaging, supplied-constraint, checkpoint and common-replay checks are retained; no coefficient truth informs settings. Four additional research fits are outside1510. Report per-arm actual budgets/caps and per-density paired coverage; missing arms are not losses. Higher densities are now the explicit user priority; do not default to another50sensorpilot.


V14 post-fit diagnosis uses the same independent mixed16384point PDE probe as v12/v13. audit_dense_stationarity.py reproduces saved final objective components, checks unchanged checkpointSHA and reports data/PDE/regularizer Euclidean gradients and state/commonCN disagreement, using supplied inputs only. Raw gradient norms/cosines depend on parameterization and scale; neither cancellation nor finite-sample collocation gaps prove convergence, a unique failure cause or an optimal penalty. No refitting or truth-based selection occurs.


## 12. Dense collocation-count control v15

See pilots/dense_resolution_v15_gpu/PROTOCOL.md and validation reports. Three fresh arms at3969observed sensors compare original4096 versus16384 random PDE points plus a matched neural-adjoint control. All representation, full-data loss, regularization, physics weight, mixed-time measure, refresh and iteration schedules are unchanged. Point counts need different compute; use actual resources/evals/caps, not an equal-wall-time speed claim. Motivation is retained dense-model quadrature variability without truth, not a verified inverse repair. All3fits separate researchcompute outside1510, all outcomes retained. Independent known constraints and own3arm protocol/checkpoint/commonCN checks precede answer-key scoring. No broad migration or further50-sensor pilots from this test.


V15 postfit descent_audit, microdescent_audit and central_difference_audit retain every predeclared probe and explicit adaptive refinement trigger. They evaluate perturbed weights in memory, accept no optimizer step, save no trialcheckpoint and use no answerkeys. Final-checkpoint AD agreement is directional, not proof of every component or a unique failure cause. Parameter-space scales depend on representation.


## 13. Dense method-by-optimizer control v16

Four fresh3969sensor arms compare SSBroyden and L-BFGS-B across BOTH neural methods, with16384PDE points for both PINNs. All supplied data, representation, initialization, objectives and iteration limits remain fixed. See pilots/optimizer_v16_gpu/PROTOCOL.md for L-BFGS-B history50/maxls30/gtol1e-12/ftol0 and memory-reset/stopping differences. The common trainer retains unused dense identity bookkeeping for L-BFGS-B; no memory-scaling or unqualified speed claim. Full-count short pipelines, exact stock SSB controls, isolated packaging and own four-arm supplied-constraint/checkpoint/commonCN/sharedSHA/gamma/count/method/optimizer fixtures passed. All four research fits outside1510, all outcomes retained; no smoke outcome or queued status is a scientific result. Independent constraints then own evaluate_dense_optimizer_pair.py must validate ALL4arms before truth scoring. No automatic broad migration.


## 14. Human-requested five-case temporal-consistency test v18

See pilots/temporal_v18_gpu/PROTOCOL.md. Five3969sensor5%noise cases each compare adapted temporal-defect PINN, h-squared continuous-residual PINN control and fresh matched neural adjoint. h=.01normalizedtime is predeclared,not claimed literalpaper timestep or selectedfromtruth. Shared serialized sensor-fittedinitial coefficient permits exact cross-hostinitialweight checks. Fullobservations/regularizer/SSBlimits retained;uniform temporalpaircollocation and effectiveh-squared physics weighting differfromolderpilots. This testslossformat,not fullThakurarchitecture/trainingreproduction. Differenthosts/partialbudgetscaps mustqualifyresults andtiming. All15researchfitsoutsidefrozen1510; no automaticbroadmigration. Firstremoteattemptfailedbeforetraining becauseinputzipwasautoexpanded;reviewedloaderrepair/source/log preserved,all10expandedarmspass/exactoldzipbranchnumerics beforev2submission. No answerkeys inprivateinput. Independentphysics/protocol/commonreplay BEFOREpostfitscores andall5casex3armcoverage required.


## 15. Explicit short ten-dataset temporal screen v19

Five fields x400/3969 sensors,5%noise, compare fresh temporal PINN and NNadjoint with50Adam+50SSB and120s internal/180s external budgets onlocalCPU2threads. Same inheritedv18physics/representation/fullobservations/regularizer;sourceSHAcheckedperarm. Independentcheckpoint/bounds/sharedinitialization/commonCN/ICBC/continuousPDEchecks savedbeforeanswerkeyscoring. All20outputs retained. ALL10PINNtimeoutswithzero completedSSBrounds versus10completeadjoints meanzeroeligiblecompletepairs,not10convergedmethodlosses. This is onlyanearlyequaltrainingbudgetscreen. No automaticrepeat ortruthbasedsettingsselection; fullresultsunderpilots/quick_temporal_v19_local.



## 16. Human-requested matched backward-Euler controls v20

Compare ALL5retainedv18temporalPINNs againstfreshneuraladjoints using100fullbackwardEulersteps(dt/T=.01),matching temporalstep/rule. Coefficientnetwork/bounds/serializedinitialSHA/fullobservations/regularizer/gamma/seed/1000Adam10x400SSBfixed. Do notchangeobservations orchoosecasesbytruth. All5newcontrolsoutside1510. Forward/reference/observation/fieldgradient/fullnetworkgradient/pipeline/checkpoint/sharedprotocol/evaluatorfixturespassedbefore scientificdispatch. CommonBEANDoriginalCNreplaysreportedforeachcoefficient,beforepostfittruthscoring,predeclaredscreen1.1foreach. Noevalsolverselectionbyoutcome. SpatialADversusFD,randomshiftedpairsversusfixedgrid,hardversussoftstate remainintrinsic. Accuracycomparisononly:retainedT4PINNsandlocalCPU2threadBEcontrolsarenotsamehostcostexperiment. PriorCNcomparisonsretained/provisional,nobroadgate release. See matched_temporal_v20_local/PROTOCOL.md andvalidationreports.



## 17. Explicit10000sensor reference pair v21

OneA_smooth5%noise/eighttimesFo.1seed42caseadds6031distinct129fine-gridinteriorlocationswhilepreservingALL3969parentCSVreadings/noiseexactly;total10000sites/80000readings. Inversiongrid65unchanged,not10000independent65gridnodes. Storedfloat32cleanfieldroundoffverified/recorded,generationonly;noanswerkeysintraininginput. Bothfresh temporalPINN/BEneuraladjointshareunchangedparentobservation-fittedinitialweights,fullobservations/regularizer/alpha1gamma,1000Adam10x400SSB/h.01. SamehostsequentialT4/CPU2threadswithmethod-specificCUDAdata/residualresourcesqualified,no unqualifiedspeedclaim. Fullcountsgradient/pipeline/isolatedZIP-expanded/independentknownphysics+own2armevaluatorfixturesPASSED,no truthselection. Version1pretrainingWindowsmanifestpathfailurepreserved,loader-onlyportabilityrepairPASSEDall4exactnumericalpackagesbeforev2retry. Allcaps/partials/outcomesretained,tworesearchfitsoutside1510. See dense10000_v21_gpu/PROTOCOL.md forbudgets/recovery/independentcommonBEandCNscoringbeforetruth.



V21 completed and independently evaluated: both full schedules/capped, no timeouts. See dense10000_v21_gpu/RESULTS.md for the single-case coefficient advantage and both common-solver replay results. No automatic expansion or resumption of human-held v20.


## 18. Explicit1,000sensor reference pair v22

See pilots/dense1000_v22_kaggle/PROTOCOL.md. Fixedsubset ofv21sameCNobservations/noise; no regeneratedtruth/settingsselection. Bothfreshmethods matcharchitecture/serializedinitial/regularizer/full8000objective/alpha1gamma/seed/1000Adam10x400SSB/h.01BE; allfullcountgradient/pipeline/packaging/knownphysics-own2armfixturechecksPASSbeforetruth. SeparateKaggleGPU PINN andCPU BEjobs due4612.684sGPUquota,4200internal4400externalperarm; differenthost/deviceaccuracyonly. BothcommonBE+CNreplayspredeclared/partialsnotlosses/no automaticexpansion. Alloldhumanholds/UIdeferralpreserved.


## 19. Human-authorized noise-density trend v23 (queued after v22)

Predeclared1,000/3,969/10,000 x1%/5%/10%,sameA_smooth/fulltemporalPINN-BEadjointprotocol;reusev21/v22frozen5%cells. See pilots/noise_density_v23/PROTOCOL.md for nesting/commonnoise/sharedwarmstart/gammaconfound/capacity/validationorder/negativepartials/reporting. No dispatch untilv22validated; no otherholdresumption.

V22 completed fullpair/independentlyvalidated before truth: adjoint21.4547%vsPINN35.3716%,adjointbothreplayspass/PINNbothfail,allroundscapped. See ownRESULTS.md. V23 now eligible forpreparation after this result report,notautomaticoldworkresumption.

IMPORTANT2026-10-21provenancecorrection: v18/v21/v22andplannedv23observationsoriginatefromdatasets_v2_rk4_zero viaextend_sensor_density.py,notCN. GroundtruthmetadataRK4verified. PriorCN-generationwordingincorrect;CNisreplay/oldadjointsolver. No numericaldata/resultschange. Seev23data_generator_provenance_correction.json.


## 20. Revisedhuman sweep v24, densityfirstat5%noise

Primarydensities1000/3000/5000/7500; FIRST5%noise(3000case02,5000case05,7500case08),THEN1%/10%. Frozenv221000eta.05reused;activev23cases01/02continueonly. Oldv23queued03..07SUPERSEDEDnotdispatched. Newninepairs18fitswithnestedlocations/commonnoise/sharedinitial/noGTtraining/unchangedphysicsanditerations. Full18armgradient/pipelinechecksPASS,independentownfixtures/packagingpendingcompletion. Globalmax2CPUcasejobs,freeonly,allpartials/capsretained,validatedphysics/checkpoints/commonBEANDCNBEFOREtruth. See noise_density_v24_1000_7500/PROTOCOL.md andpilot.dispatch_order. Allotherheldwork/UIremainsheld. OriginalsourceRK4,notCN.


Colabexecutiontransfer: pendingv24reservedforuserColab;singlefilenotebookcolab/IE643_GPU_Density_Sweep.ipynbwithfrozenvalidatedinputs/source(noanswerkeys). Defaultonepairatatime/density5%first;Drivefinishedarmcheckpoints/partialrefusal/resourcesrecorded. Samefulliterations/physicsand12600/12900caps;PINNCUDA+CPUcurvature/BECPU. LocalshortCUDA/CPUorchestrationandvalidateonlyQApassed,notresearchresults;actualColabcloud/Driveuserrunrequired. Do not launchduplicateKagglecases.


V23 dispatched cases01/02 completed and independently validated before truth: all4full/capped/no timeouts. At1%noise lowerPINNcoefficienterror fails bothreplays, notverifiedwin;at10%adjointlowererror/bothreplayspass. Per-caseRESULTS retainraw/sigmaRMS,ICBC/PDE/resources. No activeremotejob/pendingdownload;allremainingv24reserveduserColab. No automaticdispatch.


First user Colab v24 case02 (3000/5%) recovered and locally independently verified before truth; all frozen source/input identity checks pass, original outputs preserved and only separate local pilot path remapped. Both full/capped/no timeouts; PINN lower Derror but fails both1.1 replay screens, notqualifiedwin. 5000/7500 remain user-owned; densityphase incomplete/no noisephase release. See colab_returns/r672fef/RESULTS.md.


2026-10-21T21:09:43.771244+00:00 HUMAN explicitly reassigns ONLY case08(7500sensors5%noise) to agent while human runs case05(5000/5%) onColab. Supersedes all-reserved ownership for this one case ONLY. FreeGPUquota1265.156s inadequate; unchanged validated fullCPU2sequential pair/12600internal12900externalperarm25800pairloop selected. No scientificsettings/source/input changes; sourceSHA/preflight/packaging and canonicalnoexistingkernel verified. Initialhashguard exposed only CRLFfile versus original LFtextSHA, exactnormalizedtemplate equality confirmed/actualbytesSHA retained, not scientificchange or reviewrejection. Case05/usernoise/v20/allotherholdsremain. Do not duplicatecase08fromColab aftercase05. Dispatchpending,notresult.


2026-10-21T21:11:45.339789+00:00 HUMAN explicitly says varynoiseaswelltocheck: release previously gated six1%/10%pairs at3000/5000/7500NOW, agentowned/freeKaggleCPU. UserstillOWNS5000eta5%case05Colab,doNOTduplicate;7500eta5%case08alreadyRUNNINGuntouched. Noisequeuecase01(3000/1),04(5000/1),07(7500/1),03(3000/10),06(5000/10),09(7500/10), max2activeKaggleCPUcasejobsINCLUDINGcase08. Existing1000allnoise/3000eta5resultsreusedunchanged. Samefrozenvalidatedsource/data/initialSHA/architecture/full1000Adam10x400SSB/12600internal12900externalperarm25800pairloop;noGPU/paid/newsettings/truthselection. Per-inputall18pipelines/all9fixture/packagingproofsretained/sourceSHAverified. Filloneavailableslotcase01thenqueueasactualslotsopen;respectcapacityrejection/nobypass/blindretry. Independentphysics/checkpoints/commonBE+CNbeforetruth,allcaps/negativepartialsretained. UserColabstopaftercase05,remainingnoiseagentowned. Otherfields/v20/UIHELD.


V24case01n3000/1% completevalidatedbeforetruth/fullcappednobudgettimeout: lowerPINNcoefficienterror6.1172%vs38.7571% failsbothreplays,notqualifiedwin. BEadjointCNfails/BEpasses. Case04n5000/1%nowRUNNINGalongside08;usercase05excluded. SeepercaseRESULTS.


V24case08n7500/5% completevalidatedbeforetruth/fullcappednobudgettimeout: PINN8.6164%vsBE25.0896%;BOTHmethods BOTHreplay1.1screensPASS(CNnearboundary). Consistency-qualifiedcoefficientadvantageinonefield/seed/cappedcase,notgeneral/converged/speedproof. Case07n7500/1%nowRUNNINGalongside04;usercase05excluded/awaited. SeepercaseRESULTS.


V24case04n5000/1% completevalidatedbeforetruth/fullcappednobudgettimeout: PINN5.6612%vsBE42.7490%,PINNbothreplayFAIL1.1/BEadjCNFAILandBEpass. Lowercoefficienterrornotqualifiedwin. Case03n3000/10%RUNNINGalongside07;usercase05excluded. SeepercaseRESULTS.


V24case07n7500/1% completevalidatedbeforetruth/fullcappednobudgettimeout: PINN5.6110%vsBE45.6436%,PINNbothreplayFAIL1.1/BEadjCNFAILandBEpass. Notqualifiedwin. All1%noise densitiesaccounted/no newfitrequired;othernoise+densitycellsincomplete. Case06n5000/10%RUNNINGalongside03;usercase05excluded. SeepercaseRESULTS.


V24cases03/06n3000/5000eta10% complete/fullcappednobudgettimeout/validatedbeforetruth: adjointlowercoefficienterrorandbothreplayspass,PINNbothfail. Case09n7500eta10%LASTagentfitnowRUNNING;NOmoreagentdispatch. Usercase05awaited/excludedfromagentownership,notloss. SeepercaseRESULTS.


V24case09n7500/10% complete/fullcappednobudgettimeout/validatedbeforetruth:PINN22.6364%vsBE23.2632%,PINNCNpass/BE1.121848FAIL1.1,BEadjointbothPASS/notqualifiedPINNwin. Allagent-ownedfitsdone,noactiveGPU/CPUremote/pendingdownload/queue. 11/12primarycellscomplete,USERcase05n5000/5%awaitednotloss;partialscientificreportallowed,noadditionalcompute/heldwork.


UserColabcase05n5000eta5% independentrecovery/localcanonicalvalidationBEFOREtruthPASS/fullcappednobudgettimeout. PINN13.0309%vsBE23.7878%,CN1.102375narrowFAIL1.1/BE1.068481pass;BEadjointbothPASS. No rounding/thresholdadjustment. All12cellscomplete;scientificreportonly/noadditionalcompute/heldwork.


Scopedv24noise-density12/12comparisonandscientificreportcomplete;all24fitsfull/capped,onequalifiedPINNcoefficientadvantage,fourqualifiedadjointadvantages,remainingcellsnotqualifiedmethodwins. Finalreportrecordscontinuousreplaysandnarrowthresholds,alllimitsandrawRMS. Nofurtherfit/monitorneededforscope;broaderheldcampaignnotcomplete.
