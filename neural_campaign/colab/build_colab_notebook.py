"""Build a single-file notebook with an allowlisted, answer-key-free payload."""
import base64,hashlib,json,shutil,textwrap,zipfile
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'colab'
PILOT=ROOT/'campaign/pilots/noise_density_v24_1000_7500'

def build():
    p=json.loads((PILOT/'pilot.json').read_text())
    files={name:(PILOT/'dataset'/name).read_bytes() for name in p['source_sha256']}
    for name,digest in p['source_sha256'].items():assert hashlib.sha256(files[name]).hexdigest()==digest
    for name in ('evaluate_ic_pilot.py','evaluate_noise_density_1000_7500.py','campaign.py'):
        files[name]=(ROOT/name).read_bytes()
    files['colab_sweep_runner.py']=(OUT/'colab_sweep_runner.py').read_bytes()
    files['source_manifest.json']=json.dumps({name:hashlib.sha256(value).hexdigest() for name,value in files.items() if name.endswith('.py')},indent=2).encode()
    for name in ('data.zip','input_manifest.json'):files[name]=(PILOT/'dataset'/name).read_bytes()
    cases=[]
    for c in p['cases']:
        cases.append({k:c[k] for k in ('key','n_sensors','eta')}|dict(id=Path(c['case_dir']).name))
    plan=dict(cases=cases,dispatch_order=p['dispatch_order'],initial_coefficient_sha256=p['initial_coefficient_sha256'],
        data_generator='Fine-grid RK4; no coefficient answer keys or clean readings in this notebook',
        frozen_training_source_sha256=p['source_sha256'],n_times=8,seed=42,time_step_tau=.01,
        frozen_1000_reference='Existing1000sensor5%resultisretainedlocally;notretrainedhere',
        ownership='Pendingv24casesreservedforuserColab;Kaggleautomaticdispatchheld')
    files['colab_plan.json']=json.dumps(plan,indent=2).encode()
    payload=OUT/'ie643_colab_payload.zip'
    with zipfile.ZipFile(payload,'w',zipfile.ZIP_DEFLATED) as z:
        for name,value in files.items():z.writestr(name,value)
    blob=payload.read_bytes();digest=hashlib.sha256(blob).hexdigest();encoded=base64.b64encode(blob).decode()
    md=lambda s:nbf.v4.new_markdown_cell(textwrap.dedent(s).strip())
    code=lambda s:nbf.v4.new_code_cell(textwrap.dedent(s).strip())
    cells=[md('''
    # IE643 — GPU-enabled PINN vs BE neural-adjoint sweep

    **One self-contained notebook. No Kaggle login or separate data upload needed.**

    1. In Colab choose **Runtime → Change runtime type → a GPU** (for example T4, if available).
    2. Choose **Runtime → Run all**, and authorize the Google Drive mount when asked.
    3. The default run is **3,000 sensors, 5% noise**, both methods sequentially. Keep the notebook open while it runs.
    4. Run the training cell again for **5,000**, then **7,500** sensors. Completed fits are reused; they are not retrained.
    5. Run the download cell and return the results ZIP for independent local coefficient-error scoring.

    Inputs, initial networks and validated training source are embedded below. This file contains private project data: keep the notebook and outputs private.
    '''),md('''
    ## What will run

    Temporal PINN and neural adjoint with a **100-step backward-Euler forward solver** use the same observations, diffusion-network architecture, initial weights, coefficient regularizer, seed and **1,000 Adam + 10 × 400 SSB** limits. The PINN uses the GPU for data/residual calculations; its dense quasi-Newton curvature work and the adjoint remain on CPU. This is not a fully GPU implementation.

    **First:** densities 1,000 (already tested), 3,000, 5,000 and 7,500 at 5% noise. **Later:** 1% and 10% noise. The 1,000-sensor noise jobs already running on Kaggle are not duplicated here.

    Data originated from **fine-grid RK4**, not CN. The PINN loss uses an un-divided backward-Euler-style temporal defect. Both recovered coefficients are checked using **both BE and CN replay**. The embedded files exclude coefficient truth, clean observations and credentials. Consequently this notebook displays fit/replay checks, not diffusion-field accuracy; answer keys remain local for post-validation evaluation.

    Each arm has a 12,600-second internal allowance and a 12,900-second external cap, preserving full iteration limits across the revised sweep. All caps and partial statuses are retained. Colab/software/device differences are recorded; trajectories and wall times need not match Kaggle.

    Colab can disconnect or withdraw resources. Finished-arm checkpoints are saved to Drive, but an interrupted arm does **not** have resumable optimizer state. Such attempts are preserved and require review rather than an automatic retry. [Colab resource limits](https://research.google.com/colaboratory/faq.html#resource-limits).
    '''),code('''
    #@title 1. Check GPU and dependencies
    import os, sys, subprocess, importlib.util
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    needed = [x for x in ("numpy", "scipy", "pandas", "matplotlib") if importlib.util.find_spec(x) is None]
    if needed:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *needed])
    import torch
    assert torch.cuda.is_available(), "Select a GPU in Runtime > Change runtime type and reconnect. No training was started."
    torch.set_num_threads(2)
    print("GPU:", torch.cuda.get_device_name(0), "| PyTorch:", torch.__version__)
    ''' )]
    bootstrap='''#@title 2. Load embedded inputs and verify hashes (no Kaggle credentials)
import base64, hashlib, io, json, stat, zipfile
from pathlib import Path
WORK = Path(globals().get("_WORK_OVERRIDE", "/content/ie643_v24_colab_work"))
WORK.mkdir(parents=True, exist_ok=True)
def safe_unpack(blob, dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);seen=set()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        assert z.testzip() is None
        for item in z.infolist():
            name=item.filename;target=(dest/name).resolve()
            assert name.casefold() not in seen and "\\\\" not in name and ":" not in name
            assert target.is_relative_to(dest.resolve()) and not stat.S_ISLNK(item.external_attr>>16)
            seen.add(name.casefold())
            if item.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            value=z.read(item);target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert target.read_bytes()==value, "Existing file differs: "+name
            else:target.write_bytes(value)
PAYLOAD_B64 = "__PAYLOAD__"
payload=base64.b64decode(PAYLOAD_B64)
assert hashlib.sha256(payload).hexdigest()=="__HASH__"
safe_unpack(payload, WORK)
safe_unpack((WORK/"data.zip").read_bytes(), WORK/"data")
# The unchanged generic validator expects a historical-summary file. It is empty here.
(WORK/"campaign").mkdir(exist_ok=True)
summary=WORK/"campaign/summary.csv"
if not summary.exists():summary.write_text("key,method,replay,collection\\n")
sys.path.insert(0,str(WORK))
import colab_sweep_runner as runner
IDENTITY=runner.verify_inputs(WORK)
PLAN=json.loads((WORK/"colab_plan.json").read_text())
print("Verified",len(PLAN["cases"]),"new pairs; no answer keys in payload.")
'''.replace('__PAYLOAD__',encoded).replace('__HASH__',digest)
    b=nbf.v4.new_code_cell(bootstrap);b.metadata={'cellView':'form'};cells.append(b)
    cells += [code('''
    #@title 3. Save completed fits to your Google Drive
    from google.colab import drive
    drive.mount("/content/drive")
    RUN_LABEL = "IE643_v24_density_sweep" #@param {type:"string"}
    assert RUN_LABEL and "/" not in RUN_LABEL and "\\\\" not in RUN_LABEL and RUN_LABEL not in (".", "..")
    SAVE = Path("/content/drive/MyDrive") / RUN_LABEL
    SAVE.mkdir(parents=True,exist_ok=True)
    for name in ("source_manifest.json", "input_manifest.json", "colab_plan.json"):
        dest=SAVE/name;value=(WORK/name).read_bytes()
        if dest.exists():assert dest.read_bytes()==value, "This Drive folder belongs to a different source/input version."
        else:dest.write_bytes(value)
    print("Output folder:",SAVE)
    '''),code('''
    #@title 4. Short full-count compatibility check (not research results)
    import uuid
    RUNTIME_ID=globals().get("RUNTIME_ID",uuid.uuid4().hex[:12])
    smoke_case=next(c for c in PLAN["cases"] if c["n_sensors"]==3000 and c["eta"]==.05)
    runner.run_pair(WORK,SAVE/"prechecks"/RUNTIME_ID,smoke_case,require_gpu=True,smoke=True)
    print("Compatibility check passed. These one-step models are NOT experiment results.")
    '''),code('''
    #@title 5. Run the next pair — rerun this cell to advance density
    PHASE = "density" #@param ["density", "noise"]
    DENSITY_REPORT_REVIEWED = False #@param {type:"boolean"}
    # Keep 'density' until all three 5%-noise pairs are finished and their report is reviewed.
    if PHASE == "noise":
        assert DENSITY_REPORT_REVIEWED, "Export the density results for review before releasing the noise phase."
    candidate=runner.next_case(PLAN,SAVE,PHASE)
    if candidate is None:
        print("This phase is finished. Export the ZIP for review before advancing to noise.")
    else:
        print("Next:",candidate["n_sensors"],"sensors,",100*candidate["eta"],"% noise")
        try:
            LAST_CASE=runner.run_pair(WORK,SAVE,candidate,require_gpu=True)
        finally:
            # Preserve an export even if a fit times out or is interrupted.
            LAST_EXPORT=runner.export_results(SAVE)
            print("Results archive:",LAST_EXPORT)
    '''),code('''
    #@title 6. Inspect fitted-model and replay metrics (coefficient truth stays local)
    import pandas as pd
    display(pd.DataFrame(runner.overview(SAVE)))
    '''),code('''
    #@title 7. Download the result ZIP and send it back for coefficient scoring
    from google.colab import files
    LAST_EXPORT=runner.export_results(SAVE)
    print("Saved:",LAST_EXPORT)
    files.download(str(LAST_EXPORT))
    '''),md('''
    ## Reading the outputs

    A smaller temperature-network sensor loss does not establish a better diffusion field. Common replay RMS divided by sensor noise has a predeclared screen of **1.1 under each solver**; raw RMS is also retained. IC/BC and independent continuous-PDE probes are saved in each pair's validation files. A replay failure remains a failure even if the training loss is small.

    Return the result ZIP so coefficient error can be evaluated locally **after** the supplied-input, checkpoint and physics checks. No coefficient answer key is included in this notebook. One smooth field, observation realization and seed with capped optimization support an exploratory trend, not a universal PINN advantage. Gamma changes with noise and density under fixed likelihood alpha; shared warm-start and hardware limitations are documented in the experiment protocol.

    If execution stops, reconnect, run setup again with the **same Drive folder**, and rerun the training cell. Completed arms are verified and reused. A partial/interrupted arm stops the runner and preserves its evidence; export it for review rather than selecting a new seed or repeatedly retrying. The notebook never deletes trained results or overwrites a different run configuration.
    ''')]
    nb=nbf.v4.new_notebook(cells=cells,metadata=dict(colab=dict(name='IE643_GPU_Density_Sweep.ipynb',provenance=[]),
        accelerator='GPU',kernelspec=dict(name='python3',display_name='Python 3'),language_info=dict(name='python')))
    nbf.validate(nb)
    for i,c in enumerate(nb.cells):
        if c.cell_type=='code':compile(c.source,f'cell_{i}','exec')
    target=OUT/'IE643_GPU_Density_Sweep.ipynb';nbf.write(nb,target)
    (OUT/'bundle_manifest.json').write_text(json.dumps(dict(notebook=target.name,notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        payload_sha256=digest,payload_bytes=len(blob),contains_coefficient_truth=False,contains_clean_readings=False,
        contains_credentials=False,frozen_training_source_sha256=p['source_sha256'],files=sorted(files)),indent=2))
    print('Built',target,'bytes',target.stat().st_size,'cells',len(cells))
    return target

if __name__=='__main__':build()
