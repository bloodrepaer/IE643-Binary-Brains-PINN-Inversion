"""Extend spatial sensor density using existing full-field clean observations.

No PDE is re-simulated. New observations are nested subsets of the existing
full-field observations, with a shared noise realization at retained sensors.
Run: python extend_sensor_density.py
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]  # the extended_studies/ root
SOURCE=ROOT/"datasets_v2_rk4_zero"
DEST=ROOT/"datasets_v2_density_extended"
COUNTS=[5,20,50,100,200,400,800,1600,3200,3969]
ETAS=[.01,.05,.1]
TIERS=["A_smooth","B_gaussian_bump","C_inclusion","D_layered","E_grf","F_defects"]
LOCATION_SEED=6432026


def copy_unchanged(path):
    target=DEST/path.relative_to(SOURCE);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.read_bytes()!=path.read_bytes():raise RuntimeError(f"Conflicting existing problem/answer key: {target}")
    else:shutil.copy2(path,target)


def generate():
    if (DEST/"manifest.json").exists():
        print("Existing extended-density collection retained; validating it")
        return
    manifest=json.loads((SOURCE/"manifest.json").read_text())
    for folder in ("sensors","problem","ground_truth"):(DEST/folder).mkdir(parents=True,exist_ok=True)
    order=np.random.default_rng(LOCATION_SEED).permutation(3969)
    rows=[]
    for tier in TIERS:
        parent=next(e for e in manifest if e["tier"]==tier and e["Fo"]==.1 and e["protocol"]=="uniform8" and e["sensor_set"]=="full" and e["eta"]==0)
        parent_path=SOURCE/parent["files"]["sensors_npz"]
        parent_hash=hashlib.sha256(parent_path.read_bytes()).hexdigest()
        for path in (SOURCE/"problem").glob(parent["problem"]+"*"):copy_unchanged(path)
        copy_unchanged(SOURCE/parent["files"]["ground_truth"])
        with np.load(parent_path) as z:
            clean=z["u_clean"].reshape(8,3969).copy()
            x=z["x"].reshape(8,3969).copy();y=z["y"].reshape(8,3969).copy()
            times=z["t"].reshape(8,3969).copy();time_index=z["time_index"].reshape(8,3969).copy()
        sigma_ref=float(parent["sigma_ref"])
        for eta in ETAS:
            noise_seed=zlib.crc32(f"ie643-density-extended:{tier}:{eta:g}".encode())
            sigma=eta*sigma_ref
            noisy=clean+sigma*np.random.default_rng(noise_seed).standard_normal(clean.shape)
            for n in COUNTS:
                idx=order[:n]
                key=f"{tier}__Fo0.1__uniform8__n{n}__nodes_nested__eta{eta:g}"
                data=dict(x=x[:,idx].ravel(),y=y[:,idx].ravel(),t=times[:,idx].ravel(),
                          u=noisy[:,idx].ravel(),u_clean=clean[:,idx].ravel(),sigma=np.full(8*n,sigma),
                          sensor_id=np.tile(idx,8),time_index=time_index[:,idx].ravel())
                np.savez_compressed(DEST/"sensors"/(key+".npz"),**data)
                with (DEST/"sensors"/(key+".csv")).open("w",newline="",encoding="utf-8") as f:
                    writer=csv.writer(f,lineterminator="\n");writer.writerow(["x","y","t","u","sigma"])
                    writer.writerows(zip(*[[f"{v:.8g}" for v in data[c]] for c in ["x","y","t","u","sigma"]]))
                entry={**parent,"key":key,"family":"DENSITY_EXTENDED","base_tier":tier,
                       "n_sensors":n,"sensor_set":str(n),"layout":"nodes_nested","n_readings":8*n,
                       "eta":eta,"sigma":sigma,"seed_locations":LOCATION_SEED,"seed_noise":noise_seed,
                       "snr_db":20*np.log10(1/eta),"parent_collection":SOURCE.name,"parent_key":parent["key"],
                       "parent_sensor_sha256":parent_hash,"nested_locations":True,"nested_noise":True,
                       "bc_known":True,"source_known":True,
                       "files":{**parent["files"],"sensors_csv":f"sensors/{key}.csv","sensors_npz":f"sensors/{key}.npz"}}
                rows.append(entry)
        print(tier,": 30 density/noise configurations",flush=True)
    (DEST/"manifest.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    columns=[k for k in rows[0] if k not in ("files","observation_times")]
    with (DEST/"manifest.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction="ignore");writer.writeheader();writer.writerows(rows)
    (DEST/"config.json").write_text(json.dumps(dict(sensor_counts=COUNTS,noise_etas=ETAS,tiers=TIERS,
        location_seed=LOCATION_SEED,Fo=.1,protocol="uniform8",n_times=8,layout="nodes_nested",
        source_collection=SOURCE.name,reuses_clean_observations=True,nested_locations=True,nested_noise=True,
        note="Same node pool across fields; one fixed sigma reference per field. Not the historical off-grid layout."),indent=2),encoding="utf-8")


def validate():
    rows=json.loads((DEST/"manifest.json").read_text())
    assert len(rows)==len(TIERS)*len(COUNTS)*len(ETAS)
    index={(e["tier"],e["eta"],e["n_sensors"]):e for e in rows}
    for tier in TIERS:
        for eta in ETAS:
            previous=None
            for n in COUNTS:
                e=index[tier,eta,n]
                with np.load(DEST/e["files"]["sensors_npz"]) as z:
                    arrays={k:z[k].reshape(8,n).copy() for k in z.files}
                assert np.all(arrays["sigma"]==eta*e["sigma_ref"])
                assert len(np.unique(arrays["sensor_id"][0]))==n
                if previous:
                    for key in ("x","y","t","u","u_clean","sigma","sensor_id","time_index"):
                        np.testing.assert_array_equal(arrays[key][:,:previous["u"].shape[1]],previous[key])
                previous=arrays
            # Full-density noise should follow the requested Gaussian scale.
            zscore=(arrays["u"]-arrays["u_clean"])/arrays["sigma"]
            assert abs(zscore.mean())<.04 and .97<zscore.std()<1.03
            parent=SOURCE/"sensors"/(e["parent_key"]+".npz")
            with np.load(parent) as original:
                order=arrays["sensor_id"][0].astype(int)
                np.testing.assert_array_equal(arrays["u_clean"],original["u_clean"].reshape(8,3969)[:,order])
        for path in (DEST/"problem").glob(tier+"__Fo0.1*"):
            assert path.read_bytes()==(SOURCE/path.relative_to(DEST)).read_bytes()
        path=DEST/"ground_truth"/(tier+"__Fo0.1.npz")
        assert path.read_bytes()==(SOURCE/path.relative_to(DEST)).read_bytes()
    print(f"PASS: {len(rows)} configurations; nested locations/readings/noise, Gaussian scales, original clean fields and unchanged physics/answer keys")


def append_campaign():
    import campaign as C
    jobs_file=C.HOME/"jobs.json"
    if not jobs_file.exists():return
    with C.FileLock(C.HOME/"plan.lock"):
        jobs=json.loads(jobs_file.read_text());known={j["id"] for j in jobs}
        settings=json.loads((C.HOME/"settings.json").read_text());added=0
        for e in json.loads((DEST/"manifest.json").read_text()):
            for method in ("grid","adjoint_nn","pinn"):
                job_id=hashlib.sha256(f"{DEST.name}/{e['key']}/{method}/42".encode()).hexdigest()[:16]
                if job_id in known:continue
                jobs.append(dict(id=job_id,collection=DEST.name,key=e["key"],method=method,seed=42,
                    physics_mode="matched",priority=1.5,output=f"campaign/results/{DEST.name}/{job_id}",
                    code=f"campaign/code/{settings['code_hash']}",family="DENSITY_EXTENDED",n_sensors=e["n_sensors"],
                    n_times=e["n_times"],eta=e["eta"],tier=e["tier"],historical_grid=None))
                known.add(job_id);added+=1
        jobs.sort(key=lambda j:(j["priority"],j["collection"],j["key"],j["method"]))
        C.atomic_json(jobs_file,jobs)
        C.atomic_json(C.HOME/"density_extension.json",dict(collection=DEST.name,configurations=180,
            sensor_counts=COUNTS,etas=ETAS,tiers=TIERS,added_jobs=added,total_jobs=len(jobs),added=C.now(),
            note="Existing workers retain their loaded queue; new jobs are available on their next restart. Completed results were retained."))
    print(f"Appended {added} jobs; campaign now has {len(jobs)} planned fits")
    C.summarize()


if __name__=="__main__":
    generate();validate();append_campaign()
