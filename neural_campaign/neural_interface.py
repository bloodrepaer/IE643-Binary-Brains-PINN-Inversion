"""Local read-only comparison and saved-model testing interface.

Run: python -m streamlit run neural_interface.py --server.address 127.0.0.1
"""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import io,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
import torch
from neural_interface_data import ROOT,comparison_catalog,load_comparison,infer_checkpoint

st.set_page_config(page_title='IE643 · Neural inverse diffusion',layout='wide')
torch.set_num_threads(2)
st.title('Neural inverse diffusion')
st.write('Compare neural adjoint and neural PINN reconstructions, then test a saved coefficient model on your own points.')
catalog=comparison_catalog()
compare,test,status=st.tabs(['Compare saved results','Test a checkpoint','Experiment status'])

with compare:
    if not catalog:st.info('No verified saved neural pairs are available yet.')
    else:
        selected=st.selectbox('Saved comparison',range(len(catalog)),format_func=lambda i:catalog[i]['label'],key='comparison')
        item=catalog[selected];st.caption(item['protocol'])
        fields,metrics,truth,raw=load_comparison(item)
        shown=metrics.rename(columns=dict(method='Method',coefficient_error_percent='Coefficient error (%)',replay_sigma='CN replay / noise σ',sensor_fit_sigma='Sensor fit / noise σ',status='Fit status',wall_seconds='Recorded training seconds',optimizer_capped='Reached iteration cap'))
        shown['Method']=shown.Method.map(dict(adjoint_nn='Neural adjoint',pinn='Neural PINN'));st.dataframe(shown,hide_index=True,use_container_width=True)
        st.caption('Coefficient error uses the synthetic answer key after fitting. Sensor fit and coefficient replay are distinct. An empty cap cell means iteration diagnostics were not recorded. Timings can differ in hardware, initialization and scope; consult the protocol before comparing speed.')
        show_truth=st.checkbox('Show synthetic reference field',value=truth is not None,disabled=truth is None)
        arrays=[fields['adjoint_nn'],fields['pinn']];names=['Neural adjoint','Neural PINN']
        if show_truth and truth is not None:arrays.insert(0,truth);names.insert(0,'Synthetic reference')
        lo=min(float(a.min()) for a in arrays);hi=max(float(a.max()) for a in arrays)
        if hi<=lo:hi=lo+max(abs(lo)*1e-6,1e-12)
        fig,axes=plt.subplots(1,len(arrays),figsize=(4.4*len(arrays),4),constrained_layout=True)
        for ax,D,name in zip(np.atleast_1d(axes),arrays,names):
            im=ax.imshow(D.T,origin='lower',extent=[0,1,0,1],vmin=lo,vmax=hi,cmap='viridis');ax.set(title=name,xlabel='x',ylabel='y')
        fig.colorbar(im,ax=list(np.atleast_1d(axes)),shrink=.85,label='Diffusivity D');st.pyplot(fig);plt.close(fig)
        st.caption('All panels share the full colour range. Narrow spikes or near-zero fields can dominate this range; the numerical metrics remain essential.')
        left,right=st.columns(2)
        for area,method in zip([left,right],['adjoint_nn','pinn']):
            D=fields[method];X,Y=np.meshgrid(np.linspace(0,1,D.shape[0]),np.linspace(0,1,D.shape[1]),indexing='ij')
            export=pd.DataFrame(dict(x=X.ravel(),y=Y.ravel(),D=D.ravel()))
            area.download_button('Download '+('neural adjoint' if method=='adjoint_nn' else 'neural PINN')+' field',export.to_csv(index=False),file_name=method+'_D.csv',mime='text/csv')
        st.download_button('Download comparison metrics',metrics.to_csv(index=False),file_name='neural_comparison.csv',mime='text/csv')
        with st.expander('Fit diagnostics and provenance'):
            st.write('Observation collection:',item['collection']);st.write('Case:',item['key']);st.write('Physics assumption:',item['mode'])
            for method,m in raw.items():
                st.write('Neural adjoint' if method=='adjoint_nn' else 'Neural PINN')
                st.json({k:m[k] for k in ['state_model','alpha','gamma','flag','status','optimizer_diagnostics','n_steps','parameter_count'] if k in m})

with test:
    st.subheader('Evaluate a trained checkpoint')
    st.write('Use a saved model or upload model.pt. Supply x,y in the unit square; PINN temperature predictions also use physical time t within the checkpoint horizon. This evaluates the model without retraining.')
    st.caption('Each checkpoint solves its own inverse observation case. Uploading different observations here tests that retained solution; it does not reconstruct a new diffusivity field from those observations.')
    model_source=st.radio('Checkpoint source',['Saved checkpoint','Upload checkpoint'],horizontal=True)
    model_bytes=None
    if model_source=='Saved checkpoint' and catalog:
        choice=st.selectbox('Checkpoint comparison',range(len(catalog)),format_func=lambda i:catalog[i]['label'],key='checkpoint_comparison')
        method=st.selectbox('Coefficient method',['adjoint_nn','pinn'],format_func=lambda m:'Neural adjoint' if m=='adjoint_nn' else 'Neural PINN')
        model_path=catalog[choice]['paths'][method].with_name('model.pt')
        if model_path.exists():model_bytes=model_path.read_bytes()
        else:st.info('This retained fit has no available model checkpoint.')
    elif model_source=='Upload checkpoint':
        upload=st.file_uploader('Trained model checkpoint (.pt)',type=['pt'])
        if upload is not None:model_bytes=upload.getvalue()
    points_source=st.radio('Point input',['Enter CSV','Upload CSV'],horizontal=True)
    points_text=None
    if points_source=='Enter CSV':points_text=st.text_area('Point CSV',value='x,y,t\n0.3,0.3,0\n0.5,0.5,0.1\n0.7,0.65,0.2',height=140)
    else:
        upload=st.file_uploader('Point or sensor readings CSV',type=['csv'])
        if upload is not None:points_text=upload.getvalue().decode('utf-8-sig')
    st.caption('Optional D_reference provides a post-fit coefficient metric on your input points. Optional u provides a temperature-network RMSE for PINN checkpoints. That RMSE is not coefficient replay.')
    if st.button('Evaluate checkpoint',type='primary'):
        if model_bytes is None or points_text is None:st.error('Choose a checkpoint and supply input points.')
        else:
            try:
                points=pd.read_csv(io.StringIO(points_text));result,scores,has_temperature=infer_checkpoint(model_bytes,points)
                st.session_state['inference_output']=(result,scores,has_temperature)
            except Exception as ex:
                st.session_state.pop('inference_output',None);st.error(str(ex))
    if 'inference_output' in st.session_state:
        st.caption('Results from your last evaluation. Evaluate again after changing the checkpoint or input points.')
        result,scores,has_temperature=st.session_state['inference_output'];st.dataframe(result,hide_index=True,use_container_width=True)
        if scores:st.json(scores)
        if not has_temperature:st.caption('This output evaluates the coefficient D. Neural-adjoint temperature and replay require the supplied initial/boundary problem and numerical solver; its saved replay is shown in the comparison tab.')
        st.download_button('Download predictions',result.to_csv(index=False),file_name='coefficient_predictions.csv',mime='text/csv')

with status:
    st.subheader('Saved campaign status')
    for file,title in [('campaign/state.json','Broad campaign'),('campaign/worker_adjoint_nn.json','Local neural-adjoint worker'),('campaign/pinn_review_gate.json','PINN scientific validation gate')]:
        p=ROOT/file
        if p.exists():
            with st.expander(title,expanded=file.endswith('state.json')):st.json(json.loads(p.read_text()))
    for name in ['zhang_v11_gpu','calibrated_v12_gpu']:
        p=ROOT/'campaign/pilots'/name/'pilot.json'
        if p.exists():
            d=json.loads(p.read_text());st.write(name,':',d['status']);st.caption('Saved status only; this interface does not poll or resubmit remote work.')
    st.write('The campaign is incomplete. Queued or missing arms are not treated as losses. The grid baseline, historical results and research pilots are counted separately.')
    st.caption('Reload this page to read newer saved artifacts. The interface performs inference and post-fit evaluation only; it does not modify datasets, train models or submit jobs.')
