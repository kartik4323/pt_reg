"""CPU fixture diagnostics, explicitly NOT completion or assembly performance evidence."""
import argparse
import copy
import importlib.util
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from . import config,data,render,experiments
from .storage import Store,write
from .imagination_report import export,prepare_review
from .evaluate import evaluate


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--execute-notebook',action='store_true')
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    cfg=config.load(Path(__file__).parent/'configs'/'smoke.json')
    cfg=copy.deepcopy(cfg); cfg.update(render_mode='surfel',framing_fill=.35,oracles=False)
    cfg['matte']['enabled']=True; cfg['solver']['continuous_scale']=True
    dataset=a.out/'dataset'/'dataset.json'
    if not dataset.exists(): data.demo(dataset.parent,6)
    rec=next(r for r in data.inventory(dataset)['cases'] if r['split']=='dev')
    case=data.load_case(dataset,rec,cfg); store=Store(a.out/'smoke',cfg,dataset)
    for stage in ('E0','E1','E2','E4'):
        rows=experiments.STAGES[stage](store,case)
        if any(r['status']!='complete' for r in rows): raise RuntimeError(f'CPU fixture {stage} failed')
        store.index()
    evaluate(store,'dev'); export(store.root,dataset); prepare_review(store.root)
    evidence=data.rendering_evidence(case,cfg)
    camera=render.frame_observed(render.camera_for(case['points'][case['anchor']],256,1.5),evidence['points'][case['anchor']],.35)
    images=[]; diagnostics=[]
    for name,points,normals,exterior,renderer in (
        ('legacy assembly sample',case['points'][case['anchor']],case['normals'][case['anchor']],case['exterior'][case['anchor']],render.render),
        ('dense observed surfels',evidence['points'][case['anchor']],evidence['normals'][case['anchor']],evidence['exterior'][case['anchor']],render.render_surfel)):
        result=renderer(points,normals,exterior,camera)
        image=Image.new('RGB',(256,290),'white'); image.paste(Image.fromarray(result['rgb']),(0,30))
        ImageDraw.Draw(image).text((8,8),name,fill='black'); images.append(image)
        diagnostics.append(dict(renderer=name,points=len(points),foreground_fraction=float(result['valid'].mean()),
            input_only=True,completion_evidence=False))
    sheet=Image.new('RGB',(512,290),'white')
    for i,image in enumerate(images): sheet.paste(image,(i*256,0))
    sheet.save(a.out/'render_comparison.png')
    weights=Path.home()/'.u2net'/'isnet-general-use.onnx'
    import torch
    report=dict(smoke_not_research_evidence=True,cuda_available=torch.cuda.is_available(),execution='CPU_fixture_only',renderer_comparison=diagnostics,
        historical_gallery_segmentation='not_run: pinned model/isolated rembg environment must be provided on VM',
        local_rembg_available=importlib.util.find_spec('rembg') is not None,default_weights_present=weights.exists(),
        selected_priors=__import__('json').loads((store.root/'priors'/f'{rec["id"]}.json').read_text()),
        geometry_only_abstention=next(r for r in store.find('E4',rec['id']) if r['arm']=='B2__deploy')['output']['diagnostics']['abstained'])
    write(a.out/'validation.json',report)
    if a.execute_notebook:
        import os
        import nbformat
        from nbclient import NotebookClient
        notebook_path=Path(__file__).resolve().parent.parent/'imagination_state_analysis.ipynb'
        notebook=nbformat.read(notebook_path,as_version=4)
        os.environ['RUN_GROUP']=str(store.root); os.environ['DATASET']=str(dataset.resolve())
        NotebookClient(notebook,kernel_name='python3',timeout=300,resources=dict(metadata=dict(path=str(notebook_path.parent)))).execute()
        nbformat.write(notebook,a.out/'imagination_state_analysis_executed.ipynb')
    print(a.out/'validation.json')


if __name__=='__main__': main()
