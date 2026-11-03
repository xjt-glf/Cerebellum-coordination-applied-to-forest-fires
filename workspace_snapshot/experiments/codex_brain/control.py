"""Submit literal plans and inspect paused environments. This module makes no decisions."""
import os,json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'research_outputs/codex_brain_3runs_20261022'
SEEDS=[730000,730001,730002]
def states():
    for seed in SEEDS:
        folder=OUT/f'codex_{seed}'
        if (folder/'FAILED.txt').exists():print('FAILED',seed,(folder/'FAILED.txt').read_text());continue
        if (folder/'summary.json').exists():print('DONE',seed,json.dumps(json.loads((folder/'summary.json').read_text()),ensure_ascii=False));continue
        p=folder/'waiting.json'
        if not p.exists():print(f'STARTING {seed}');continue
        d=json.loads(p.read_text());compact={'seed':seed,'t':d['t'],'metrics':list(d['metrics'].values()),'fires':[[*c['xy'],c['size'],c['contained'],c['unknown_boundary']] for c in d['true_clusters']],'U':[[u['id'],*u['xy'],u['E'],u['task'],u['target'],u['d'],u['done'],u['roi'],u['link']] for u in d['uavs']],'P':[[r['id'],*r['xy'],r['aoi'],r['link']] for r in d['people']],'unknown_tiles':d['unknown_tiles_x_rows_y_cols_centers_10_30_50_70_90']}
        print(json.dumps(compact,ensure_ascii=False,separators=(',',':')))
def submit(items):
    for seed,t,orders,reason in items:
        folder=OUT/f'codex_{seed}';current=json.loads((folder/'waiting.json').read_text());assert current['t']==t
        path=folder/f'command_{t:04d}.json';assert not path.exists(),'Refusing duplicate decision'
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps({'seed':seed,'t':t,'plan':{'uavs':orders},'reason':reason},ensure_ascii=False));tmp.replace(path)
    start=time.monotonic()
    while time.monotonic()-start<25:
        if all((OUT/f'codex_{seed}'/'summary.json').exists() or json.loads((OUT/f'codex_{seed}'/'waiting.json').read_text())['t']>t or (OUT/f'codex_{seed}'/'FAILED.txt').exists() for seed,t,_,_ in items):break
        time.sleep(.2)
    states()
if __name__=='__main__':states()
