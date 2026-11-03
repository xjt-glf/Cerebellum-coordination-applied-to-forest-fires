import json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from webapp.route_converter import validate_route
OUT=ROOT/'research_outputs/codex_short_routes_3runs_20261022';SEEDS=[730000,730001,730002]
def states():
    for s in SEEDS:
        f=OUT/f'codex_{s}'
        if (f/'FAILED.txt').exists():print((f/'FAILED.txt').read_text());continue
        if (f/'summary.json').exists():print('DONE',s,(f/'summary.json').read_text());continue
        p=f/'waiting.json';print(json.dumps(json.loads(p.read_text()),ensure_ascii=False,separators=(',',':')) if p.exists() else f'STARTING {s}')
def submit(items):
    # Validate every command before committing any file.
    for s,t,orders,reason in items:
        f=OUT/f'codex_{s}';d=json.loads((f/'waiting.json').read_text());assert d['t']==t and len(orders)==4
        for i,o in enumerate(orders):
            assert o['task']!='patrol'
            if 'route' in o:validate_route(o['route'],d['U'][i][1:3])
        assert not (f/f'command_{t:04d}.json').exists()
    for s,t,orders,reason in items:
        path=OUT/f'codex_{s}'/f'command_{t:04d}.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps({'seed':s,'t':t,'plan':{'uavs':orders},'reason':reason},ensure_ascii=False));tmp.replace(path)
    start=time.monotonic()
    while time.monotonic()-start<20:
        if all((OUT/f'codex_{s}'/'summary.json').exists() or (OUT/f'codex_{s}'/'FAILED.txt').exists() or json.loads((OUT/f'codex_{s}'/'waiting.json').read_text())['t']>t for s,t,_,_ in items):break
        time.sleep(.15)
    states()
if __name__=='__main__':states()
