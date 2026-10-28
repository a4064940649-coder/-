import sys; sys.path.insert(0, '.')
from rlib import *
import json
rows = []
for f in sorted(AG.glob('**/*.parquet')):
    rel = f.relative_to(AG)
    if rel.parts[0] == 'o_main_spa_breaks': continue
    try:
        import pyarrow.parquet as pq
        md = pq.read_metadata(f); nc = md.num_columns; nr = md.num_rows
    except Exception as e:
        rows.append(dict(f=str(rel), err=str(e)[:80])); continue
    info = dict(f=str(rel), nrow=nr, ncol=nc)
    if nc <= 6000 and f.stat().st_size < 200e6:
        try:
            d = pd.read_parquet(f)
            info['idx'] = type(d.index).__name__
            if isinstance(d.index, pd.DatetimeIndex) and len(d) > 1:
                dd = np.median(np.diff(d.index.values).astype('timedelta64[D]').astype(float))
                info['med_gap'] = float(dd); info['start'] = str(d.index[0].date()); info['end'] = str(d.index[-1].date())
            info['cols'] = [str(c) for c in d.columns[:8]]
        except Exception as e:
            info['err'] = str(e)[:80]
    rows.append(info)
json.dump(rows, open(OUT / 'inventory.json', 'w'), ensure_ascii=False, indent=0)
for r in rows: print(r.get('f'), r.get('nrow'), r.get('ncol'), r.get('idx'), r.get('med_gap'), r.get('start'), r.get('end'), (r.get('cols') or [])[:4], r.get('err',''))
