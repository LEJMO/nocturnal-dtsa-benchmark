"""Integrity check: which TEB namelist parameters actually vary per site?
If thermal/radiative properties are identical across all 20 sites, they were
inherited from the CAPITOUL (Toulouse) template, not site-derived.
"""
import re
from pathlib import Path

RUNS = Path(__file__).resolve().parents[3] / 'external' / 'teb_runs'
SCALARS = ['ZBLD', 'ZBLD_HEIGHT', 'ZWALL_O_HOR', 'ZZ0', 'ZGARDEN',
           'ZALB_ROOF', 'ZALB_ROAD', 'ZALB_WALL',
           'ZEMIS_ROOF', 'ZEMIS_ROAD', 'ZEMIS_WALL']
ARRAYS = ['ZHC_ROOF', 'ZTC_ROOF', 'ZD_ROOF', 'ZHC_ROAD', 'ZTC_ROAD', 'ZD_ROAD',
          'ZHC_WALL', 'ZTC_WALL', 'ZD_WALL']

vals = {k: {} for k in SCALARS + ARRAYS}
sites = []
for p in sorted(RUNS.glob('*/input.nml')):
    site = p.parent.name
    sites.append(site)
    t = p.read_text(encoding='utf-8', errors='replace')
    for k in SCALARS:
        m = re.search(rf'^\s*{k}\s*=\s*([-\d.eE+]+)', t, re.M)
        vals[k][site] = m.group(1) if m else None
    for k in ARRAYS:
        m = re.findall(rf'{k}\(1,\d\)\s*=\s*([-\d.eE+]+)', t)
        vals[k][site] = tuple(m) if m else None

print(f'sites examined: {len(sites)}\n')
print(f"{'parameter':16s} {'distinct':>8s}  {'verdict':22s} example value")
print('-' * 92)
varying, constant = [], []
for k in SCALARS + ARRAYS:
    d = {v for v in vals[k].values() if v is not None}
    n = len(d)
    verdict = 'VARIES per site' if n > 1 else 'IDENTICAL all sites'
    (varying if n > 1 else constant).append(k)
    ex = next(iter(d)) if d else 'n/a'
    exs = str(ex)[:34]
    print(f'{k:16s} {n:8d}  {verdict:22s} {exs}')

print('\n=== SUMMARY ===')
print('VARIES per site  :', ', '.join(varying) if varying else '(none)')
print('IDENTICAL (template-inherited):', ', '.join(constant) if constant else '(none)')

# cross-check against what Urban-PLUMBER sitedata actually supplies
import csv
FC = Path(__file__).resolve().parents[3] / 'data' / 'urban-plumber' / 'FullCollection'
one = next(FC.glob('*/*_sitedata_v1.csv'))
params = [r['parameter'] for r in csv.DictReader(one.open(encoding='utf-8'))]
print(f'\nUrban-PLUMBER sitedata supplies {len(params)} parameters:')
for p in params:
    print('   ', p)
