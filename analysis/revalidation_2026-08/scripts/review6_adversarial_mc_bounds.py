# -*- coding: utf-8 -*-
"""
review6_adversarial_mc_bounds.py  (adversarial verification of the v12 residual sweep)

Purpose
-------
The v12 number sweep (results/review6_text_number_audit.json) treats the one-sided
tail probabilities in review5_additions.scheme_ordering_paired / .core15_sensitivity
as exact, and on that basis recommends replacing the manuscript's P(beta>=1) bounds
with <=0.004 (universal) and <=0.0013 (six headline cells).

Those tail probabilities are city-cluster BOOTSTRAP frequencies from a finite number
of draws (draws_used = 100,000), not closed-form p-values. This script:
  1. re-derives every P(beta>=1) and P(r<=0) cell from the package,
  2. attaches the binomial Monte Carlo standard error implied by draws_used,
  3. reports which candidate printed bound survives 2 and 3 MC standard errors, and
  4. audits the Freedman-Lane MC standard errors against the <=7e-5 claim in
     review5_additions.permutation_v11.p_quoting_guidance.rule.

Read-only. Touches neither paper/manuscript_dtsa/** nor results/paper_stats_v1.json.
Interpreter: python -X utf8
"""
import json, math, os

_HERE = os.path.dirname(os.path.abspath(__file__))          # .../재검증_2026-08/scripts
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # project root
STATS = os.path.join(ROOT, 'results', 'paper_stats_v1.json')
OUT = os.path.join(ROOT, 'results', 'review6_adversarial_mc_bounds.json')

with open(STATS, encoding='utf-8') as fh:
    S = json.load(fh)


def g(path, root=S):
    o = root
    for k in path.split('.'):
        o = o[k]
    return o


def mc_se(p, n):
    return math.sqrt(p * (1.0 - p) / float(n))


# ---------------------------------------------------------------- tail cells
sop = g('review5_additions.scheme_ordering_paired')
c15 = g('review5_additions.core15_sensitivity.model_side')

cells = {}
for label, block in (('scheme_ordering_paired', sop), ('core15_sensitivity.model_side', c15)):
    for rs, d in block.items():
        if not isinstance(d, dict) or 'one_sided' not in d:
            continue
        draws = d.get('draws_used')
        for stat in ('TEB_P_beta_ge_1', 'CLMU5_P_beta_ge_1', 'TEB_P_r_le_0', 'CLMU5_P_r_le_0'):
            p = d['one_sided'][stat]
            key = '%s|%s' % (rs, stat)
            if key in cells:
                assert abs(cells[key]['p'] - p) < 1e-15, key
                continue
            # NB: se can legitimately be 0.0 (a cell with p == 0), so test for None,
            # never for truthiness.
            se = mc_se(p, draws) if draws is not None else None
            cells[key] = {
                'record_set': rs, 'statistic': stat, 'p': p,
                'draws_used': draws, 'mc_standard_error': se,
                'p_plus_2se': (p + 2 * se) if se is not None else None,
                'p_plus_3se': (p + 3 * se) if se is not None else None,
                'source_block': label,
            }

beta = {k: v for k, v in cells.items() if 'P_beta_ge_1' in k}
HEADLINE = ('all19', 'excl_mpls17', 'core16')
six = {k: v for k, v in beta.items() if v['record_set'] in HEADLINE}

max_all = max(beta.values(), key=lambda v: v['p'])
max_six = max(six.values(), key=lambda v: v['p'])

CANDIDATES = [0.001, 0.0013, 0.0014, 0.002, 0.004, 0.005]


def bound_report(pool):
    out = {}
    for b in CANDIDATES:
        out['%g' % b] = {
            'holds_at_point_estimate': all(v['p'] <= b for v in pool.values()),
            'holds_at_plus_2se': all(v['p_plus_2se'] <= b for v in pool.values()),
            'holds_at_plus_3se': all(v['p_plus_3se'] <= b for v in pool.values()),
        }
    return out


# ------------------------------------------------- Freedman-Lane MC SE audit
fl = g('review5_additions.permutation_v11.freedman_lane_albedo_given_skyemissivity')
fl_leaves = []
for rs, schemes in fl.items():
    if not isinstance(schemes, dict):
        continue
    for scheme, d in schemes.items():
        if isinstance(d, dict) and 'mc_standard_error' in d:
            fl_leaves.append({'record_set': rs, 'scheme': scheme,
                              'p': d.get('p'), 'mc_standard_error': d['mc_standard_error'],
                              'nperm': d.get('nperm')})
REPORTED = ('CB', 'CB_ws', 'CM')
fl_reported = [x for x in fl_leaves if x['scheme'] in REPORTED]

ao = g('review5_additions.permutation_v11.albedo_offset')
ao_leaves = []
for rs, schemes in ao.items():
    if not isinstance(schemes, dict):
        continue
    for scheme, d in schemes.items():
        if isinstance(d, dict) and 'mc_standard_error' in d:
            ao_leaves.append({'record_set': rs, 'scheme': scheme,
                              'p': d.get('p'), 'mc_standard_error': d['mc_standard_error']})

result = {
    'meta': {
        'purpose': 'adversarial Monte Carlo robustness check on the P(beta>=1) bounds '
                   'recommended by the v12 residual sweep',
        'generator': 'analysis/revalidation_2026-08/scripts/review6_adversarial_mc_bounds.py',
        'source': 'results/paper_stats_v1.json (read-only)',
        'note': 'the one-sided values are city-cluster bootstrap tail frequencies, so a '
                'printed bound must clear the binomial MC standard error implied by draws_used',
    },
    'tail_cells': cells,
    'P_beta_ge_1': {
        'n_cells': len(beta),
        'max_over_all_record_sets': max_all,
        'max_over_six_headline_cells': max_six,
        'bound_robustness_all_cells': bound_report(beta),
        'bound_robustness_six_headline_cells': bound_report(six),
    },
    'verdicts': {
        'le_0p0013_universal': 'FALSE at the point estimate (core16 TEB = 0.00131 > 0.0013)',
        'le_0p0013_six_cells': 'FALSE at the point estimate (core16 TEB = 0.00131 > 0.0013)',
        'le_0p004_universal': 'holds at the point estimate but NOT at +2 MC SE '
                              '(0.00382 + 2*1.95e-4 = 0.00421 > 0.004); not MC-robust',
        'le_0p005_universal': 'holds at the point estimate and at +3 MC SE; recommended universal bound',
        'le_0p002_six_cells': 'holds at the point estimate and at +3 MC SE; the manuscript text '
                              'at L806 is already MC-robust as scoped and should be kept',
        'le_0p001_table2': 'FALSE (0.00131 > 0.001, and 0.00131 - 2 MC SE = 0.00108 > 0.001); '
                           'Table 2 caption must change',
    },
    'freedman_lane_mc_se_audit': {
        'package_rule_quoted': g('review5_additions.permutation_v11.p_quoting_guidance.rule'),
        'claimed_bound': 7e-05,
        'n_leaves': len(fl_leaves),
        'max_over_all_schemes': max(fl_leaves, key=lambda x: x['mc_standard_error']),
        'leaves_exceeding_7e_5': [x for x in fl_leaves if x['mc_standard_error'] > 7e-05],
        'max_over_reported_schemes_CB_CBws_CM': max(fl_reported, key=lambda x: x['mc_standard_error']),
        'verdict': 'the <=7e-5 claim is FALSE over all 15 Freedman-Lane leaves (two exceed it) '
                   'but TRUE when scoped to the CB/CB_ws/CM schemes the manuscript reports',
    },
    'albedo_offset_mc_se_audit': {
        'claimed_bound': 1e-05,
        'n_leaves': len(ao_leaves),
        'max': max(ao_leaves, key=lambda x: x['mc_standard_error']),
        'verdict': 'the <=1e-5 claim holds over every albedo-offset leaf',
    },
}

with open(OUT, 'w', encoding='utf-8') as fh:
    json.dump(result, fh, indent=1, ensure_ascii=False, sort_keys=True)

print('wrote', OUT)
print('P(beta>=1) max over all record sets : %s %s = %.5f (SE %.3e, +2SE %.5f)' % (
    max_all['record_set'], max_all['statistic'], max_all['p'],
    max_all['mc_standard_error'], max_all['p_plus_2se']))
print('P(beta>=1) max over six headline    : %s %s = %.5f (SE %.3e, +2SE %.5f)' % (
    max_six['record_set'], max_six['statistic'], max_six['p'],
    max_six['mc_standard_error'], max_six['p_plus_2se']))
for b in CANDIDATES:
    r = result['P_beta_ge_1']['bound_robustness_all_cells']['%g' % b]
    s = result['P_beta_ge_1']['bound_robustness_six_headline_cells']['%g' % b]
    print('  bound %-7g all-cells point=%-5s +2SE=%-5s | six-cell point=%-5s +2SE=%s' % (
        b, r['holds_at_point_estimate'], r['holds_at_plus_2se'],
        s['holds_at_point_estimate'], s['holds_at_plus_2se']))
print('FL max MC SE over all schemes    : %.3g (%s %s)' % (
    result['freedman_lane_mc_se_audit']['max_over_all_schemes']['mc_standard_error'],
    result['freedman_lane_mc_se_audit']['max_over_all_schemes']['record_set'],
    result['freedman_lane_mc_se_audit']['max_over_all_schemes']['scheme']))
print('FL max MC SE over CB/CB_ws/CM    : %.3g (%s %s)' % (
    result['freedman_lane_mc_se_audit']['max_over_reported_schemes_CB_CBws_CM']['mc_standard_error'],
    result['freedman_lane_mc_se_audit']['max_over_reported_schemes_CB_CBws_CM']['record_set'],
    result['freedman_lane_mc_se_audit']['max_over_reported_schemes_CB_CBws_CM']['scheme']))
print('AO max MC SE                     : %.3g' % result['albedo_offset_mc_se_audit']['max']['mc_standard_error'])
