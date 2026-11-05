"""Walk-forward backtest of the defense-log signals.

For every 2026 row (player P vs defense D on date t) we rebuild the archetype, position and comp
signals from rows dated BEFORE t only, then check: when the signal leaned over/under, did the
result beat the price (hit rate vs implied chance)? Archetype labels are this season's, a small
look-ahead noted in the report.
"""
import os, json, collections, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import DATA_DIR, build_rows, build_archetypes, defense_tiers, build_comps, excess_lean, lean_from, under_odds_model

def bt(sport):
    d = json.load(open(os.path.join(DATA_DIR, f'{sport}.json')))
    rows = build_rows(d); arch = build_archetypes(d, sport)
    by = collections.defaultdict(list)
    for r in rows:
        by[(r['def_'], r['pos'], r['mkt'])].append(r)
    for v in by.values():
        v.sort(key=lambda r: r['dk'])
    season = max(r['season'] for r in rows)
    # rebuild comps every two weeks of the season from past data only
    dates = sorted({r['dk'] for r in rows if r['season'] == season})
    checkpoints = dates[::10] if sport == 'nba' else dates[::14] or dates
    comp_cache = {}
    def comps_at(dk):
        cp = max([c for c in checkpoints if c <= dk], default=None)
        if cp is None:
            return {}
        if cp not in comp_cache:
            comp_cache[cp] = build_comps(rows, arch, defense_tiers(rows, cp), cp)
        return comp_cache[cp]
    out = []
    for r in rows:
        if r['season'] != season:
            continue
        hist = [x for x in by[(r['def_'], r['pos'], r['mkt'])] if x['dk'] < r['dk'] and x['player'] != r['player']]
        a = arch.get(r['player'])
        arch_rows = [x for x in hist if a and arch.get(x['player']) == a]
        cl = comps_at(r['dk']).get(r['player'], [])
        comp_rows = [(x, s) for q, s, _ in cl for x in hist if x['player'] == q]
        if comp_rows:
            ws = sum(s for _, s in comp_rows)
            c = (lean_from(sum(s * (x['hit'] - x['imp']) for x, s in comp_rows) / ws, len(comp_rows), 3), len(comp_rows))
        else:
            c = (0.0, 0)
        out.append(dict(r=r, arch=excess_lean(arch_rows, 4), pos=excess_lean(hist), comps=c, na=len(arch_rows)))
    return out

def bucket_report(res, key, label):
    lines = [f'  {label}']
    for lo, hi, name in [(-9, -0.3, 'strong under'), (-0.3, -0.1, 'lean under'), (-0.1, 0.1, 'no signal'),
                         (0.1, 0.3, 'lean over'), (0.3, 9, 'strong over')]:
        sel = [x for x in res if x[key][1] and lo <= x[key][0] < hi]
        if not sel:
            continue
        hr = statistics.mean(x['r']['hit'] for x in sel); imp = statistics.mean(x['r']['imp'] for x in sel)
        lines.append(f'    {name:13s} n={len(sel):5d}  over hit {hr:.1%}  priced at {imp:.1%}  beat price by {hr-imp:+.1%}')
    return '\n'.join(lines)

def roi_report(res, sport):
    """Flat 1-unit bets: overs at the logged over price; unders at the modeled under price for that over price."""
    lines = []
    for name, f in [('arch+pos+comps, overs', lambda x: max(0, (3 * x['arch'][0] + 1.5 * x['pos'][0] + 2 * x['comps'][0]) / 6.5)
                        if x['na'] >= 3 else 0),
                    ('arch+pos+comps, unders', lambda x: min(0, (3 * x['arch'][0] + 1.5 * x['pos'][0] + 2 * x['comps'][0]) / 6.5)
                        if x['na'] >= 3 else 0),
                    ('comps-backed overs', lambda x: (3 * x['arch'][0] + 1.5 * x['pos'][0] + 2 * x['comps'][0]) / 6.5
                        if x['na'] >= 3 and x['comps'][1] >= 2 and x['comps'][0] >= 0.3 else 0),
                    ('archetype only', lambda x: x['arch'][0] if x['na'] >= 3 else 0),
                    ('arch + position + comps', lambda x: (3 * x['arch'][0] + 1.5 * x['pos'][0] + 2 * x['comps'][0]) / 6.5
                        if x['na'] >= 3 else 0)]:
        for thr in (0.15, 0.25, 0.35):
            n = units = 0; wins = 0
            for x in res:
                s = f(x); r = x['r']
                if s >= thr and r['odds'] <= 200:
                    n += 1; win = r['hit']; wins += win
                    units += (r['odds'] / 100 if r['odds'] > 0 else 100 / -r['odds']) if win else -1
                elif s <= -thr and -250 <= r['odds'] <= 150:
                    u = under_odds_model(r['odds'])
                    n += 1; win = 1 - r['hit'] if r['res'] != r['line'] else 0; wins += win
                    units += (u / 100 if u > 0 else 100 / -u) if win else -1
            if n:
                lines.append(f'  {name:24s} threshold {thr:.2f}: {n:5d} bets, win {wins/n:.1%}, ROI {units/n:+.1%}')
    return '\n'.join(lines)

if __name__ == '__main__':
    report = []
    for sport in ['nfl', 'nba']:
        res = bt(sport)
        report.append(f'===== {sport.upper()}  ({len(res)} props replayed, this season)')
        report.append(bucket_report([x for x in res if x['na'] >= 3], 'arch', 'Archetype vs defense (3+ prior same-type games)'))
        report.append(bucket_report(res, 'pos', 'Position vs defense'))
        report.append(bucket_report(res, 'comps', 'Comps vs defense'))
        report.append(roi_report(res, sport))
    txt = '\n'.join(report); print(txt)
    open(os.path.join(DATA_DIR, 'backtest_report.txt'), 'w').write(txt)
