"""Betting agent v1 — reasons over props.cash data the way the playbook describes.

Every signal asks the same question: does this beat the PRICE, not a coin flip?
  arch    same-archetype players vs this defense: hit rate minus the odds' implied chance
  pos     same-position players vs this defense: same measure
  comps   players who perform like this one vs similar defenses, against this defense
  form    the player's own hit rate vs tonight's implied chance
  proj    props.cash projection vs the line
  dvp     defense-vs-position rank for the stat
Risks: injury status, teammates out, blowout spread, heavy juice, thin samples.
"""
import os
import re, json, math, statistics, collections, sys
DATA_DIR = os.environ.get('AGENT_DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))

from under_prices import fit_model, under_imp_from_over, raw_imp, to_american, live_unders

# How books price the under given the over, learned from every both-sides book quote in the data
# (NFL + NBA pooled; hold is ~6.5-7.5%). Used to strip the vig and to price unders in the backtest.
_MODEL = []
for _sp in ('nfl', 'nba'):
    try:
        _MODEL += fit_model(json.load(open(os.path.join(DATA_DIR, f'{_sp}.json'))))
    except FileNotFoundError:
        pass
_bins = collections.defaultdict(list)
for _x, _y, _n in _MODEL:
    _bins[_x].append((_y, _n))
UNDER_MODEL = sorted((x, sum(y * n for y, n in v) / sum(n for _, n in v), sum(n for _, n in v)) for x, v in _bins.items())

def under_odds_model(over_odds):
    return to_american(under_imp_from_over(UNDER_MODEL, over_odds))

# ---------- helpers ----------
def norm(name):
    n = re.sub(r"[.'’]", "", name.lower())
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n)
    return re.sub(r"\s+", " ", n).strip()

def implied(odds):
    """fair (no-vig) chance of the OVER, using the modeled under price as the other side"""
    if odds is None:
        return None
    o = raw_imp(odds)
    return o / (o + under_imp_from_over(UNDER_MODEL, odds))

def date_key(md, season):
    m, d = (int(x) for x in md.split('/'))
    return (season if m >= 7 else season + 1, m, d)

def lean_from(excess, n, k=6):
    """excess = hit rate minus implied chance; scaled to [-1,1], shrunk for small samples"""
    if not n:
        return 0.0
    return max(-1.0, min(1.0, excess / 0.25)) * (n / (n + k))

def margin(res, line):
    return (res - line) / max(abs(line), 1.0)

# ---------- archetypes ----------
NBA_PT = ['PRBallHandler', 'Isolation', 'Transition', 'Spotup', 'OffScreen', 'Handoff',
          'PRRollman', 'Cut', 'OffRebound', 'Postup', 'Misc']

def build_archetypes(d, sport):
    """normalized name -> archetype. props.cash roles first, then a classifier for the rest."""
    arch = {norm(n): r['type'] for n, r in d['roles'].items() if r.get('type')}
    if sport == 'nba':
        # learn each props.cash type's play-type mix, then classify unlabeled players by nearest mix
        cent = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in d['roles'].values():
            if not r.get('type') or not r.get('mix'):
                continue
            v = dict(r['mix'])
            for k in NBA_PT:
                cent[(r['pos'], r['type'])][k].append(v.get(k, 0))
        cent = {k: {pt: statistics.mean(v) for pt, v in vv.items()} for k, vv in cent.items()}
        for name, p in d['players'].items():
            if norm(name) in arch or not p.get('playTypes'):
                continue
            v = {t['type']: t['share'] for t in p['playTypes']}
            cands = [(k, c) for k, c in cent.items() if k[0] == p['pos']] or list(cent.items())
            best = min(cands, key=lambda kc: sum((v.get(pt, 0) - kc[1].get(pt, 0)) ** 2 for pt in NBA_PT))
            arch[norm(name)] = best[0][1]
    if sport == 'nfl':
        lines = collections.defaultdict(list); posof = {}
        for tv in d['teams'].values():
            for pos, mm in tv['vsPosLog'].items():
                for mkt, rows in mm.items():
                    for r in rows:
                        lines[(norm(r[0]), mkt)].append(r[2]); posof[norm(r[0])] = pos
        for p in d['props']:
            lines[(norm(p['player']), p['market'])].append(p['line']); posof.setdefault(norm(p['player']), p['pos'])
        med = lambda p, m: statistics.median(lines[(p, m)]) if lines.get((p, m)) else None
        for p, pos in posof.items():
            if p in arch:
                continue
            if pos == 'TE':
                ry = med(p, 'Receiving Yards')
                if ry is not None:
                    arch[p] = 'Receiving TE' if ry >= 38 else ('Secondary TE' if ry >= 22 else 'Blocking TE')
            elif pos == 'QB':
                rush, py = med(p, 'Rush Yards'), med(p, 'Pass Yards')
                if rush is not None and rush >= 25:
                    arch[p] = 'Dual-threat QB'
                elif py is not None:
                    arch[p] = 'Pocket passer' if py >= 235 else 'Game manager'
    return arch

# ---------- defense logs ----------
def build_rows(d):
    out = []
    for dteam, tv in d['teams'].items():
        for pos, mm in tv['vsPosLog'].items():
            for mkt, rows in mm.items():
                for r in rows:
                    name, res, line, odds, team, md, season, hit = r
                    if res is None or line is None or odds is None:
                        continue
                    out.append(dict(def_=dteam, pos=pos, mkt=mkt, player=norm(name), disp=name, res=res,
                                    line=line, odds=odds, imp=implied(odds), team=team,
                                    dk=date_key(md, season), season=season, hit=1 if res > line else 0))
    return out

def defense_tiers(rows, before=None):
    agg = collections.defaultdict(list)
    for r in rows:
        if before and r['dk'] >= before:
            continue
        agg[(r['def_'], r['pos'], r['mkt'])].append(r['hit'] - r['imp'])
    by_pm = collections.defaultdict(list)
    for (dt, pos, mkt), ms in agg.items():
        if len(ms) >= 3:
            by_pm[(pos, mkt)].append((statistics.mean(ms), dt))
    tiers = {}
    for (pos, mkt), lst in by_pm.items():
        lst.sort(); n = len(lst)
        for i, (_, dt) in enumerate(lst):
            tiers[(dt, pos, mkt)] = 'tough' if i < n / 3 else ('mid' if i < 2 * n / 3 else 'soft')
    return tiers

def build_comps(rows, arch, tiers, before=None):
    """comps = same archetype, similar results vs line against the same TYPES of defense"""
    prof = collections.defaultdict(lambda: collections.defaultdict(list)); pos_of = {}
    for r in rows:
        if before and r['dk'] >= before:
            continue
        t = tiers.get((r['def_'], r['pos'], r['mkt']))
        if t:
            prof[r['player']][(r['mkt'], t)].append(r['hit'] - r['imp'])
            pos_of[r['player']] = r['pos']
    vec = {p: {k: statistics.mean(v) for k, v in pv.items()} for p, pv in prof.items()}
    groups = collections.defaultdict(list)
    for p in vec:
        if p in arch:
            groups[(pos_of[p], arch[p])].append(p)
    comps = {}
    for members in groups.values():
        for p in members:
            scored = []
            for q in members:
                if q == p:
                    continue
                shared = set(vec[p]) & set(vec[q])
                if len(shared) < 4:
                    continue
                dist = math.sqrt(sum((vec[p][k] - vec[q][k]) ** 2 for k in shared) / len(shared))
                scored.append((q, round(1 / (1 + 3 * dist), 3), len(shared)))
            scored.sort(key=lambda x: -x[1])
            comps[p] = scored[:6]
    return comps

def summarize(sel):
    if not sel:
        return dict(n=0, overs=0, rate=None, imp=None, med=None)
    n = len(sel)
    return dict(n=n, overs=sum(r['hit'] for r in sel), rate=round(sum(r['hit'] for r in sel) / n, 3),
                imp=round(statistics.mean(r['imp'] for r in sel), 3),
                med=round(statistics.median(r['res'] - r['line'] for r in sel), 1))

def excess_lean(sel, k=6):
    """season-weighted (this season 1.0, last 0.5) hit rate minus implied chance"""
    if not sel:
        return 0.0, 0
    top = max(r['season'] for r in sel)
    w = [1.0 if r['season'] == top else 0.5 for r in sel]
    ex = sum(wi * (r['hit'] - r['imp']) for wi, r in zip(w, sel)) / sum(w)
    return lean_from(ex, sum(w), k), sum(w)

# ---------- the agent ----------
WEIGHTS = dict(arch=3.0, pos=1.5, comps=2.0, form=1.0, proj=1.5, dvp=0.75)
SKIP_MKTS = {'Sacks', 'Field Goals', 'Kicking Points', 'Extra Points', 'Interceptions', 'Triple Double'}
TD_MKTS = {'Anytime TD', '1Q Anytime TD', 'Double Double'}

def signals_for(sport, prop, rows_by_dpm, arch, comps, before=None):
    name = norm(prop['player']); pos = 'RB' if prop['pos'] == 'FB' else prop['pos']
    mkt, opp = prop['market'], prop['opp']
    pos_rows = [r for r in rows_by_dpm.get((opp, pos, mkt), []) if r['player'] != name
                and (before is None or r['dk'] < before)]
    a = arch.get(name)
    arch_rows = [r for r in pos_rows if a and arch.get(r['player']) == a]
    sig = {'pos': excess_lean(pos_rows), 'arch': excess_lean(arch_rows, 4)}
    cl = comps.get(name, [])
    comp_rows = [(r, s) for q, s, _ in cl for r in pos_rows if r['player'] == q]
    if comp_rows:
        ws = sum(s for _, s in comp_rows)
        ex = sum(s * (r['hit'] - r['imp']) for r, s in comp_rows) / ws
        sig['comps'] = (lean_from(ex, len(comp_rows), 3), len(comp_rows))
    else:
        sig['comps'] = (0.0, 0)
    return sig, a, pos_rows, arch_rows, comp_rows, cl

def evaluate(d, sport, rows, arch, comps):
    games = {g['id']: g for g in d['games']}
    players = d['players']
    live = live_unders(d)
    by = collections.defaultdict(list)
    for r in rows:
        by[(r['def_'], r['pos'], r['mkt'])].append(r)
    out = []
    for p in d['props']:
        mkt, line, odds = p['market'], p['line'], p.get('odds')
        if mkt in SKIP_MKTS or odds is None:
            continue
        imp_over = implied(odds)
        sig, a, pos_rows, arch_rows, comp_rows, cl = signals_for(sport, p, by, arch, comps)
        # own form vs tonight's price
        h = p.get('hit') or {}; n26 = (p.get('hitN') or {}).get('2026', 0) or 0
        parts = [(h.get('L10'), 10), (h.get('2026'), n26)]
        parts = [(r / 100, n) for r, n in parts if r is not None and n]
        if parts:
            rate = sum(r * n for r, n in parts) / sum(n for _, n in parts)
            sig['form'] = (lean_from(rate - imp_over, min(20, sum(n for _, n in parts)), 10), 1)
        else:
            sig['form'] = (0.0, 0)
        if mkt not in TD_MKTS and p.get('projection') is not None:
            sig['proj'] = (max(-1, min(1, (p['projection'] - line) / max(line, 1) * 5)), 1)
        elif p.get('tdProb') is not None:
            sig['proj'] = (max(-1, min(1, (p['tdProb'] / 100 - imp_over) * 4)), 1)
        else:
            sig['proj'] = (0.0, 0)
        if p.get('dvp'):
            n_t = 30 if sport == 'nba' else 32
            sig['dvp'] = ((p['dvp'] - (n_t + 1) / 2) / ((n_t - 1) / 2), 1)
        else:
            sig['dvp'] = (0.0, 0)

        maxs = sum(WEIGHTS[k] for k, v in sig.items() if v[1])
        score = sum(WEIGHTS[k] * v[0] for k, v in sig.items()) / maxs if maxs else 0
        side = 'Over' if score >= 0 else 'Under'
        sgn = 1 if score >= 0 else -1
        agree = [k for k, v in sig.items() if v[1] and v[0] * sgn > 0.05]
        against = [k for k, v in sig.items() if v[1] and v[0] * sgn < -0.15]

        g = games.get(p['game'], {}); pl = players.get(p['player'], {})
        risks = []
        for rr in pl.get('roster') or []:
            if norm(rr[0]) == norm(p['player']) and rr[2] in ('Questionable', 'Doubtful', 'Out'):
                risks.append(f"{p['player']} listed {rr[2]} ({rr[3]})")
        outs = [f"{rr[0]} ({rr[1]}, {rr[2]})" for rr in (pl.get('roster') or [])
                if norm(rr[0]) != norm(p['player']) and rr[2] in ('Out', 'Doubtful')]
        if g.get('spread') is not None and abs(g['spread']) >= (10 if sport == 'nba' else 9.5):
            risks.append(f"big spread ({g['fav']} {g['spread']}): blowout / bench risk")
        if sig['arch'][1] < 3:
            risks.append('thin archetype sample vs this defense')

        # Prices: overs at the scraped over price; unders at the live book consensus when props.cash
        # lists one at this line, else the price modeled from the over (flagged as estimated).
        lu = live.get((p['player'], mkt, line))
        if lu:
            u_price, u_best, u_book, u_est = lu['consensus'], lu['best'], lu['best_book'], False
        else:
            u_price = under_odds_model(odds); u_best, u_book, u_est = u_price, None, True
        if side == 'Over':
            priceable = odds <= (400 if mkt in TD_MKTS else 200)
        else:
            priceable = mkt not in TD_MKTS and -250 <= odds <= 150 and raw_imp(u_price) <= 0.615  # under no worse than -160
        # Gate tuned to the walk-forward replay with real hold (backtest_report.txt):
        #   NFL unders: core <= -0.35 -> +5.7% on 547 bets.
        #   Overs: only when comps agree; core >= 0.25 -> +11.1% (179), >= 0.35 -> +14.8% (124).
        #   NBA: unders lost badly (-44.8% on 48 bets) -> none; comps overs capped at 'Watch'.
        core = (3 * sig['arch'][0] + 1.5 * sig['pos'][0] + 2 * sig['comps'][0]) / 6.5 \
            if sig['arch'][1] >= 3 else 0.0
        comps_agree = sig['comps'][1] >= 2 and sig['comps'][0] * sgn >= 0.3
        conf = None
        if priceable and len(against) <= 1 and score * sgn >= 0.15:
            if side == 'Over' and comps_agree and core >= 0.25:
                conf = 'Watch' if sport == 'nba' else ('Strong' if core >= 0.35 and not against else 'Lean')
            elif side == 'Under' and sport == 'nfl' and core <= -0.35:
                conf = 'Strong' if (comps_agree and not against and not u_est) else 'Lean'
        if any(('listed Out' in r) or ('listed Doubtful' in r) for r in risks):
            conf = None
        if conf == 'Strong' and any('listed Questionable' in r for r in risks):
            conf = 'Lean'

        out.append(dict(
            sport=sport, player=p['player'], team=p['team'], pos=p['pos'], arch=a, opp=p['opp'],
            game=p['game'], kickoff=g.get('kickoff'), market=mkt, line=line, odds=odds if side == 'Over' else u_price, side=side,
            best_odds=odds if side == 'Over' else u_best, best_book=u_book if side == 'Under' else None,
            price_est=(side == 'Under' and u_est),
            score=round(score, 3), core=round(core, 3), conf=conf, agree=agree, against=against,
            signals={k: [round(v[0], 2), round(v[1], 1)] for k, v in sig.items()},
            arch_sum=summarize(arch_rows), pos_sum=summarize(pos_rows),
            comps=[(q, s) for q, s, _ in cl[:4]],
            comp_games=[(r['disp'], r['line'], r['res'], 'O' if r['hit'] else 'U') for r, _ in comp_rows][:6],
            arch_games=[(r['disp'], r['line'], r['res'], f"{r['dk'][1]}/{r['dk'][2]}/{str(r['dk'][0])[2:]}")
                        for r in sorted(arch_rows, key=lambda r: r['dk'], reverse=True)][:8],
            form=dict(L10=h.get('L10'), season=h.get('2026'), n=n26, l10avg=p.get('l10avg')),
            projection=p.get('projection'), tdProb=p.get('tdProb'), dvp=p.get('dvp'),
            total=g.get('total'), spread=g.get('spread'), fav=g.get('fav'),
            teammates_out=outs[:5], risks=risks))
    return out

def run(sport, path):
    d = json.load(open(path))
    rows = build_rows(d)
    arch = build_archetypes(d, sport)
    tiers = defense_tiers(rows)
    comps = build_comps(rows, arch, tiers)
    return d, rows, arch, tiers, comps, evaluate(d, sport, rows, arch, comps)

if __name__ == '__main__':
    for sport in sys.argv[1:] or ['nfl', 'nba']:
        d, rows, arch, tiers, comps, res = run(sport, os.path.join(DATA_DIR, f'{sport}.json'))
        plays = [r for r in res if r['conf']]
        cover = sum(1 for p in d['props'] if norm(p['player']) in arch) / len(d['props'])
        print(sport, 'props', len(res), 'plays', len(plays), collections.Counter(r['conf'] for r in plays),
              f'archetype coverage {cover:.0%}', 'players with comps', sum(1 for v in comps.values() if v))
        json.dump(res, open(os.path.join(DATA_DIR, f'{sport}_eval.json'), 'w'), indent=1)
