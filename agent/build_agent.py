"""Build agent.html from the data already embedded in index.html.

Run from anywhere:  python agent/build_agent.py
  1. pulls window.EMBEDDED_PROPS (NFL) and window.EMBEDDED_NBA_PROPS out of index.html
  2. runs the agent (agent.py) and the walk-forward replay (backtest.py)
  3. writes agent.html next to index.html
  4. makes sure index.html has the "Agent" tab (the PC pipeline rebuilds index.html, so this re-adds it)
  5. appends tonight's plays to agent/plays_log.json (for grading later)
Needs only Python 3 standard library.
"""
import os, re, json, sys, tempfile, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INDEX = os.path.join(ROOT, 'index.html')
OUT = os.path.join(ROOT, 'agent.html')
LOG = os.path.join(HERE, 'plays_log.json')
TAB = ('      <a class="sporttab" href="agent.html" id="agentTab" '
       'title="The betting agent\'s play card" style="text-decoration:none">Agent</a>')

def extract(html, name):
    m = re.search(r'window\.' + name + r'\s*=\s*(.*?);\s*\n', html)
    if not m:
        sys.exit(f'{name} not found in index.html')
    return json.loads(m.group(1))

def ensure_tab(html):
    if 'id="agentTab"' in html:
        return html, False
    anchor = re.search(r'\n[^\n]*id="dailyTab"[^\n]*\n', html)
    if anchor:
        i = anchor.end()
    else:  # no Action Daily tab: put it after the last sport tab
        m = list(re.finditer(r'\n[^\n]*class="sporttab[^"]*" data-sport="[^"]+"[^\n]*\n', html))
        if not m:
            print('warning: tab row not found; Agent tab not added')
            return html, False
        i = m[-1].end()
    return html[:i] + TAB + '\n' + html[i:], True

def main():
    html = open(INDEX, encoding='utf-8').read()
    data_dir = tempfile.mkdtemp(prefix='agent_data_')
    os.environ['AGENT_DATA_DIR'] = data_dir
    sources = {'nfl': 'EMBEDDED_PROPS', 'nba': 'EMBEDDED_NBA_PROPS'}
    scraped = {}
    for sp, name in sources.items():
        d = extract(html, name)
        scraped[sp] = d.get('scraped_at')
        json.dump(d, open(os.path.join(data_dir, f'{sp}.json'), 'w'))

    sys.path.insert(0, HERE)
    import agent, backtest  # imported after the data folder is set (agent fits the price model on import)
    from agent import under_odds_model

    plays = []
    for sp in sources:
        *_, res = agent.run(sp, os.path.join(data_dir, f'{sp}.json'))
        plays += [r for r in res if r['conf']]

    # replay summary for the trust box
    summary = []
    pay = lambda o: o / 100 if o > 0 else 100 / -o
    core = lambda x: (3 * x['arch'][0] + 1.5 * x['pos'][0] + 2 * x['comps'][0]) / 6.5 if x['na'] >= 3 else 0
    def roi(sel, kind):
        n = u = 0
        for x in sel:
            c, r = core(x), x['r']
            if kind == 'over' and c >= 0.25 and r['odds'] <= 200 and x['comps'][1] >= 2 and x['comps'][0] >= 0.3:
                win, o = r['hit'], r['odds']
            elif kind == 'under' and c <= -0.35 and -250 <= r['odds'] <= 150:
                win, o = (1 - r['hit']) if r['res'] != r['line'] else 0, under_odds_model(r['odds'])
            else:
                continue
            n += 1; u += pay(o) if win else -1
        return n, (u / n if n else 0)
    for sp in sources:
        res = backtest.bt(sp)
        last = max(x['r']['dk'] for x in res)
        cut = (datetime.date(*last) - datetime.timedelta(days=21))
        recent = [x for x in res if datetime.date(*x['r']['dk']) > cut]
        for kind, label in (('over', 'Comps-backed overs'), ('under', 'Unders')):
            if sp == 'nba' and kind == 'under':
                n, r = roi(res, kind)
                if n:
                    summary.append(dict(label='NBA unders (not played)', n=n, roi=r))
                continue
            n, r = roi(res, kind)
            if n:
                summary.append(dict(label=f'{sp.upper()} {label.lower()}, season', n=n, roi=r))
            n, r = roi(recent, kind)
            if n and sp == 'nfl':
                summary.append(dict(label=f'{sp.upper()} {label.lower()}, last 3 weeks', n=n, roi=r))

    tpl = open(os.path.join(HERE, 'card_template.html'), encoding='utf-8').read()
    payload = json.dumps({'plays': plays, 'scraped': scraped, 'summary': summary,
                          'built': datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})
    page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + tpl.replace('__DATA__', payload.replace('</', '<\\/')) + '\n</html>\n')
    open(OUT, 'w', encoding='utf-8').write(page)

    html2, added = ensure_tab(html)
    if added:
        open(INDEX, 'w', encoding='utf-8').write(html2)

    # play log (one entry per play per day; grading comes later)
    try:
        log = json.load(open(LOG))
    except (FileNotFoundError, json.JSONDecodeError):
        log = []
    seen = {(e['date'], e['player'], e['market'], e['line'], e['side']) for e in log}
    today = datetime.date.today().isoformat()
    for p in plays:
        k = (today, p['player'], p['market'], p['line'], p['side'])
        if k not in seen:
            log.append(dict(date=today, sport=p['sport'], game=p['game'], kickoff=p['kickoff'], player=p['player'],
                            market=p['market'], line=p['line'], side=p['side'], odds=p['odds'],
                            conf=p['conf'], core=p['core'], result=None))
    json.dump(log, open(LOG, 'w'), indent=1)

    print(f'agent.html written: {len(plays)} plays; data NFL {scraped["nfl"]}, NBA {scraped["nba"]}; '
          f'tab {"added" if added else "already there"}; log {len(log)} entries')
    for s in summary:
        print(f'  {s["label"]}: {s["n"]} bets, {s["roi"]:+.1%}')

if __name__ == '__main__':
    main()
