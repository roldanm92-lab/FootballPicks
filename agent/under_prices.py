"""Real under prices.

1. Live: for tonight's props, the consensus (median) and best under price across sportsbooks, from
   players[*].books[market].rows = [book, over, under].
2. Model: how books price the under given the over (median under implied chance per bin of over implied
   chance, by sport), learned from every book row that quotes both sides. Used by the backtest, because the
   defense logs only keep the over price.
"""
import os
import json, statistics, collections, bisect
DATA_DIR = os.environ.get('AGENT_DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))

DFS_OR_EXCHANGE = {'PrizePicks', 'Underdog', 'Sleeper', 'Dabble', 'Betr', 'ProphetX', 'Novig',
                   'Kalshi', 'Polymarket', 'Sporttrade'}

def raw_imp(o):
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)

def to_american(p):
    p = min(max(p, 0.01), 0.99)
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)

def book_rows(d):
    for name, pl in d['players'].items():
        for mkt, b in (pl.get('books') or {}).items():
            for book, ov, un in b['rows']:
                if book in DFS_OR_EXCHANGE or ov is None or un is None:
                    continue
                if abs(ov) > 2000 or abs(un) > 2000:
                    continue
                yield name, mkt, b['line'], book, ov, un

def fit_model(d):
    """bins of over raw implied (0.05 wide) -> median under raw implied"""
    bins = collections.defaultdict(list)
    for *_, ov, un in book_rows(d):
        bins[round(raw_imp(ov) * 20) / 20].append(raw_imp(un))
    model = sorted((k, statistics.median(v), len(v)) for k, v in bins.items() if len(v) >= 8)
    return model

def under_imp_from_over(model, over_odds):
    x = raw_imp(over_odds)
    xs = [m[0] for m in model]
    i = bisect.bisect_left(xs, x)
    if i <= 0:
        return model[0][1] + (model[0][0] - x)
    if i >= len(xs):
        return model[-1][1] - (x - model[-1][0])
    (x0, y0, _), (x1, y1, _) = model[i - 1], model[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

def live_unders(d):
    """(player, market, line) -> dict(consensus, best, best_book, n_books)"""
    out = {}
    for name, pl in d['players'].items():
        for mkt, b in (pl.get('books') or {}).items():
            us = [(un, book) for book, ov, un in b['rows'] if un is not None and book not in DFS_OR_EXCHANGE
                  and abs(un) <= 2000]
            if not us:
                continue
            cons = to_american(statistics.median(raw_imp(u) for u, _ in us))
            best = max(us, key=lambda x: -raw_imp(x[0]))
            out[(name, mkt, b['line'])] = dict(consensus=cons, best=best[0], best_book=best[1], n_books=len(us))
    return out

if __name__ == '__main__':
    for sp in ['nfl', 'nba']:
        d = json.load(open(os.path.join(DATA_DIR, f'{sp}.json')))
        m = fit_model(d)
        print(sp, 'rows', sum(x[2] for x in m))
        for x, y, n in m:
            print(f'  over priced {x:.2f} ({to_american(x):+5d}) -> under priced {y:.2f} ({to_american(y):+5d})  hold {x+y-1:+.1%}  n={n}')
