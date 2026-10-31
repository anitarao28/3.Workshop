"""Value x risk matrix, budget allocation scenarios and the 2011 reveal."""
import pandas as pd, numpy as np, json
c = pd.read_pickle('cust_clv.pkl')
s = pd.read_pickle('pred_scores.pkl')
c = c.join(s[['p_lr', 'churn_6m', 'risk_decile']])
assert c.p_lr.notna().all()
out = {}
COST, SAVE, MARGIN = 25, 0.20, 0.30
out['assumptions'] = dict(cost_per_contact=COST, save_rate=SAVE, margin=MARGIN)

hv = c.clv_12m >= c.clv_12m.median(); hr = c.p_lr >= c.p_lr.median()
c['quadrant'] = np.select([hv & hr, hv & ~hr, ~hv & hr, ~hv & ~hr], ['High value / High risk', 'High value / Low risk', 'Low value / High risk', 'Low value / Low risk'], default='')
q = c.groupby('quadrant').agg(n=('clv_12m', 'size'), clv_mean=('clv_12m', 'mean'), clv_total=('clv_12m', 'sum'), p_churn=('p_lr', 'mean'),
                              last_year_rev=('monetary', 'sum'), actual_churn=('churn_6m', 'mean'), actual_12m_rev=('hold_revenue', 'sum'), buy12=('hold_bought', 'mean'))
q['value_at_risk_model'] = c.assign(v=c.p_lr * c.clv_12m).groupby('quadrant').v.sum()
q['share_customers'] = q.n / len(c)
out['quadrants'] = q.round(3).reset_index().to_dict('records')
out['segment_by_quadrant'] = pd.crosstab(c.segment, c.quadrant).to_dict()

c['exp_gain'] = c.p_lr * SAVE * c.clv_12m * MARGIN

def evaluate(sel, label):
    t = c[sel]; n = len(t); cost = n * COST
    planned = t.exp_gain.sum()
    lapsers = t[t.churn_6m == 1]
    # realised (same assumptions, but actual lapse outcome replaces the model's probability)
    realised = SAVE * MARGIN * lapsers.clv_12m.sum()
    lost12 = t[t.hold_bought == 0]
    return dict(strategy=label, n=n, cost=cost, planned_gain=float(planned), planned_roi=float(planned / cost - 1),
                avg_clv=float(t.clv_12m.mean()), avg_p_churn=float(t.p_lr.mean()),
                actual_lapsers=int(len(lapsers)), pct_lapsed=float(t.churn_6m.mean()), pct_bought_anyway=float(1 - t.churn_6m.mean()),
                share_all_lapser_value=float(lapsers.clv_12m.sum() / c.loc[c.churn_6m == 1, 'clv_12m'].sum()),
                pct_lost_full_year=float(len(lost12) / n),
                realised_gain=float(realised), realised_roi=float(realised / cost - 1))

res = []
for budget in [50000, 30000]:
    N = budget // COST
    strategies = {
        'Protect top CLV': c.clv_12m.rank(ascending=False, method='first') <= N,
        'Highest churn risk': c.p_lr.rank(ascending=False, method='first') <= N,
        'Value at risk (risk x CLV)': c.exp_gain.rank(ascending=False, method='first') <= N,
        'Random / spread evenly': c.index.isin(c.sample(N, random_state=1).index),
    }
    for k, sel in strategies.items():
        r = evaluate(sel, k); r['budget'] = budget; res.append(r)
    # how many contacts are worth it at all (planned gain > cost)
    out[f'n_worth_contacting_{budget}'] = int((c.exp_gain > COST).sum())
out['strategies'] = res
# segment mix of each strategy at 50k
N = 2000
mix = {}
for k, col, asc in [('Protect top CLV', 'clv_12m', False), ('Highest churn risk', 'p_lr', False), ('Value at risk (risk x CLV)', 'exp_gain', False)]:
    sel = c[col].rank(ascending=asc, method='first') <= N
    mix[k] = c[sel].segment.value_counts().to_dict()
out['strategy_segment_mix'] = mix
out['n_profitable_contacts_planned'] = int((c.exp_gain > COST).sum())
out['total_lapser_last_year_rev'] = float(c.loc[c.churn_6m == 1, 'monetary'].sum())
out['n_lapsers'] = int(c.churn_6m.sum())
# sensitivity of planned ROI for VaR strategy to save rate
sens = []
sel = c.exp_gain.rank(ascending=False, method='first') <= 2000
for sr in [0.05, 0.10, 0.20, 0.30]:
    g = (c[sel].p_lr * sr * c[sel].clv_12m * MARGIN).sum()
    sens.append(dict(save_rate=sr, planned_gain=float(g), roi=float(g / 50000 - 1)))
out['save_rate_sensitivity'] = sens
c.to_pickle('cust_final.pkl')
json.dump(out, open('facts_alloc.json', 'w'), indent=1, default=float)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
for k, v in out.items():
    print('==', k); print(pd.DataFrame(v) if isinstance(v, list) and v and isinstance(v[0], dict) else v)
