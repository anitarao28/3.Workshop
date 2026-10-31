"""CLV: BG/NBD (weekly) for purchase counts x shrunk spend per purchase; validated on 2011 holdout."""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from lifetimes import BetaGeoFitter

cust = pd.read_pickle('cust_seg.pkl')
raw = pd.read_pickle('raw.pkl').rename(columns={'Customer ID': 'CustomerID'})
h = raw[(raw.InvoiceDate >= '2010-12-01') & (raw.InvoiceDate < '2011-12-01') & raw.CustomerID.notna()
        & ~raw.Invoice.str.startswith('C') & (raw.Quantity > 0) & (raw.Price > 0)].drop_duplicates()
pdays = h.groupby('CustomerID').InvoiceDate.apply(lambda s: s.dt.normalize().nunique())
pdays.index = pdays.index.astype(int)
cust['hold_pdays'] = pdays.reindex(cust.index).fillna(0)

out = {}
fr, rc, T = cust.lt_frequency, cust.lt_recency / 7, cust.lt_T / 7
bg = BetaGeoFitter().fit(fr, rc, T)
out['bgnbd_params_weekly'] = bg.params_.to_dict()
cust['exp_pdays_12m'] = bg.conditional_expected_number_of_purchases_up_to_time(52, fr, rc, T)
cust['p_alive'] = bg.conditional_probability_alive(fr, rc, T)

# spend per purchase day, shrunk toward population mean (k = 3 pseudo-purchases)
spd = cust.monetary / (cust.lt_frequency + 1)
pop = spd.mean(); n = cust.lt_frequency + 1; K = 3
cust['spend_per_visit'] = (n * spd + K * pop) / (n + K)
cust['clv_12m'] = cust.exp_pdays_12m * cust.spend_per_visit
out['pop_spend_per_visit'] = float(pop)

out['pred_total_12m'] = float(cust.clv_12m.sum())
out['actual_total_12m'] = float(cust.hold_revenue.sum())
out['overprediction_pct'] = out['pred_total_12m'] / out['actual_total_12m'] - 1
out['pred_visits'] = float(cust.exp_pdays_12m.sum()); out['actual_visits'] = float(cust.hold_pdays.sum())
out['spearman_pred_actual'] = float(cust[['clv_12m', 'hold_revenue']].corr('spearman').iloc[0, 1])
out['naive_spearman_lastyear'] = float(cust[['monetary', 'hold_revenue']].corr('spearman').iloc[0, 1])

cust['clv_decile'] = pd.qcut(cust.clv_12m.rank(method='first'), 10, labels=False) + 1
dec = cust.groupby('clv_decile').agg(pred=('clv_12m', 'mean'), actual=('hold_revenue', 'mean'), buy=('hold_bought', 'mean'))
out['clv_deciles'] = dec.round(1).reset_index().to_dict('records')
top10 = cust.clv_decile == 10
out['top_decile_share_actual_2011'] = float(cust.loc[top10, 'hold_revenue'].sum() / cust.hold_revenue.sum())
naive_top = cust.monetary.rank(ascending=False, method='first') <= len(cust) * 0.1
out['naive_top_decile_share_actual_2011'] = float(cust.loc[naive_top, 'hold_revenue'].sum() / cust.hold_revenue.sum())
out['top10_overlap_clv_vs_lastyear'] = float((top10 & naive_top).sum() / top10.sum())

# tenure: where does overprediction come from?
cust['tenure_band'] = pd.cut(cust.tenure_days, [-1, 90, 180, 270, 366], labels=['0-3 mo', '3-6 mo', '6-9 mo', '9-12 mo'])
tb = cust.groupby('tenure_band', observed=True).agg(n=('clv_12m', 'size'), pred_total=('clv_12m', 'sum'), actual_total=('hold_revenue', 'sum'),
                                                    hist_median=('monetary', 'median'), buy=('hold_bought', 'mean'))
tb['over_pct'] = tb.pred_total / tb.actual_total - 1
out['tenure_bands'] = tb.round(3).reset_index().astype({'tenure_band': str}).to_dict('records')
# acquisition month of the newest customers
first_month = (pd.Timestamp('2010-12-01') - pd.to_timedelta(cust.tenure_days, 'D')).dt.to_period('M').astype(str)
out['acq_by_month'] = first_month.value_counts().sort_index().to_dict()

# segment names for k=4 log-scaled RFM
names = {}
for s, g in cust.groupby('seg_k4'):
    fq, r = g.frequency.median(), g.recency_days.median()
    names[s] = 'Champions' if fq >= 6 else ('Lapsing one-timers' if r >= 120 else ('Steady regulars' if fq >= 3 else 'Recent newcomers'))
cust['segment'] = cust.seg_k4.map(names)
sp = cust.groupby('segment').agg(n=('clv_12m', 'size'), recency=('recency_days', 'median'), frequency=('frequency', 'median'), monetary=('monetary', 'median'),
                                 rev_share=('monetary', 'sum'), clv_mean=('clv_12m', 'mean'), clv_median=('clv_12m', 'median'),
                                 clv_p10=('clv_12m', lambda s: s.quantile(.1)), clv_p90=('clv_12m', lambda s: s.quantile(.9)),
                                 clv_total=('clv_12m', 'sum'), actual_mean=('hold_revenue', 'mean'), buy_again=('hold_bought', 'mean'))
sp['rev_share'] = sp.rev_share / cust.monetary.sum()
sp['clv_share'] = sp.clv_total / sp.clv_total.sum()
sp['pct_customers'] = sp.n / len(cust)
out['segment_clv'] = sp.round(3).reset_index().to_dict('records')
out['top_decile_by_segment'] = cust[top10].segment.value_counts().to_dict()
# overlap of CLV distributions: share of Steady regulars with CLV above Champions median
ch_med = cust.loc[cust.segment == 'Champions', 'clv_12m'].median()
out['regulars_above_champion_median'] = float((cust.loc[cust.segment == 'Steady regulars', 'clv_12m'] > ch_med).mean())
out['champions_below_regulars_median'] = float((cust.loc[cust.segment == 'Champions', 'clv_12m'] < cust.loc[cust.segment == 'Steady regulars', 'clv_12m'].median()).mean())

# CFO twist sensitivity
def clv_h(months, disc, margin):
    tot = np.zeros(len(cust)); prev = np.zeros(len(cust))
    for m in range(1, months + 1):
        cum = bg.conditional_expected_number_of_purchases_up_to_time(m * 52 / 12, fr, rc, T).values
        tot += (cum - prev) * cust.spend_per_visit.values * margin / (1 + disc) ** (m / 12)
        prev = cum
    return tot
scen = [('1-yr revenue, no discounting', 12, 0, 1), ('3-yr revenue, no discounting', 36, 0, 1), ('3-yr revenue, 10% discount', 36, .10, 1),
        ('3-yr gross margin (30%), 10% discount', 36, .10, .30), ('5-yr gross margin (30%), 10% discount', 60, .10, .30)]
base_top = None; sens = []
for name, mths, d, mg in scen:
    v = clv_h(mths, d, mg)
    top = pd.Series(v, index=cust.index).rank(ascending=False, method='first') <= len(cust) * 0.1
    if base_top is None: base_top = top
    sens.append(dict(scenario=name, total=float(v.sum()), avg=float(v.mean()), median=float(np.median(v)),
                     top10_overlap=float((top & base_top).sum() / base_top.sum())))
    if name.startswith('3-yr gross'): cust['clv_3y_margin'] = v
out['cfo_sensitivity'] = sens
r = cust.hold_bought.mean()  # instructor-only: realised retention
out['simple_formula_note'] = 'm*(1+d)/(1+d-r): with r=0.64, d=0.10 multiplier = %.2f' % ((1.1) / (1.1 - 0.64))

cust['segment'] = cust['segment']
cust.to_pickle('cust_clv.pkl')
json.dump(out, open('facts_clv.json', 'w'), indent=1, default=float)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
for k, v in out.items():
    print('==', k); print(pd.DataFrame(v) if isinstance(v, list) and v and isinstance(v[0], dict) else v)
