"""K-means scenarios for the segmentation debrief."""
import pandas as pd, numpy as np, json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

cust = pd.read_pickle('cust.pkl')
out = {}
RFM = ['recency_days', 'frequency', 'monetary']

def fit(X, k, seed=42):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)

def profile(labels, cols=RFM + ['hold_bought', 'hold_revenue']):
    p = cust.assign(seg=labels).groupby('seg').agg(n=('frequency', 'size'), **{c: (c, 'median') for c in cols if c not in ('hold_bought', 'hold_revenue')},
                                                   rev_share=('monetary', 'sum'), buy_again=('hold_bought', 'mean'), hold_rev_mean=('hold_revenue', 'mean'))
    p['rev_share'] = p.rev_share / cust.monetary.sum()
    p['pct_customers'] = p.n / len(cust)
    return p.sort_values('monetary', ascending=False)

# Scenario A: raw, unscaled RFM
XA = cust[RFM].values
la = fit(XA, 4).labels_
out['A_raw_unscaled'] = profile(la).round(3).reset_index().to_dict('records')

# Scenario B: z-scored, no log
XB = StandardScaler().fit_transform(cust[RFM])
lb = fit(XB, 4).labels_
out['B_scaled_nolog'] = profile(lb).round(3).reset_index().to_dict('records')

# Scenario C: log + z-score  (recommended)
logged = cust[RFM].copy()
logged['frequency'] = np.log1p(logged.frequency); logged['monetary'] = np.log1p(logged.monetary); logged['recency_days'] = np.log1p(logged.recency_days)
XC = StandardScaler().fit_transform(logged)
ks = range(2, 9)
inertia, sil = [], []
for k in ks:
    km = fit(XC, k)
    inertia.append(float(km.inertia_)); sil.append(float(silhouette_score(XC, km.labels_, sample_size=3000, random_state=0)))
out['C_elbow'] = dict(k=list(ks), inertia=inertia, silhouette=sil)
lc = fit(XC, 4).labels_
out['C_log_scaled_k4'] = profile(lc).round(3).reset_index().to_dict('records')
lc5 = fit(XC, 5).labels_
out['C_log_scaled_k5'] = profile(lc5).round(3).reset_index().to_dict('records')
# seed stability
seeds = [fit(XC, 4, s).labels_ for s in range(5)]
out['C_seed_ARI_k4'] = [float(adjusted_rand_score(seeds[0], s)) for s in seeds[1:]]
seeds6 = [fit(XC, 6, s).labels_ for s in range(5)]
out['C_seed_ARI_k6'] = [float(adjusted_rand_score(seeds6[0], s)) for s in seeds6[1:]]
# label ids change between seeds even when partition is same
out['label_ids_seed0_vs1_first10'] = [seeds[0][:10].tolist(), seeds[1][:10].tolist()]

# Scenario D: add behavioural vars (breadth, basket size, AOV), logged + scaled
D = ['recency_days', 'frequency', 'monetary', 'product_breadth', 'avg_items_per_order', 'avg_order_value']
XDdf = np.log1p(cust[D]); XD = StandardScaler().fit_transform(XDdf)
ld = fit(XD, 4).labels_
out['D_behav_k4'] = profile(ld, D + ['hold_bought', 'hold_revenue']).round(3).reset_index().to_dict('records')
out['D_vs_C_ARI'] = float(adjusted_rand_score(lc, ld))

# Scenario E: add international binary flag
XE = StandardScaler().fit_transform(pd.concat([logged, cust[['international']]], axis=1))
le = fit(XE, 4).labels_
pe = cust.assign(seg=le).groupby('seg').agg(n=('frequency', 'size'), intl=('international', 'mean'), mon=('monetary', 'median'))
out['E_with_intl_flag'] = pe.round(3).reset_index().to_dict('records')

# Name segments for scenario C k=4 by medians
p = profile(lc)
cust['seg_k4'] = lc
cust['XC0'], cust['XC1'], cust['XC2'] = XC[:, 0], XC[:, 1], XC[:, 2]
cust['seg_A'] = la; cust['seg_B'] = lb
cust.to_pickle('cust_seg.pkl')
json.dump(out, open('facts_seg.json', 'w'), indent=1, default=float)
for k_, v in out.items():
    print('==', k_); print(pd.DataFrame(v) if isinstance(v, list) and v and isinstance(v[0], dict) else v)
