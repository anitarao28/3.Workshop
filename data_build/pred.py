"""Out-of-time churn model: train on snapshot 2010-06-01 (label Jun-Nov 2010), score at 2010-12-01, reveal on Dec 2010-May 2011."""
import pandas as pd, numpy as np, json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

raw = pd.read_pickle('raw.pkl').rename(columns={'Customer ID': 'CustomerID'}).drop_duplicates()
raw = raw[raw.CustomerID.notna()].copy(); raw['CustomerID'] = raw.CustomerID.astype(int)
raw['Revenue'] = raw.Quantity * raw.Price
nonprod = raw.StockCode.str.upper().str.match(r'^(POST|DOT|M|D|C2|CRUK|BANK CHARGES|AMAZONFEE|ADJUST|ADJUST2|S|B|PADS|TEST.*|GIFT.*|SP1002)$')
raw = raw[~nonprod]
is_c = raw.Invoice.str.startswith('C')
sales = raw[~is_c & (raw.Quantity > 0) & (raw.Price > 0)]
canc = raw[is_c]
H = pd.DateOffset(months=6)

def features(snap):
    s = sales[sales.InvoiceDate < snap]; c = canc[canc.InvoiceDate < snap]
    inv = s.groupby(['CustomerID', 'Invoice']).agg(date=('InvoiceDate', 'min'), rev=('Revenue', 'sum'), qty=('Quantity', 'sum')).reset_index()
    g = inv.groupby('CustomerID')
    tenure = (snap - g.date.min()).dt.days
    months = (tenure.clip(lower=30) / 30.4)
    last90 = inv[inv.date >= snap - pd.Timedelta(days=90)].groupby('CustomerID').Invoice.nunique()
    f = pd.DataFrame({
        'recency_days': (snap - g.date.max()).dt.days.clip(upper=182),
        'orders_per_month': g.Invoice.nunique() / months,
        'orders_last_90d': last90.reindex(g.size().index).fillna(0),
        'spend_per_month': g.rev.sum() / months,
        'avg_order_value': g.rev.mean(),
        'tenure_days': tenure.clip(upper=182),
        'product_breadth': s.groupby('CustomerID').StockCode.nunique(),
        'avg_items_per_order': g.qty.mean(),
        'is_repeat': (g.Invoice.nunique() > 1).astype(int),
    })
    cv = (-c.Revenue).groupby(c.CustomerID).sum().reindex(f.index).fillna(0)
    f['return_rate'] = cv / (g.rev.sum() + cv)
    f['international'] = (s.groupby('CustomerID').Country.agg(lambda x: x.mode().iloc[0]) != 'United Kingdom').astype(int)
    fut = sales[(sales.InvoiceDate >= snap) & (sales.InvoiceDate < snap + H)]
    f['churn_6m'] = (~f.index.isin(fut.CustomerID.unique())).astype(int)
    return f

LOGCOLS = ['orders_per_month', 'orders_last_90d', 'spend_per_month', 'avg_order_value', 'product_breadth', 'avg_items_per_order']
def X(f):
    x = f.drop(columns='churn_6m').copy()
    for c in LOGCOLS: x[c] = np.log1p(x[c])
    return x

tr = features(pd.Timestamp('2010-06-01'))
sc = features(pd.Timestamp('2010-12-01'))
out = dict(n_train=len(tr), n_score=len(sc), churn_rate_train=float(tr.churn_6m.mean()), churn_rate_actual_score=float(sc.churn_6m.mean()))

models = {'Logistic regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
          'Gradient boosting': GradientBoostingClassifier(random_state=0, max_depth=3, n_estimators=200, learning_rate=0.05)}
# random split within the training snapshot (what students usually see)
xtr, xte, ytr, yte = train_test_split(X(tr), tr.churn_6m, test_size=0.3, random_state=0, stratify=tr.churn_6m)
res = {}
XS = X(sc)
for n, m in models.items():
    m.fit(xtr, ytr); auc_rand = roc_auc_score(yte, m.predict_proba(xte)[:, 1])
    m.fit(X(tr), tr.churn_6m); p = m.predict_proba(XS)[:, 1]
    res[n] = dict(auc_random_split=float(auc_rand), auc_out_of_time=float(roc_auc_score(sc.churn_6m, p)), mean_pred=float(p.mean()))
    sc['p_' + ('lr' if n.startswith('Log') else 'gb')] = p
out['models'] = res
# naive rule: recency > 90 days
out['auc_recency_only'] = float(roc_auc_score(sc.churn_6m, sc.recency_days))
lr = models['Logistic regression']
coefs = pd.Series(lr[-1].coef_[0], index=XS.columns).sort_values()
out['lr_coefs_std'] = coefs.round(3).to_dict()

p = sc.p_lr; y = sc.churn_6m
# decile lift
sc['risk_decile'] = pd.qcut(p.rank(method='first'), 10, labels=False) + 1
d = sc.groupby('risk_decile').agg(pred=('p_lr', 'mean'), actual=('churn_6m', 'mean'), n=('p_lr', 'size')).reset_index()
d['lift'] = d.actual / y.mean()
out['deciles'] = d.round(3).to_dict('records')
# confusion matrices at thresholds
cms = {}
for t in [0.5, 0.3, 0.7]:
    tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int)).ravel()
    cms[str(t)] = dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), accuracy=float((tp + tn) / len(y)), precision=float(tp / (tp + fp)), recall=float(tp / (tp + fn)), flagged=int(tp + fp))
out['confusion'] = cms
out['accuracy_predict_nobody_churns'] = float(1 - y.mean())
out['accuracy_predict_everyone_churns'] = float(y.mean())

sc.to_pickle('pred_scores.pkl')
json.dump(out, open('facts_pred.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))

# ---- validation metrics inside the training snapshot (what students see before the reveal)
vm = models['Logistic regression']
vm.fit(xtr, ytr); pv = vm.predict_proba(xte)[:, 1]
v = pd.DataFrame({'p': pv, 'y': yte.values})
v['dec'] = pd.qcut(v.p.rank(method='first'), 10, labels=False) + 1
vd = v.groupby('dec').agg(pred=('p', 'mean'), actual=('y', 'mean'), n=('y', 'size')).reset_index()
val = dict(n=len(v), base_rate=float(v.y.mean()), auc=float(roc_auc_score(v.y, v.p)), deciles=vd.round(3).to_dict('records'))
val['acc_predict_nobody'] = float(1 - v.y.mean())
cmv = {}
for t in [0.2, 0.3, 0.5]:
    tn, fp, fn, tp = confusion_matrix(v.y, (v.p >= t).astype(int)).ravel()
    cmv[str(t)] = dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), flagged=int(tp + fp), accuracy=float((tp + tn) / len(v)), precision=float(tp / (tp + fp)) if tp + fp else None, recall=float(tp / (tp + fn)))
val['confusion'] = cmv
val['auc_recency_only'] = float(roc_auc_score(v.y, xte.recency_days.values))
json.dump(val, open('facts_pred_val.json', 'w'), indent=1)
print(json.dumps(val, indent=1))
