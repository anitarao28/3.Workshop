"""Build the small, precomputed data files the Streamlit app reads (no modelling libraries needed at runtime)."""
import pandas as pd, numpy as np, json, os, warnings
warnings.filterwarnings('ignore')
from lifetimes import BetaGeoFitter

OUT = 'app/data'; os.makedirs(OUT, exist_ok=True)
c = pd.read_pickle('cust_final.pkl')          # calibration features + CLV + risk + outcomes
s = pd.read_pickle('pred_scores.pkl')

# 1) customers.csv — what students see (state of the world on 1 Dec 2010)
cols = ['recency_days', 'frequency', 'monetary', 'tenure_days', 'avg_order_value', 'avg_items_per_order',
        'product_breadth', 'return_rate', 'international', 'country']
cust = c[cols].copy()
cust['monetary'] = cust.monetary.round(2); cust['avg_order_value'] = cust.avg_order_value.round(2)
cust['avg_items_per_order'] = cust.avg_items_per_order.round(1); cust['return_rate'] = cust.return_rate.round(4)
cust['p_churn_6m'] = c.p_lr.round(4)        # risk score from the churn model (scored 1 Dec 2010)
cust.index.name = 'customer_id'
cust.to_csv(f'{OUT}/customers.csv')

# 2) clv_curve.csv — expected cumulative purchase visits at month 1..60 (BG/NBD, weekly units) + spend per visit
fr, rc, T = c.lt_frequency, c.lt_recency / 7, c.lt_T / 7
bg = BetaGeoFitter().fit(fr, rc, T)
curve = pd.DataFrame(index=c.index)
for m in range(1, 61):
    curve[f'm{m}'] = bg.conditional_expected_number_of_purchases_up_to_time(m * 52 / 12, fr, rc, T).round(4)
curve.insert(0, 'spend_per_visit', c.spend_per_visit.round(2))
curve.index.name = 'customer_id'
curve.to_csv(f'{OUT}/clv_curve.csv')

# 3) model_validation.csv — hold-out customers from the TRAINING period (June 2010 snapshot) for the Predict tab
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import importlib.util, sys
spec = importlib.util.spec_from_file_location('pred', 'pred.py')
# re-create the training snapshot features without re-running everything
exec(open('pred.py').read().split("tr = features(")[0])   # defines features(), X(), LOGCOLS
tr = features(pd.Timestamp('2010-06-01'))
xtr, xte, ytr, yte = train_test_split(X(tr), tr.churn_6m, test_size=0.3, random_state=0, stratify=tr.churn_6m)
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(xtr, ytr)
val = pd.DataFrame({'p_churn': m.predict_proba(xte)[:, 1].round(4), 'churned': yte.values})
val.to_csv(f'{OUT}/model_validation.csv', index=False)
full = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X(tr), tr.churn_6m)
drivers = pd.Series(full[-1].coef_[0], index=X(tr).columns).round(3).sort_values()
nice = {'tenure_days': 'Tenure (days as customer)', 'orders_last_90d': 'Orders in last 90 days', 'product_breadth': 'Product breadth',
        'spend_per_month': 'Spend per month', 'orders_per_month': 'Orders per month', 'is_repeat': 'Has bought more than once',
        'avg_order_value': 'Average order value', 'return_rate': 'Return rate', 'international': 'International customer',
        'avg_items_per_order': 'Items per order', 'recency_days': 'Days since last order'}
drivers.rename(index=nice).rename('coef').to_csv(f'{OUT}/model_drivers.csv')

# 4) reveal.csv — what actually happened (Dec 2010 - Nov 2011). Only the Reveal tab reads this.
rev = pd.DataFrame({'bought_next_12m': c.hold_bought, 'revenue_next_12m': c.hold_revenue.round(2),
                    'visits_next_12m': c.hold_pdays.astype(int), 'lapsed_next_6m': c.churn_6m.astype(int)}, index=c.index)
rev.index.name = 'customer_id'
rev.to_csv(f'{OUT}/reveal.csv')
meta = dict(snapshot='2010-12-01', calibration='2009-12-01 to 2010-11-30', holdout='2010-12-01 to 2011-11-30',
            n_customers=len(cust), new_customers_2011=1585, train_snapshot='2010-06-01', train_churn_rate=float(tr.churn_6m.mean()))
json.dump(meta, open(f'{OUT}/meta.json', 'w'), indent=1)
for f in os.listdir(OUT): print(f, os.path.getsize(f'{OUT}/{f}') // 1024, 'KB')
