"""Clean UCI Online Retail II and build customer-level tables for the workshop."""
import pandas as pd, numpy as np, json

raw = pd.read_pickle('raw.pkl')
raw = raw.rename(columns={'Customer ID': 'CustomerID'})
facts = {}
facts['raw_rows'] = len(raw)
dup = raw.duplicated()
facts['duplicate_rows'] = int(dup.sum())
df = raw[~dup].copy()
facts['rows_after_dedup'] = len(df)
facts['share_rows_no_id'] = float(df.CustomerID.isna().mean())
df['Revenue'] = df.Quantity * df.Price
facts['share_revenue_no_id'] = float(df.loc[df.CustomerID.isna() & (df.Revenue > 0), 'Revenue'].sum() / df.loc[df.Revenue > 0, 'Revenue'].sum())

df['is_cancel'] = df.Invoice.str.startswith('C')
facts['share_rows_cancel'] = float(df.is_cancel.mean())

# non-product codes (postage, manual, fees, adjustments)
nonprod = df.StockCode.str.upper().str.match(r'^(POST|DOT|M|D|C2|CRUK|BANK CHARGES|AMAZONFEE|ADJUST|ADJUST2|S|B|PADS|TEST.*|GIFT.*|SP1002)$')
facts['nonproduct_rows'] = int(nonprod.sum())
df = df[~nonprod & df.CustomerID.notna()].copy()
df['CustomerID'] = df.CustomerID.astype(int)

sales = df[~df.is_cancel & (df.Quantity > 0) & (df.Price > 0)].copy()
cancels = df[df.is_cancel].copy()

CUTOFF = pd.Timestamp('2010-12-01')
HOLD_END = pd.Timestamp('2011-12-01')
cal = sales[sales.InvoiceDate < CUTOFF]
hold = sales[(sales.InvoiceDate >= CUTOFF) & (sales.InvoiceDate < HOLD_END)]
cal_c = cancels[cancels.InvoiceDate < CUTOFF]

facts['cal_customers'] = int(cal.CustomerID.nunique())
facts['cal_revenue'] = float(cal.Revenue.sum())
facts['cal_invoices'] = int(cal.Invoice.nunique())
facts['hold_customers_total'] = int(hold.CustomerID.nunique())

# customer features (calibration year)
inv = cal.groupby(['CustomerID', 'Invoice']).agg(date=('InvoiceDate', 'min'), rev=('Revenue', 'sum'), qty=('Quantity', 'sum')).reset_index()
g = inv.groupby('CustomerID')
cust = pd.DataFrame({
    'recency_days': (CUTOFF - g.date.max()).dt.days,
    'frequency': g.Invoice.nunique(),
    'monetary': g.rev.sum(),
    'tenure_days': (CUTOFF - g.date.min()).dt.days,
    'avg_order_value': g.rev.mean(),
    'avg_items_per_order': g.qty.mean(),
})
cust['product_breadth'] = cal.groupby('CustomerID').StockCode.nunique()
canc_val = (-cal_c.Revenue).groupby(cal_c.CustomerID).sum()
cust['return_rate'] = (canc_val.reindex(cust.index).fillna(0) / (cust.monetary + canc_val.reindex(cust.index).fillna(0))).clip(0, 1)
country = cal.groupby('CustomerID').Country.agg(lambda s: s.mode().iloc[0])
cust['country'] = country
cust['international'] = (country != 'United Kingdom').astype(int)
# days in calibration as a lifetimes summary
first = g.date.min(); last = g.date.max()
inv['day'] = inv.date.dt.normalize()
daily = inv.groupby(['CustomerID', 'day']).rev.sum().reset_index()
gd = daily.groupby('CustomerID')
cust['lt_frequency'] = gd.day.count() - 1  # repeat purchase days
cust['lt_recency'] = (gd.day.max() - gd.day.min()).dt.days
cust['lt_T'] = (CUTOFF - gd.day.min()).dt.days
rep = daily.sort_values('day').groupby('CustomerID').apply(lambda d: d.rev.iloc[1:].mean() if len(d) > 1 else np.nan, include_groups=False)
cust['lt_monetary'] = rep  # mean value of repeat transactions

# holdout outcomes
hinv = hold.groupby(['CustomerID', 'Invoice']).Revenue.sum().reset_index()
hg = hinv.groupby('CustomerID')
cust['hold_bought'] = cust.index.isin(hg.groups.keys()).astype(int)
cust['hold_revenue'] = hg.Revenue.sum().reindex(cust.index).fillna(0)
cust['hold_orders'] = hg.Invoice.nunique().reindex(cust.index).fillna(0).astype(int)

facts['hold_customers_from_cal'] = int(cust.hold_bought.sum())
facts['new_customers_in_hold'] = facts['hold_customers_total'] - facts['hold_customers_from_cal']
facts['base_rate_buy_again'] = float(cust.hold_bought.mean())
facts['single_purchasers_share'] = float((cust.frequency == 1).mean())

# concentration
m = cust.monetary.sort_values(ascending=False)
for p in [0.01, 0.05, 0.10, 0.20]:
    facts[f'top{int(p*100)}_rev_share'] = float(m.iloc[:int(len(m) * p)].sum() / m.sum())
facts['median_monetary'] = float(cust.monetary.median())
facts['mean_monetary'] = float(cust.monetary.mean())
facts['max_monetary'] = float(cust.monetary.max())
facts['median_frequency'] = float(cust.frequency.median())
facts['corr_freq_monetary'] = float(cust[['frequency', 'monetary']].corr().iloc[0, 1])
facts['corr_logfreq_logmon'] = float(np.log1p(cust[['frequency', 'monetary']]).corr().iloc[0, 1])
facts['share_international'] = float(cust.international.mean())

cust.to_pickle('cust.pkl')
cust.reset_index().to_csv('customers_calibration_2010.csv', index=False)
json.dump(facts, open('facts_prep.json', 'w'), indent=1)
print(json.dumps(facts, indent=1))
print(cust.describe().T.round(1))
