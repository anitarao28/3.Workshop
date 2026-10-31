# CRM Budget Simulation — customer analytics workshop

A Streamlit app for two 90-minute sessions on segmentation, CLV and prediction using the
UCI Online Retail II dataset. Students play the new Head of CRM with a £50k budget on 1 December 2010.
Everything after that date is hidden until the Reveal tab.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

To deploy on Streamlit Community Cloud, push this folder to GitHub and point the app at `app.py`.
Set the Reveal passcode in the app's **Secrets** as `REVEAL_CODE = "your-code"`.
Without a secret, the passcode is `holdout2011`.

> The passcode keeps the Reveal closed during class. It is not security. If students can see the repository,
> they can read `data/reveal.csv`. Share only the app link, not the repository.

## Tabs

| Tab | What students do | Key controls |
|---|---|---|
| Brief | Read the case | — |
| ① Segment | k-means on variables they choose | variables, log-transform, standardise, k, seed, segment names and actions |
| ② Value | CLV and the CFO challenge | horizon, discount rate, revenue vs margin, saved scenarios |
| ③ Predict | Read a churn model, pick a threshold | threshold, confusion matrix, lift by decile, drivers |
| ④ Plan | Value × risk map and budget allocation | budget, 40% cut twist, cost per contact, save rate, margin, targeting rule, lock in plan |
| 🔓 Reveal | What really happened in 2011 | passcode |

The sidebar has a **Show coaching hints** toggle. Hints flag common pitfalls as students work.
Suggested use: on for undergraduates, off for MBAs.

## Data files (`data/`)

| File | Contents |
|---|---|
| `customers.csv` | 4,239 customers: RFM and behaviour for 1 Dec 2009 – 30 Nov 2010, plus churn risk score |
| `clv_curve.csv` | Expected cumulative purchase visits for months 1–60 (BG/NBD model) and spend per visit |
| `model_validation.csv` | Held-out customers from the churn model's training period (snapshot 1 June 2010) |
| `model_drivers.csv` | Standardised logistic regression coefficients |
| `reveal.csv` | Actual behaviour 1 Dec 2010 – 30 Nov 2011 (read only by the Reveal tab) |

## How the numbers were built

- **Cleaning:** removed duplicate rows, rows without a customer ID, cancellations (kept as a return rate),
  postage/fees/adjustments, and zero-price lines.
- **CLV:** BG/NBD model (weekly time units) for expected purchase visits, multiplied by each customer's
  spend per visit shrunk toward the population average (3 pseudo-visits). The model is fitted on the
  calibration year only.
- **Churn model:** logistic regression trained on customers as of 1 June 2010; outcome = no purchase in the
  next 6 months. Scored on all customers as of 1 December 2010. Trained out-of-time, so the 2011 Reveal is
  a genuine test.

Source: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
