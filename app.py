"""
Customer Analytics Workshop Simulation
UCI Online Retail II — "You're the new Head of CRM. You have £50k. Who do you spend it on?"

Run:  streamlit run app.py
Data: precomputed files in ./data (built from the UCI Online Retail II dataset).
Reveal passcode: set REVEAL_CODE in .streamlit/secrets.toml (default: reveal).
"""
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="CRM Budget Simulation", page_icon="🛍️", layout="wide")

PAL = ["#1F6F78", "#E4572E", "#F2A541", "#2E4A62", "#9BC1BC", "#7A8B99", "#B56576", "#6D9F71"]
INK, MUTED = "#1B2A41", "#6B7785"

VARS = {
    "recency_days": "Recency (days since last order)",
    "frequency": "Frequency (orders in the year)",
    "monetary": "Monetary (£ spent in the year)",
    "tenure_days": "Tenure (days since first order)",
    "avg_order_value": "Average order value (£)",
    "avg_items_per_order": "Items per order",
    "product_breadth": "Product breadth (distinct products)",
    "return_rate": "Return rate (share of £ cancelled)",
    "international": "International customer (0/1)",
}
BINARY = {"international"}


# ----------------------------------------------------------------------------- data
@st.cache_data
def load():
    cust = pd.read_csv("data/customers.csv", index_col="customer_id")
    curve = pd.read_csv("data/clv_curve.csv", index_col="customer_id").loc[cust.index]
    val = pd.read_csv("data/model_validation.csv")
    drivers = pd.read_csv("data/model_drivers.csv", index_col=0)["coef"]
    meta = json.load(open("data/meta.json"))
    return cust, curve, val, drivers, meta


@st.cache_data
def load_reveal():
    return pd.read_csv("data/reveal.csv", index_col="customer_id")


cust, curve, val, drivers, meta = load()
N = len(cust)
VISITS = curve[[f"m{m}" for m in range(1, 61)]].to_numpy()
SPV = curve["spend_per_visit"].to_numpy()


def clv(horizon_years=1, discount=0.0, margin=1.0):
    """Expected (discounted) value over the horizon: monthly expected visits x spend per visit x margin."""
    months = int(horizon_years * 12)
    inc = np.diff(np.hstack([np.zeros((N, 1)), VISITS[:, :months]]), axis=1)
    disc = 1 / (1 + discount) ** (np.arange(1, months + 1) / 12)
    return pd.Series((inc * disc).sum(axis=1) * SPV * margin, index=cust.index)


CLV_12M = clv(1, 0, 1)  # 12-month revenue CLV used for planning


def gbp(x, d=0):
    if abs(x) >= 1e6:
        return f"£{x / 1e6:,.2f}M"
    if abs(x) >= 1e4:
        return f"£{x / 1e3:,.0f}k"
    return f"£{x:,.{d}f}"


def pct(x, d=0):
    return f"{x * 100:.{d}f}%"


def style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), font=dict(color=INK),
                      plot_bgcolor="white", paper_bgcolor="white", legend_title_text="")
    fig.update_xaxes(showgrid=False, linecolor="#CBD2D9")
    fig.update_yaxes(gridcolor="#EEF2F5", zeroline=False)
    return fig


ss = st.session_state
ss.setdefault("seg", None)            # pd.Series cluster id per customer
ss.setdefault("seg_names", {})        # cluster id -> name
ss.setdefault("seg_config", None)
ss.setdefault("clv_scenarios", [])
ss.setdefault("plan", None)

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🛍️ CRM Budget Simulation")
    team = st.text_input("Team name", value=ss.get("team", ""), placeholder="e.g. Team 4")
    ss["team"] = team
    hints = st.toggle("Show coaching hints", value=False,
                      help="Hints flag common pitfalls as you work. Your instructor will tell you whether to turn these on.")
    st.divider()
    st.caption("**Your progress**")
    st.caption(("✅" if ss.seg is not None else "⬜") + " 1 · Segments built")
    st.caption(("✅" if ss.clv_scenarios else "⬜") + " 2 · CLV scenario saved")
    st.caption(("✅" if ss.get("threshold_chosen") else "⬜") + " 3 · Risk threshold chosen")
    st.caption(("✅" if ss.plan is not None else "⬜") + " 4 · Plan locked in")
    st.divider()
    st.caption(f"Data: UCI Online Retail II · {N:,} customers. "
               "Everything that happens after today is hidden until the Reveal.")


def hint(msg):
    if hints:
        st.info("💡 " + msg)


tabs = st.tabs(["📋 Brief", "① Segment", "② Value (CLV)", "③ Predict", "④ Plan", "🔓 Reveal"])

# ============================================================================= BRIEF
with tabs[0]:
    st.title("You're the new Head of CRM")
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown(f"""
It is the start of December. You have just joined a UK online retailer that sells giftware and homeware,
mostly to **small shops and wholesalers**, plus some consumers. Over the past year it served
**{N:,} identifiable customers** who spent **{gbp(cust.monetary.sum())}**.

The CEO has given you **£50,000** for retention and growth campaigns over the next 12 months and wants answers to three questions:

1. **Who are our customers?** Build segments the business can act on.
2. **What are they worth?** Estimate customer lifetime value and defend your assumptions.
3. **Who should we spend the £50k on, and what return should we expect?**

Work through the tabs in order. Each one feeds the next: your segments carry into the value and planning tabs,
and the plan you lock in is scored against **what customers actually did over the following 12 months** at the end of Session 2.
""")
        st.markdown("#### How the sessions run")
        st.markdown("""
| Session | Tabs | You leave with |
|---|---|---|
| **1 · Diagnose** | ① Segment → ② Value | A value map: named segments and what each is worth |
| **2 · Act** | ③ Predict → ④ Plan → 🔓 Reveal | A funded plan, a 3-minute pitch, and a reality check |
""")
    with c2:
        st.markdown("#### The customer base at a glance")
        a, b = st.columns(2)
        a.metric("Customers", f"{N:,}")
        b.metric("Revenue (12 months)", gbp(cust.monetary.sum()))
        a.metric("Median spend per customer", gbp(cust.monetary.median()))
        b.metric("Mean spend per customer", gbp(cust.monetary.mean()))
        a.metric("Bought only once", pct((cust.frequency == 1).mean()))
        b.metric("Customers outside the UK", f"{int(cust.international.sum()):,}")
        # The top-10% revenue share is the gut-check answer: it is revealed in the Value tab, not here.
        with st.expander("What's in the data?"):
            st.markdown("""
Each row is one customer, summarising their purchases over the **last 12 months** (December to November).

- Transactions without a customer ID (about 23% of rows) and cancellations have been removed;
  cancellations are kept only as a *return rate*.
- Postage, fees and manual adjustments have been removed.
- *Source: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.*
""")

# ============================================================================= SEGMENT
with tabs[1]:
    st.header("① Segment: who are our customers?")
    st.caption("Choose the variables k-means should use, decide how to prepare them, pick k, then name the segments.")

    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        chosen = st.multiselect("Variables to cluster on", list(VARS), default=["recency_days", "frequency", "monetary"],
                                format_func=lambda v: VARS[v])
        log_t = st.checkbox("Log-transform skewed variables", value=False,
                            help="Applies log(1+x) to every non-binary variable.")
        scale_t = st.checkbox("Standardise variables (z-scores)", value=False,
                              help="Puts every variable on the same scale: mean 0, standard deviation 1.")
        k = st.slider("Number of segments (k)", 2, 8, 4)
        seed = st.number_input("Random seed", 0, 999, 42, help="k-means starts from random points. Try another seed: do your segments change?")
        run = st.button("Run k-means", type="primary", width="stretch", disabled=len(chosen) == 0)

    @st.cache_data(show_spinner=False)
    def prep(cols, log_t, scale_t):
        X = cust[list(cols)].astype(float).copy()
        if log_t:
            for c_ in cols:
                if c_ not in BINARY:
                    X[c_] = np.log1p(X[c_].clip(lower=0))
        X = X.to_numpy()
        if scale_t:
            X = StandardScaler().fit_transform(X)
        return X

    @st.cache_data(show_spinner="Clustering…")
    def kmeans(cols, log_t, scale_t, k, seed):
        X = prep(cols, log_t, scale_t)
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        # relabel clusters 1..k by median spend so numbering is stable and readable
        lab = pd.Series(km.labels_, index=cust.index)
        order = cust.monetary.groupby(lab).median().sort_values(ascending=False).index
        mapping = {old: i + 1 for i, old in enumerate(order)}
        return lab.map(mapping)

    @st.cache_data(show_spinner="Scoring k = 2 to 8…")
    def elbow(cols, log_t, scale_t, seed):
        X = prep(cols, log_t, scale_t)
        rows = []
        for kk in range(2, 9):
            km = KMeans(n_clusters=kk, n_init=5, random_state=seed).fit(X)
            sil = silhouette_score(X, km.labels_, sample_size=min(2000, N), random_state=0)
            rows.append(dict(k=kk, spread=km.inertia_, silhouette=sil))
        return pd.DataFrame(rows)

    if run:
        ss.seg = kmeans(tuple(chosen), log_t, scale_t, k, seed)
        ss.seg_config = dict(vars=list(chosen), log=log_t, scale=scale_t, k=k, seed=seed)
        ss.seg_names = {i: ss.seg_names.get(i, f"Segment {i}") for i in range(1, k + 1)} if len(ss.seg_names) == k else {i: f"Segment {i}" for i in range(1, k + 1)}

    with c2:
        if ss.seg is None:
            st.info("Choose your settings and press **Run k-means**.")
        else:
            cfg = ss.seg_config
            if cfg["vars"] != chosen or cfg["log"] != log_t or cfg["scale"] != scale_t or cfg["k"] != k or cfg["seed"] != seed:
                st.warning("Settings changed since the last run. Press **Run k-means** to update.")
            seg = ss.seg
            prof = pd.DataFrame({
                "Customers": seg.value_counts().sort_index(),
                "% of customers": seg.value_counts(normalize=True).sort_index(),
                "% of revenue": cust.monetary.groupby(seg).sum() / cust.monetary.sum(),
            })
            shown = list(dict.fromkeys(["recency_days", "frequency", "monetary"] + cfg["vars"]))
            med = cust[shown].groupby(seg).median()
            med.columns = ["Median " + VARS[c_].split(" (")[0].lower() for c_ in shown]
            prof = prof.join(med)
            prof.index = [f"{i} · {ss.seg_names.get(i, f'Segment {i}')}" for i in prof.index]

            sh = prof[["% of customers", "% of revenue"]].reset_index().melt(id_vars="index", var_name="Measure", value_name="Share")
            fig = px.bar(sh, x="index", y="Share", color="Measure", barmode="group", text_auto=".0%",
                         color_discrete_sequence=[PAL[5], PAL[0]], title="Share of customers vs share of revenue")
            fig.update_yaxes(tickformat=".0%", title=None); fig.update_xaxes(title=None)
            st.plotly_chart(style(fig, 330), width="stretch")

            fmt = {"% of customers": "{:.1%}", "% of revenue": "{:.1%}"}
            for c_ in prof.columns:
                if c_.startswith("Median"):
                    fmt[c_] = "£{:,.0f}" if ("monetary" in c_ or "order value" in c_) else "{:,.1f}"
            st.dataframe(prof.style.format(fmt), width="stretch")

            # automatic flags
            shares = seg.value_counts(normalize=True)
            if hints:
                if shares.max() > 0.70:
                    hint(f"One segment holds {pct(shares.max())} of customers. Is k-means splitting customers, or isolating a few extreme ones? Check which variable has the biggest numbers.")
                if (shares < 0.02).any():
                    hint(f"{int((shares < 0.02).sum())} segment(s) hold under 2% of customers. Would you run a campaign for a handful of accounts, or are these outliers?")
                if not cfg["scale"]:
                    hint("Variables are on very different scales (days vs £). k-means uses distance, so the variable with the largest numbers dominates.")
                if any(v in BINARY for v in cfg["vars"]):
                    hint("You included a 0/1 variable. Look at whether one segment is simply 'everyone with a 1'.")
                if "frequency" in cfg["vars"] and "monetary" in cfg["vars"]:
                    hint("Frequency and monetary move together strongly (customers who order more, spend more). Including both gives 'size' extra weight.")

    if ss.seg is not None:
        st.subheader("Look inside your segments")
        c1, c2 = st.columns([2, 1], gap="large")
        with c1:
            a, b, l = st.columns([2, 2, 1])
            xv = a.selectbox("X axis", list(VARS), index=0, format_func=lambda v: VARS[v])
            yv = b.selectbox("Y axis", list(VARS), index=2, format_func=lambda v: VARS[v])
            lg = l.checkbox("Log axes", True)
            d = cust[[xv, yv]].copy()
            d["Segment"] = ss.seg.map(lambda i: f"{i} · {ss.seg_names.get(i, '')}")
            d = d.sample(min(2500, N), random_state=0).sort_values("Segment")
            fig = px.scatter(d, x=xv, y=yv, color="Segment", opacity=0.6, color_discrete_sequence=PAL,
                             log_x=lg and xv not in BINARY, log_y=lg and yv not in BINARY,
                             labels={xv: VARS[xv], yv: VARS[yv]})
            fig.update_traces(marker=dict(size=5))
            st.plotly_chart(style(fig, 420), width="stretch")
        with c2:
            st.markdown("**Name your segments** and note what you would *do* with each.")
            names_df = pd.DataFrame({"Segment": list(ss.seg_names), "Name": list(ss.seg_names.values()),
                                     "Action": [ss.get("seg_actions", {}).get(i, "") for i in ss.seg_names]})
            edited = st.data_editor(names_df, hide_index=True, width="stretch", disabled=["Segment"],
                                    key=f"names_{hash(str(ss.seg_config))}")
            if st.button("Save names", width="stretch"):
                ss.seg_names = dict(zip(edited.Segment, edited.Name))
                ss["seg_actions"] = dict(zip(edited.Segment, edited.Action))
                st.rerun()

        with st.expander("How many segments? Elbow and silhouette for your current settings"):
            e = elbow(tuple(ss.seg_config["vars"]), ss.seg_config["log"], ss.seg_config["scale"], ss.seg_config["seed"])
            a, b = st.columns(2)
            f1 = px.line(e, x="k", y="spread", markers=True, title="Within-segment spread (look for the elbow)",
                         color_discrete_sequence=[PAL[0]])
            f2 = px.line(e, x="k", y="silhouette", markers=True, title="Silhouette (higher = better separated)",
                         color_discrete_sequence=[PAL[1]])
            a.plotly_chart(style(f1, 300), width="stretch")
            b.plotly_chart(style(f2, 300), width="stretch")
            hint("The statistically 'best' k is not always the most useful one. Ask: could the business run a different campaign for each segment?")

# ============================================================================= VALUE
with tabs[2]:
    st.header("② Value: what is each customer worth?")
    st.caption("CLV here = expected future purchase visits (projected from each customer's buying pattern) × their typical spend per visit, "
               "summed over your chosen horizon, then converted to margin and discounted.")

    c1, c2 = st.columns([1, 3], gap="large")
    with c1:
        horizon = st.select_slider("Horizon (years)", [1, 2, 3, 4, 5], value=1)
        disc = st.slider("Annual discount rate", 0, 20, 0, format="%d%%") / 100
        basis = st.radio("Measure value as", ["Revenue", "Gross margin"], horizontal=True)
        margin = st.slider("Gross margin", 10, 60, 30, format="%d%%", disabled=basis == "Revenue") / 100 if basis == "Gross margin" else 1.0
        v = clv(horizon, disc, margin)
        label = f"{horizon}-yr {basis.lower()}" + (f" @ {int(margin * 100)}%" if basis == "Gross margin" else "") + f", {int(disc * 100)}% discount"
        if st.button("Save this scenario", width="stretch"):
            top = set(v.sort_values(ascending=False).index[: N // 10])
            ss.clv_scenarios.append(dict(Scenario=label, Total=v.sum(), Mean=v.mean(), Median=v.median(), _top=top))
    with c2:
        s_ = v.sort_values(ascending=False)
        a, b, c, d = st.columns(4)
        a.metric("Total CLV", gbp(v.sum()))
        b.metric("Mean per customer", gbp(v.mean()))
        c.metric("Median per customer", gbp(v.median()))
        d.metric("Share held by top 10%", pct(s_.iloc[: N // 10].sum() / s_.sum()))
        m_ = cust.monetary.sort_values(ascending=False)
        st.info(f"**Check your gut:** last year the top 10% of customers brought in "
                f"**{pct(m_.iloc[: N // 10].sum() / m_.sum())}** of revenue. Compare it with the share of future value above.")
        hint("Mean vs median: when they are far apart, a few very large customers are pulling the average up. Which number would you put in front of the CEO?")

        cc1, cc2 = st.columns(2)
        with cc1:
            cum = pd.DataFrame({"Share of customers": np.arange(1, N + 1) / N, "Share of CLV": s_.cumsum().to_numpy() / s_.sum()})
            fig = px.area(cum.iloc[::10], x="Share of customers", y="Share of CLV", title="How concentrated is value?",
                          color_discrete_sequence=[PAL[0]])
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=MUTED), name="If everyone were equal")
            fig.update_xaxes(tickformat=".0%"); fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(style(fig, 340), width="stretch")
        with cc2:
            if ss.seg is None:
                st.info("Build segments in tab ① to see value by segment.")
            else:
                d_ = pd.DataFrame({"CLV": v, "Segment": ss.seg.map(lambda i: f"{i} · {ss.seg_names.get(i, '')}")}).sort_values("Segment")
                fig = px.box(d_, x="Segment", y="CLV", color="Segment", log_y=True, points=False,
                             color_discrete_sequence=PAL, title="CLV by segment (log scale)")
                fig.update_layout(showlegend=False); fig.update_xaxes(title=None); fig.update_yaxes(title="CLV (£)")
                st.plotly_chart(style(fig, 340), width="stretch")
        if ss.seg is not None:
            t = pd.DataFrame({"CLV": v, "seg": ss.seg}).groupby("seg").CLV.agg(["size", "mean", "median", "sum"])
            t["share"] = t["sum"] / t["sum"].sum()
            t.index = [f"{i} · {ss.seg_names.get(i, '')}" for i in t.index]
            t.columns = ["Customers", "Mean CLV", "Median CLV", "Total CLV", "Share of total CLV"]
            st.dataframe(t.style.format({"Mean CLV": "£{:,.0f}", "Median CLV": "£{:,.0f}", "Total CLV": "£{:,.0f}", "Share of total CLV": "{:.1%}"}),
                         width="stretch")
            hint("Look at the spread inside each segment, not just the middle. Are some customers in a 'low' segment worth more than some in a 'high' one?")

    if ss.clv_scenarios:
        st.subheader("Your saved scenarios")
        base = ss.clv_scenarios[0]["_top"]
        tab = pd.DataFrame([{k_: v_ for k_, v_ in s.items() if k_ != "_top"} | {"Same top-10% customers as first scenario": len(s["_top"] & base) / len(base)}
                            for s in ss.clv_scenarios])
        st.dataframe(tab.style.format({"Total": "£{:,.0f}", "Mean": "£{:,.0f}", "Median": "£{:,.0f}", "Same top-10% customers as first scenario": "{:.0%}"}),
                     width="stretch", hide_index=True)
        if st.button("Clear scenarios"):
            ss.clv_scenarios = []; st.rerun()

    with st.expander("Top 25 customers by CLV"):
        top = cust.loc[s_.index[:25], ["country", "frequency", "monetary", "recency_days", "tenure_days"]].assign(CLV=s_.iloc[:25])
        if ss.seg is not None:
            top["Segment"] = ss.seg.loc[top.index].map(lambda i: ss.seg_names.get(i, i))
        st.dataframe(top.style.format({"monetary": "£{:,.0f}", "CLV": "£{:,.0f}"}), width="stretch")

# ============================================================================= PREDICT
with tabs[3]:
    st.header("③ Predict: who are we about to lose?")
    st.caption(f"A churn model scores every customer's risk of **placing no order in the next 6 months**. "
               "It was built on customers as they looked six months ago, and checked on customers it had not seen.")

    st.subheader("How good is the model?")
    st.caption(f"Checked on {len(val):,} held-out customers from the training period. "
               f"{pct(val.churned.mean())} of them did not order in the following 6 months.")
    auc = roc_auc_score(val.churned, val.p_churn)
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        thr = st.slider("Flag a customer as 'at risk' when risk score is at least", 0.05, 0.80, 0.50, 0.05)
        ss["threshold"] = thr
        ss["threshold_chosen"] = st.checkbox("We've agreed our threshold", value=ss.get("threshold_chosen", False))
        pred = val.p_churn >= thr
        tp = int((pred & (val.churned == 1)).sum()); fp = int((pred & (val.churned == 0)).sum())
        fn = int((~pred & (val.churned == 1)).sum()); tn = int((~pred & (val.churned == 0)).sum())
        cm = pd.DataFrame([[tp, fn], [fp, tn]], index=["Actually lapsed", "Actually stayed"], columns=["Flagged at risk", "Not flagged"])
        st.dataframe(cm, width="stretch")
        a, b = st.columns(2)
        a.metric("Accuracy", pct((tp + tn) / len(val)))
        b.metric("Baseline accuracy", pct(1 - val.churned.mean()), help="What you'd score by simply predicting that every customer stays.")
        a.metric("Precision", pct(tp / (tp + fp)) if tp + fp else "–", help="Of those we flag, how many really lapse?")
        b.metric("Recall", pct(tp / (tp + fn)), help="Of those who lapse, how many do we catch?")
        st.caption(f"Ranking quality (AUC): **{auc:.2f}**  ·  0.5 = coin flip, 1.0 = perfect")
        hint("Accuracy can look good while the model catches almost nobody. Compare it with the 'nobody churns' accuracy, and look at recall.")
    with c2:
        vv = val.copy()
        vv["Risk decile"] = pd.qcut(vv.p_churn.rank(method="first"), 10, labels=range(1, 11)).astype(int)
        dec = vv.groupby("Risk decile").agg(Predicted=("p_churn", "mean"), Actual=("churned", "mean")).reset_index()
        fig = go.Figure()
        fig.add_bar(x=dec["Risk decile"], y=dec["Actual"], name="Actually lapsed", marker_color=PAL[1])
        fig.add_scatter(x=dec["Risk decile"], y=dec["Predicted"], name="Model's predicted risk", mode="lines+markers", line=dict(color=INK))
        fig.add_hline(y=val.churned.mean(), line_dash="dot", line_color=MUTED, annotation_text="Average", annotation_position="top left")
        fig.update_layout(title="Lapse rate by risk decile (1 = lowest risk, 10 = highest)")
        fig.update_yaxes(tickformat=".0%"); fig.update_xaxes(title="Risk decile", dtick=1)
        st.plotly_chart(style(fig, 360), width="stretch")
        fig = px.bar(drivers.reset_index().rename(columns={"index": "Driver", "coef": "Effect"}), x="Effect", y="Driver",
                     orientation="h", title="What moves the risk score (standardised effect; right = riskier)",
                     color=drivers.values > 0, color_discrete_map={True: PAL[1], False: PAL[0]})
        fig.update_layout(showlegend=False); fig.update_yaxes(title=None); fig.update_xaxes(title=None)
        st.plotly_chart(style(fig, 360), width="stretch")
        hint("These are associations, not levers. Customers with many orders are lower risk; that doesn't mean pushing extra orders onto someone will keep them.")

    st.subheader("Today's customers")
    p = cust.p_churn_6m
    a, b, c = st.columns(3)
    a.metric("Average risk score", pct(p.mean()))
    b.metric(f"Customers flagged at ≥ {thr:.2f}", f"{int((p >= thr).sum()):,}")
    c.metric("12-month CLV held by flagged customers", gbp(CLV_12M[p >= thr].sum()))
    if ss.seg is not None:
        r = pd.DataFrame({"Risk": p, "seg": ss.seg}).groupby("seg").Risk.agg(["mean", lambda s: (s >= thr).mean()])
        r.index = [f"{i} · {ss.seg_names.get(i, '')}" for i in r.index]; r.columns = ["Average risk", "Share flagged"]
        st.dataframe(r.style.format("{:.0%}"), width="stretch")
    hint("The model learned from June–November, the run-up to Christmas. Is the next six months (December–May) likely to look the same for a gift wholesaler?")

# ============================================================================= PLAN
with tabs[4]:
    st.header("④ Plan: where does the money go?")
    c1, c2 = st.columns([1, 3], gap="large")
    with c1:
        st.markdown("**Campaign economics**")
        budget = st.number_input("Budget (£)", 0, 200000, 50000, 5000)
        cut = st.toggle("⚡ Twist: budget cut by 40%")
        if cut:
            budget = int(budget * 0.6)
            st.caption(f"Budget is now **{gbp(budget)}**")
        cost = st.number_input("Cost per customer contacted (£)", 1, 500, 25, help="Account-manager call plus offer.")
        save = st.slider("Save rate", 0, 50, 20, format="%d%%",
                         help="Of customers who WOULD have lapsed, the share the campaign keeps.") / 100
        mg = st.slider("Gross margin", 10, 60, 30, format="%d%%") / 100
        st.caption("Expected gain from contacting a customer = risk × save rate × 12-month CLV × margin")
    risk = cust.p_churn_6m
    gain = risk * save * CLV_12M * mg
    with c2:
        st.markdown("**Value × risk map**")
        a, b = st.columns(2)
        vcut = a.slider("'High value' = top … of customers by 12-month CLV", 10, 90, 50, 5, format="%d%%") / 100
        rcut = b.slider("'High risk' = risk score at least", 0.05, 0.80, float(ss.get("threshold", 0.30)) if ss.get("threshold_chosen") else 0.30, 0.05,
                        help="Defaults to the threshold you agreed in tab ③ (or 0.30 if you haven't).")
        hv = CLV_12M >= CLV_12M.quantile(1 - vcut); hr = risk >= rcut
        quad = pd.Series(np.select([hv & hr, hv & ~hr, ~hv & hr], ["High value · High risk", "High value · Low risk", "Low value · High risk"],
                                   "Low value · Low risk"), index=cust.index)
        qcol = {"High value · High risk": PAL[1], "High value · Low risk": PAL[0], "Low value · High risk": PAL[2], "Low value · Low risk": PAL[5]}
        d = pd.DataFrame({"12-month CLV (£)": CLV_12M, "Risk score": risk, "Quadrant": quad}).sample(2500, random_state=1)
        fig = px.scatter(d, x="Risk score", y="12-month CLV (£)", color="Quadrant", log_y=True, opacity=0.55, color_discrete_map=qcol)
        fig.add_vline(x=rcut, line_dash="dot", line_color=MUTED); fig.add_hline(y=CLV_12M.quantile(1 - vcut), line_dash="dot", line_color=MUTED)
        fig.update_traces(marker=dict(size=5))
        st.plotly_chart(style(fig, 380), width="stretch")
        qt = pd.DataFrame({"CLV": CLV_12M, "risk": risk, "q": quad, "gain": gain}).groupby("q").agg(
            Customers=("CLV", "size"), **{"Avg CLV": ("CLV", "mean")}, **{"Avg risk": ("risk", "mean")}, **{"Expected gain if contacted": ("gain", "sum")})
        qt.index.name = "Quadrant"
        st.dataframe(qt.style.format({"Avg CLV": "£{:,.0f}", "Avg risk": "{:.0%}", "Expected gain if contacted": "£{:,.0f}"}), width="stretch")

    st.markdown("**Who do you contact?**")
    c1, c2 = st.columns([1, 3], gap="large")
    with c1:
        rule = st.radio("Targeting rule", ["Highest CLV first", "Highest risk first", "Highest value at risk (risk × CLV)", "Choose quadrants / segments"])
        opts_q, opts_s = None, None
        if rule == "Choose quadrants / segments":
            opts_q = st.multiselect("Quadrants", list(qcol), default=["High value · High risk"])
            if ss.seg is not None:
                opts_s = st.multiselect("Segments (optional)", list(ss.seg_names), format_func=lambda i: f"{i} · {ss.seg_names[i]}")
            st.caption("Within your selection, customers are contacted in order of expected gain.")
        only_profitable = st.checkbox("Only contact if expected gain > cost")
    order = {"Highest CLV first": CLV_12M, "Highest risk first": risk}.get(rule, gain)
    pool = cust.index
    if rule == "Choose quadrants / segments":
        mask = quad.isin(opts_q or [])
        if opts_s:
            mask &= ss.seg.isin(opts_s)
        pool = cust.index[mask]
    if only_profitable:
        pool = pool[gain.loc[pool] > cost]
    ranked = order.loc[pool].sort_values(ascending=False).index
    n_max = int(budget // cost)
    target = ranked[:n_max]
    spend = len(target) * cost
    exp_gain = gain.loc[target].sum()
    with c2:
        a, b, c, d = st.columns(4)
        a.metric("Customers contacted", f"{len(target):,}", help=f"Budget allows up to {n_max:,}")
        b.metric("Spend", gbp(spend), delta=f"{gbp(budget - spend)} unspent" if budget - spend >= cost else None, delta_color="off")
        c.metric("Expected gain (margin)", gbp(exp_gain))
        d.metric("Expected ROI", pct(exp_gain / spend - 1) if spend else "–")
        cc1, cc2 = st.columns(2)
        with cc1:
            g = gain.loc[target]
            curve_ = pd.DataFrame({"Spend": np.arange(1, len(g) + 1) * cost, "Cumulative expected gain": g.cumsum().to_numpy()})
            fig = px.line(curve_, x="Spend", y="Cumulative expected gain", title="Return as you spend down the list", color_discrete_sequence=[PAL[0]])
            fig.add_scatter(x=[0, max(spend, cost)], y=[0, max(spend, cost)], mode="lines", line=dict(dash="dot", color=PAL[1]), name="Break-even (gain = spend)")
            fig.update_xaxes(tickprefix="£"); fig.update_yaxes(tickprefix="£")
            st.plotly_chart(style(fig, 330), width="stretch")
        with cc2:
            mix = quad.loc[target].value_counts().rename_axis("Quadrant").reset_index(name="Customers")
            fig = px.bar(mix, x="Customers", y="Quadrant", orientation="h", color="Quadrant", color_discrete_map=qcol, title="Who you are contacting")
            fig.update_layout(showlegend=False); fig.update_yaxes(title=None)
            st.plotly_chart(style(fig, 330), width="stretch")
        if ss.seg is not None and len(target):
            sm = ss.seg.loc[target].map(lambda i: f"{i} · {ss.seg_names.get(i, '')}").value_counts()
            st.caption("By segment: " + " · ".join(f"{k_}: {v_:,}" for k_, v_ in sm.items()))
        hint("Check the average risk of who you're contacting. If most of them would have bought anyway, what is the money buying?")
        hint("Where does the curve cross the break-even line? Spending the whole budget is a choice, not a rule.")

        a, b = st.columns([1, 1])
        if a.button("🔒 Lock in this plan", type="primary", width="stretch", disabled=len(target) == 0):
            ss.plan = dict(ids=list(target), rule=rule, budget=budget, cost=cost, save=save, margin=mg, cut=cut,
                           spend=spend, exp_gain=exp_gain, team=team)
            st.success("Plan locked in. You'll see how it did in the Reveal.")
        if len(target):
            out = cust.loc[target, ["country", "frequency", "monetary", "recency_days"]].assign(
                clv_12m=CLV_12M.loc[target].round(0), risk=risk.loc[target].round(3), expected_gain=gain.loc[target].round(2), quadrant=quad.loc[target])
            b.download_button("⬇️ Download contact list (CSV)", out.to_csv().encode(), f"contact_list_{team or 'team'}.csv", width="stretch")

# ============================================================================= REVEAL
with tabs[5]:
    st.header("🔓 Reveal: what actually happened next")
    code = st.text_input("Instructor passcode", type="password")
    try:
        secret = st.secrets.get("REVEAL_CODE", "reveal")
    except Exception:
        secret = "reveal"
    if code != secret:
        st.info("Locked until your instructor opens it at the end of Session 2.")
        st.stop()

    rv = load_reveal().loc[cust.index]
    st.success("Showing what customers actually did over the 12 months after today.")

    st.subheader("1 · Your segments")
    if ss.seg is not None:
        t = pd.DataFrame({"seg": ss.seg, "pred": CLV_12M, "bought": rv.bought_next_12m, "rev": rv.revenue_next_12m}).groupby("seg").agg(
            Customers=("pred", "size"), **{"Bought again (next 12 months)": ("bought", "mean")}, **{"Predicted 12-month CLV (avg)": ("pred", "mean")},
            **{"Actual revenue, next 12 months (avg)": ("rev", "mean")})
        t["Prediction error"] = t["Predicted 12-month CLV (avg)"] / t["Actual revenue, next 12 months (avg)"] - 1
        t.index = [f"{i} · {ss.seg_names.get(i, '')}" for i in t.index]
        st.dataframe(t.style.format({"Bought again (next 12 months)": "{:.0%}", "Predicted 12-month CLV (avg)": "£{:,.0f}", "Actual revenue, next 12 months (avg)": "£{:,.0f}",
                                     "Prediction error": "{:+.0%}"}), width="stretch")
    else:
        st.caption("No segments built in this browser session.")

    st.subheader("2 · The CLV model")
    a, b, c, d = st.columns(4)
    a.metric("Predicted revenue, next 12 months", gbp(CLV_12M.sum()))
    b.metric("Actual revenue, next 12 months", gbp(rv.revenue_next_12m.sum()), delta=f"{CLV_12M.sum() / rv.revenue_next_12m.sum() - 1:+.0%} over-predicted", delta_color="off")
    top10 = CLV_12M.rank(ascending=False, method="first") <= N / 10
    naive = cust.monetary.rank(ascending=False, method="first") <= N / 10
    c.metric("Actual revenue from model's top 10%", pct(rv.revenue_next_12m[top10].sum() / rv.revenue_next_12m.sum()))
    d.metric("…from last year's top 10% spenders", pct(rv.revenue_next_12m[naive].sum() / rv.revenue_next_12m.sum()))
    band = pd.cut(cust.tenure_days, [-1, 90, 180, 270, 366], labels=["New (0–3 months)", "3–6 months", "6–9 months", "9–12 months"])
    tb = pd.DataFrame({"band": band, "Predicted": CLV_12M, "Actual": rv.revenue_next_12m}).groupby("band", observed=True)[["Predicted", "Actual"]].sum()
    fig = px.bar(tb.reset_index().melt(id_vars="band", var_name="", value_name="Revenue"), x="band", y="Revenue", color="", barmode="group",
                 color_discrete_sequence=[PAL[5], PAL[0]], title="Predicted vs actual revenue over the next 12 months, by how long the customer had been with us")
    fig.update_xaxes(title=None); fig.update_yaxes(tickprefix="£")
    st.plotly_chart(style(fig, 360), width="stretch")
    st.caption(f"Plus **{meta['new_customers_2011']:,} brand-new customers** arrived during those 12 months. No CLV of the existing base includes them.")

    st.subheader("3 · The churn model")
    a, b, c = st.columns(3)
    a.metric("Average predicted risk", pct(cust.p_churn_6m.mean()))
    b.metric("Actually lapsed (next 6 months)", pct(rv.lapsed_next_6m.mean()))
    c.metric("Ranking quality (AUC) on actual outcomes", f"{roc_auc_score(rv.lapsed_next_6m, cust.p_churn_6m):.2f}")
    dd = pd.DataFrame({"p": cust.p_churn_6m, "y": rv.lapsed_next_6m})
    dd["Risk decile"] = pd.qcut(dd.p.rank(method="first"), 10, labels=range(1, 11)).astype(int)
    dec = dd.groupby("Risk decile").agg(Predicted=("p", "mean"), Actual=("y", "mean")).reset_index()
    fig = go.Figure()
    fig.add_bar(x=dec["Risk decile"], y=dec["Actual"], name="Actually lapsed", marker_color=PAL[1])
    fig.add_scatter(x=dec["Risk decile"], y=dec["Predicted"], name="Predicted risk", mode="lines+markers", line=dict(color=INK))
    fig.update_yaxes(tickformat=".0%"); fig.update_xaxes(title="Risk decile", dtick=1)
    fig.update_layout(title="The ranking held. The level did not.")
    st.plotly_chart(style(fig, 340), width="stretch")

    st.subheader("4 · Your plan vs the standard playbooks")
    p_ = ss.plan
    B, CST, SV, MG = (p_["budget"], p_["cost"], p_["save"], p_["margin"]) if p_ else (50000, 25, 0.20, 0.30)
    g_ = cust.p_churn_6m * SV * CLV_12M * MG
    n_ = int(B // CST)

    def score(ids, name):
        ids = pd.Index(ids); lapsed = rv.lapsed_next_6m.loc[ids]
        planned = g_.loc[ids].sum(); real = SV * MG * CLV_12M.loc[ids][lapsed == 1].sum(); spent = len(ids) * CST
        return {"Plan": name, "Contacted": len(ids), "Spend": spent, "Planned ROI": planned / spent - 1 if spent else np.nan,
                "Lapsed (contact could matter)": lapsed.mean(), "Bought anyway": 1 - lapsed.mean(),
                "Hindsight ROI": real / spent - 1 if spent else np.nan}
    rows = []
    if p_:
        rows.append(score(p_["ids"], f"⭐ {p_.get('team') or 'Your team'}: {p_['rule']}"))
    rows += [score(CLV_12M.sort_values(ascending=False).index[:n_], "Highest CLV first"),
             score(cust.p_churn_6m.sort_values(ascending=False).index[:n_], "Highest risk first"),
             score(g_.sort_values(ascending=False).index[:n_], "Highest value at risk"),
             score(cust.sample(n_, random_state=1).index, "Random")]
    st.dataframe(pd.DataFrame(rows).style.format({"Spend": "£{:,.0f}", "Planned ROI": "{:+.0%}", "Lapsed (contact could matter)": "{:.0%}",
                                                  "Bought anyway": "{:.0%}", "Hindsight ROI": "{:+.0%}"}), hide_index=True, width="stretch")
    st.caption(f"Assumptions: budget {gbp(B)}, £{CST} per contact, {int(SV * 100)}% save rate, {int(MG * 100)}% margin. "
               "Hindsight ROI swaps the model's risk for what customers actually did, but still assumes the save rate. "
               "Without a control group, nobody can know the true save rate.")
