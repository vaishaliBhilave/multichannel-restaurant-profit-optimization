import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Restaurant Profit AI System", layout="wide")
st.title("🍽️ Multi-Channel Restaurant Profit Intelligence System")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("SkyCity Auckland Restaurants & Bars.csv")

df = load_data()

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df["TotalRevenue"] = df["InStoreRevenue"] + df["UberEatsRevenue"] + df["DoorDashRevenue"] + df["SelfDeliveryRevenue"]

df["TotalNetProfit"] = df["InStoreNetProfit"] + df["UberEatsNetProfit"] + df["DoorDashNetProfit"] + df["SelfDeliveryNetProfit"]

df["ProfitPerOrder"] = df["TotalNetProfit"] / df["MonthlyOrders"]

df["Commission_UE_Interaction"] = df["CommissionRate"] * df["UE_share"]
df["DeliveryCost_SD_Interaction"] = df["DeliveryCostPerOrder"] * df["SD_share"]

# -------------------------------
# OUTLIER HANDLING
# -------------------------------
def winsorize(df, cols):
    df_copy = df.copy()
    for col in cols:
        l = df_copy[col].quantile(0.01)
        u = df_copy[col].quantile(0.99)
        df_copy[col] = np.clip(df_copy[col], l, u)
    return df_copy

df = winsorize(df, ["AOV","MonthlyOrders","CommissionRate","DeliveryCostPerOrder","TotalNetProfit","ProfitPerOrder"])

# -------------------------------
# MODEL TRAINING
# -------------------------------
@st.cache_resource
def train_model(df):

    num = ["AOV","MonthlyOrders","GrowthFactor",
           "CommissionRate","DeliveryCostPerOrder",
           "InStoreShare","UE_share","DD_share","SD_share",
           "ProfitPerOrder","Commission_UE_Interaction","DeliveryCost_SD_Interaction"]

    cat = ["CuisineType","Segment","Subregion"]

    X = df[num + cat]
    y = df["TotalNetProfit"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("rf", RandomForestRegressor(n_estimators=150, random_state=42))
    ])

    model.fit(X, y)
    return model, X, y

model, X, y = train_model(df)

# -------------------------------
# TRAIN-TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)

# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
st.sidebar.header("⚙️ Controls")

ue = st.sidebar.slider("UE Share", 0.0, 1.0, 0.3)
dd = st.sidebar.slider("DD Share", 0.0, 1.0, 0.3)
sd = st.sidebar.slider("Self Delivery", 0.0, 1.0, 0.2)

instore = max(0.0, 1 - (ue + dd + sd))

commission = st.sidebar.slider("Commission Rate", 0.05, 0.5, 0.25)
delivery_cost = st.sidebar.slider("Delivery Cost", 0.5, 6.0, 2.0)

aov = st.sidebar.slider("AOV", 20.0, 60.0, 35.0)
orders = st.sidebar.slider("Monthly Orders", 100, 5000, 1000)
growth = st.sidebar.slider("Growth Factor", 0.9, 1.1, 1.0)

# -------------------------------
# INPUT DATA
# -------------------------------
input_data = pd.DataFrame([{
    "AOV": aov,
    "MonthlyOrders": orders,
    "GrowthFactor": growth,
    "CommissionRate": commission,
    "DeliveryCostPerOrder": delivery_cost,
    "InStoreShare": instore,
    "UE_share": ue,
    "DD_share": dd,
    "SD_share": sd,
    "CuisineType": df["CuisineType"].mode()[0],
    "Segment": df["Segment"].mode()[0],
    "Subregion": df["Subregion"].mode()[0],
    "ProfitPerOrder": 0,
    "Commission_UE_Interaction": commission * ue,
    "DeliveryCost_SD_Interaction": delivery_cost * sd
}])

# -------------------------------
# PREDICTION
# -------------------------------
pred = model.predict(input_data)[0]
input_data["ProfitPerOrder"] = pred / orders

# -------------------------------
# CONFIDENCE + RISK (FIXED)
# -------------------------------
trees = model.named_steps["rf"].estimators_
X_trans = model.named_steps["preprocessor"].transform(input_data)
tree_preds = [t.predict(X_trans)[0] for t in trees]

lower = np.percentile(tree_preds, 10)
upper = np.percentile(tree_preds, 90)

risk = np.std(tree_preds)

# -------------------------------
# SENSITIVITY (FIXED)
# -------------------------------
def sensitivity(feature, delta=0.05):
    temp = input_data.copy()

    base_val = temp[feature].iloc[0]
    change = base_val * delta

    plus = temp.copy()
    minus = temp.copy()

    plus[feature] += change
    minus[feature] -= change

    # update interactions
    plus["Commission_UE_Interaction"] = plus["CommissionRate"] * plus["UE_share"]
    minus["Commission_UE_Interaction"] = minus["CommissionRate"] * minus["UE_share"]

    plus["DeliveryCost_SD_Interaction"] = plus["DeliveryCostPerOrder"] * plus["SD_share"]
    minus["DeliveryCost_SD_Interaction"] = minus["DeliveryCostPerOrder"] * minus["SD_share"]

    return (model.predict(plus)[0] - model.predict(minus)[0]) / (2 * change)

# -------------------------------
# KPI 3
# -------------------------------
df["ChannelEfficiency"] = df["TotalNetProfit"] / df["TotalRevenue"]

# -------------------------------
# KPI DISPLAY (UNCHANGED)
# -------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 Predicted Net Profit")
    st.metric("Profit", f"${pred:,.2f}")

with col2:
    st.subheader("📉 Risk / Confidence Band")
    st.metric("📉 Confidence Band (Lower)", f"${lower:,.2f}")
    st.metric("📉 Confidence Band (Upper)", f"${upper:,.2f}")

with col3:
    st.subheader("📌 Profit Sensitivity Index")
    st.write("Commission Sensitivity:", sensitivity("CommissionRate"))

# -------------------------------
# MORE KPIs
# -------------------------------
col4, col5, col6 = st.columns(3)

with col4:
    st.subheader("📊 Channel Efficiency")
    st.write(df["ChannelEfficiency"].mean())

with col5:
    def break_even():
        rates = np.linspace(0.05, 0.6, 20)
        profits = []

        for r in rates:
            temp = input_data.copy()
            temp["CommissionRate"] = r

            # FIX
            temp["Commission_UE_Interaction"] = r * temp["UE_share"]

            profits.append(model.predict(temp)[0])

        idx = np.argmin(np.abs(profits))
        return rates[idx]

    st.subheader("⚖️ Break-Even Commission Rate")
    st.write(break_even())

with col6:
    baseline = df["TotalNetProfit"].mean()
    uplift = ((pred - baseline) / baseline) * 100

    st.subheader("🚀 Optimization Uplift (%)")
    st.write(uplift)

# -------------------------------
# PLOTS (UNCHANGED)
# -------------------------------
st.subheader("📊 Actual vs Predicted")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.5)
st.pyplot(fig)

st.subheader("📊 Channel Mix")
fig, ax = plt.subplots()
ax.bar(["UE", "DD", "SD", "InStore"], [ue, dd, sd, instore])
st.pyplot(fig)

# -------------------------------
# SENSITIVITY CURVE (FIXED)
# -------------------------------
st.subheader("📉 Commission Sensitivity Curve")

rates = np.linspace(0.1, 0.5, 10)
vals = []

for r in rates:
    temp = input_data.copy()
    temp["CommissionRate"] = r

    # FIX
    temp["Commission_UE_Interaction"] = r * temp["UE_share"]

    vals.append(model.predict(temp)[0])

fig, ax = plt.subplots()
ax.plot(rates, vals)
st.pyplot(fig)

# -------------------------------
# OPTIMIZATION (FIXED)
# -------------------------------
st.subheader("🏆 Best Channel Mix")

best = -np.inf
best_cfg = None

for u in np.linspace(0.1, 0.7, 5):
    for s in np.linspace(0.1, 0.7, 5):
        d = 1 - u - s
        if d < 0:
            continue

        temp = input_data.copy()

        temp["UE_share"] = u
        temp["DD_share"] = d
        temp["SD_share"] = s
        temp["InStoreShare"] = max(0, 1 - (u + d + s))

        # FIX
        temp["Commission_UE_Interaction"] = temp["CommissionRate"] * u
        temp["DeliveryCost_SD_Interaction"] = temp["DeliveryCostPerOrder"] * s

        temp["ProfitPerOrder"] = pred / orders

        p = model.predict(temp)[0]

        if p > best:
            best = p
            best_cfg = (round(temp["InStoreShare"].iloc[0],2), round(u,2), round(d,2), round(s,2))

if best_cfg:
    st.write("Best (InStore, UE, DD, SD):", best_cfg)
    st.write("Max Profit:", f"{best:,.2f}")
else:
    st.error("No valid configuration found")

# -------------------------------
# RISK DISPLAY (FIXED LOGIC ONLY)
# -------------------------------
st.subheader("⚠️ Strategy Risk Level")
st.write("Risk Indicator:", f"{risk:,.2f}")

if risk > 300:
    st.error("High Risk Strategy")
elif risk > 150:
    st.warning("Moderate Risk Strategy")
else:
    st.success("Stable Strategy")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.write("© 2026 Full Predictive + Prescriptive Restaurant Intelligence System. All rights reserved.")    
st.write("Developed by Vaishali.")