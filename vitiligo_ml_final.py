import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import pandas as pd

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧬 Background",
    "🎯 Objective",
    "⚙️ Working",
    "🧪 Applications",
    "👩‍💻 Team"
])

# -----------------------------
# TAB 1: BACKGROUND
# -----------------------------
with tab1:
    st.header("Background")
    st.write("""
Vitiligo is a skin condition where melanocytes are destroyed, leading to white patches.
This tool simulates disease progression using stochastic modeling and machine learning.
""")

# -----------------------------
# TAB 2: OBJECTIVE
# -----------------------------
with tab2:
    st.header("Objective")
    st.write("""
- Simulate melanocyte loss
- Analyze treatment timing
- Predict disease progression
- Provide risk scoring
- Generate insights
""")

# -----------------------------
# TAB 3: WORKING (YOUR FULL CODE)
# -----------------------------
with tab3:

    # SIDEBAR
    st.sidebar.title('Vitiligo Simulator')
    st.sidebar.write('Adjust parameters below')

    uname = st.sidebar.text_input('Enter your name:')

    immune = st.sidebar.slider("Immune Attack", 0.0, 1.0, 0.3)
    stress = st.sidebar.slider("Oxidative Stress", 0.0, 1.0, 0.2)
    treatment = st.sidebar.slider("Treatment Strength", 0.0, 1.0, 0.4)
    start = st.sidebar.slider("Treatment Start Time", 0, 100, 30)

    notify = st.sidebar.checkbox('Enable notifications')

    # HEADER
    st.header(f"Welcome, {uname if uname else 'User'}")
    st.write("Vitiligo progression simulation using ML + stochastic modeling")

    if notify:
        st.success("Notifications enabled")

    # SIMULATION FUNCTION
    def simulate(t, i, s, tr, stt):
        mel = [100]
        for step in range(1, t):
            prev = mel[-1]
            treat = tr if step >= stt else 0
            noise = np.random.normal(0, 0.5)
            new = prev - (i + s) + treat + noise
            new = max(min(new, 100), 0)
            mel.append(new)
        return mel

    # USER PROFILE
    st.subheader('Patient Profile')
    col1, col2 = st.columns(2)
    with col1:
        fname = st.text_input('First name')
    with col2:
        lname = st.text_input('Last name')

    # SIMULATION RESULTS
    st.subheader("Simulation Results")
    data = simulate(100, immune, stress, treatment, start)

    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Melanocyte Level (%)")
    ax.set_title("Disease Progression")
    st.pyplot(fig)

    st.caption("Downward trend = melanocyte loss (depigmentation). Stabilization = treatment effect.")

    # COMPARISON
    st.subheader("Treatment Comparison")
    early = simulate(100, immune, stress, treatment, 10)
    late = simulate(100, immune, stress, treatment, 60)

    fig2, ax2 = plt.subplots()
    ax2.plot(early, label="Early Treatment")
    ax2.plot(late, label="Late Treatment")
    ax2.legend()
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Melanocyte Level (%)")
    ax2.set_title("Effect of Treatment Timing")
    st.pyplot(fig2)

    # ML MODEL
    X, y = [], []
    for _ in range(150):
        i = np.random.uniform(0,1)
        s = np.random.uniform(0,1)
        tr = np.random.uniform(0,1)
        stt = np.random.randint(0,100)
        sim = simulate(100, i, s, tr, stt)
        X.append([i, s, tr, stt])
        y.append(sim[-1])

    model = RandomForestRegressor()
    model.fit(X, y)

    pred = model.predict([[immune, stress, treatment, start]])[0]

    st.subheader("Predicted Outcome")
    st.write(f"Final Melanocyte Level: {pred:.2f}")
    st.caption("Higher value = healthier pigmentation, lower = more depigmentation")

    # CLUSTERING
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(X)
    cluster = kmeans.predict([[immune, stress, treatment, start]])[0]

    cluster_map = {
        0: "Slow progression group",
        1: "Moderate progression group",
        2: "Aggressive progression group"
    }

    st.subheader("Patient Category")
    st.write(cluster_map.get(cluster, "Unknown"))

    # ANALYSIS
    st.subheader("Analysis")
    if pred > 70:
        st.success("Mild condition")
    elif pred > 40:
        st.warning("Moderate progression")
    else:
        st.error("Severe condition")

    # SMART INTERPRETATION
    st.subheader("🧠 Smart Interpretation")
    if immune > 0.7 and stress > 0.6:
        st.warning("High immune attack and oxidative stress may accelerate depigmentation.")
    elif treatment > 0.6 and start < 30:
        st.success("Early and strong treatment is helping control disease progression.")
    elif stress > 0.6:
        st.info("Oxidative stress is a key contributing factor.")
    else:
        st.info("Condition appears relatively stable.")

    # RISK SCORE
    st.subheader("🎯 Risk Score")
    risk_score = (immune * 0.4 + stress * 0.3 + (1 - treatment) * 0.3) * 100
    st.write(f"Risk Score: {risk_score:.2f} / 100")

    # WHAT-IF
    st.subheader("🔮 What-If Analysis")
    reduced_stress = max(stress - 0.2, 0)
    new_pred = model.predict([[immune, reduced_stress, treatment, start]])[0]
    st.write(f"If stress is reduced, melanocyte level improves by **{new_pred - pred:.2f}**")

    # FEATURE IMPORTANCE
    features = ['Immune', 'Stress', 'Treatment', 'Start']
    importance = model.feature_importances_
    df = pd.DataFrame({'Feature':features, 'Importance':importance})
    st.bar_chart(df.set_index('Feature'))

# -----------------------------
# TAB 4: APPLICATIONS
# -----------------------------
with tab4:
    st.header("Applications")
    st.write("""
- Educational tool for understanding vitiligo  
- Research simulation platform  
- Treatment strategy comparison  
- Medical training aid  
""")

# -----------------------------
# TAB 5: TEAM
# -----------------------------
with tab5:
    st.header("Team Members")
    st.write("""
- Disha Thorat  
- Shreya Bhore 
""")
