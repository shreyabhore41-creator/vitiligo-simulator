import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import pandas as pd

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# -----------------------------
# HEADER
# -----------------------------
st.title("🧬 Vitiligo AI Simulator")
st.write("ML-based simulation of vitiligo progression")

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🧬 Background", 
    "🎯 Objective", 
    "⚙️ Working", 
    "🧪 Applications", 
    "👩‍💻 Team",
    "🧪 Simulator"
])

# -----------------------------
# 🧬 BACKGROUND
# -----------------------------
with tab1:
    st.subheader("Background of Vitiligo")

    st.write("""
Vitiligo is a chronic skin disorder characterized by the loss of melanocytes, 
which are responsible for producing melanin, the pigment that gives color to the skin.

This leads to white patches appearing on different parts of the body. The exact cause is not fully understood, 
but it is associated with autoimmune responses, oxidative stress, and genetic factors.

Understanding vitiligo progression is important for predicting disease severity and evaluating treatment strategies.
    """)

# -----------------------------
# 🎯 OBJECTIVE
# -----------------------------
with tab2:
    st.subheader("Objective")

    st.write("""
The objective of this project is to simulate the progression of vitiligo using computational modeling.

This tool aims to:
- Predict melanocyte loss over time
- Analyze the impact of immune activity and oxidative stress
- Evaluate treatment effectiveness
- Apply machine learning for outcome prediction
    """)

# -----------------------------
# ⚙️ WORKING
# -----------------------------
with tab3:
    st.subheader("How the Tool Works")

    st.write("""
The system uses a stochastic simulation model to represent melanocyte dynamics.

- Immune attack and oxidative stress reduce melanocyte levels
- Treatment helps in recovery
- Random noise is added to mimic biological variability
- A Random Forest model predicts final melanocyte levels
- K-means clustering categorizes disease progression patterns
    """)

# -----------------------------
# 🧪 APPLICATIONS
# -----------------------------
with tab4:
    st.subheader("Applications")

    st.write("""
- Educational tool for dermatology studies
- Demonstration of machine learning in healthcare
- Visualization of disease progression
- Useful for bioinformatics learning and research
    """)

# -----------------------------
# 👩‍💻 TEAM
# -----------------------------
with tab5:
    st.subheader("Team")

    st.write("""
- Disha Thorat  
- Shreya Bhore

This project is developed as part of academic coursework in bioinformatics.
    """)

# -----------------------------
# 🧪 SIMULATOR TAB
# -----------------------------
with tab6:

    # SIDEBAR
    st.sidebar.title('Simulation Controls')

    uname = st.sidebar.text_input('Enter your name:')
    immune = st.sidebar.slider("Immune Attack", 0.0, 1.0, 0.3)
    stress = st.sidebar.slider("Oxidative Stress", 0.0, 1.0, 0.2)
    treatment = st.sidebar.slider("Treatment Strength", 0.0, 1.0, 0.4)
    start = st.sidebar.slider("Treatment Start Time", 0, 100, 30)

    st.subheader(f"Welcome {uname if uname else 'User'}")

    # -----------------------------
    # SIMULATION FUNCTION
    # -----------------------------
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

    # -----------------------------
    # GRAPH
    # -----------------------------
    st.subheader("Simulation Results")

    np.random.seed(42)
    data = simulate(100, immune, stress, treatment, start)

    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_title("Disease Progression")

    st.pyplot(fig)

    # -----------------------------
    # ML MODEL
    # -----------------------------
    X, y = [], []

    for _ in range(150):
        i, s, tr = np.random.rand(3)
        stt = np.random.randint(0,100)
        sim = simulate(100, i, s, tr, stt)
        X.append([i, s, tr, stt])
        y.append(sim[-1])

    model = RandomForestRegressor()
    model.fit(X, y)

    pred = model.predict([[immune, stress, treatment, start]])[0]

    # -----------------------------
    # CLUSTERING
    # -----------------------------
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(X)

    cluster = kmeans.predict([[immune, stress, treatment, start]])[0]

    cluster_map = {
        0: "Slow progression",
        1: "Moderate progression",
        2: "Aggressive progression"
    }

    cluster_text = cluster_map[cluster]

    # -----------------------------
    # RISK
    # -----------------------------
    risk_score = (immune*0.4 + stress*0.3 + (1-treatment)*0.3)*100

    st.subheader("Results")
    st.write(f"Melanocyte Level: {pred:.2f}")
    st.write(f"Risk Score: {risk_score:.2f}")
    st.write(f"Category: {cluster_text}")

    # -----------------------------
    # PDF
    # -----------------------------
    def create_pdf():
        fig.savefig("graph.png")

        doc = SimpleDocTemplate("report.pdf")
        styles = getSampleStyleSheet()
        content = []

        content.append(Paragraph("<b>Vitiligo Simulation Report</b>", styles["Title"]))
        content.append(Spacer(1, 15))
        content.append(Paragraph(f"Prediction: {pred:.2f}", styles["Normal"]))
        content.append(Paragraph(f"Risk: {risk_score:.2f}", styles["Normal"]))
        content.append(Paragraph(f"Category: {cluster_text}", styles["Normal"]))
        content.append(Spacer(1, 15))
        content.append(Image("graph.png", width=5*inch, height=3*inch))

        doc.build(content)

    create_pdf()

    with open("report.pdf", "rb") as f:
        st.download_button("📄 Download Report", f)
