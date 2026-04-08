import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# -----------------------------
# 🔥 SIDEBAR (OUTSIDE TABS)
# -----------------------------
st.sidebar.title('Simulation Controls')

uname = st.sidebar.text_input('Enter your name:')
immune = st.sidebar.slider("Immune Attack", 0.0, 1.0, 0.3)
stress = st.sidebar.slider("Oxidative Stress", 0.0, 1.0, 0.2)
treatment = st.sidebar.slider("Treatment Strength", 0.0, 1.0, 0.4)
start = st.sidebar.slider("Treatment Start Time", 0, 100, 30)

# -----------------------------
# HEADER
# -----------------------------
st.title("🧬 Vitiligo AI Simulator")
st.write("Simulation of vitiligo progression using ML + stochastic modeling")

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
# BACKGROUND
# -----------------------------
with tab1:
    st.subheader("Background")
    st.write("""
Vitiligo is a skin disorder caused by loss of melanocytes leading to white patches. 
It is influenced by immune response, oxidative stress, and genetic factors.
""")

# -----------------------------
# OBJECTIVE
# -----------------------------
with tab2:
    st.subheader("Objective")
    st.write("""
To simulate vitiligo progression and predict outcomes using machine learning.
""")

# -----------------------------
# WORKING
# -----------------------------
with tab3:
    st.subheader("Working")
    st.write("""
- Immune + stress decrease melanocytes  
- Treatment helps recovery  
- Random noise mimics biological variation  
- ML predicts outcome  
- Clustering categorizes severity  
""")

# -----------------------------
# APPLICATIONS
# -----------------------------
with tab4:
    st.subheader("Applications")
    st.write("""
- Disease modeling  
- Educational tool  
- ML demonstration  
- Research use  
""")

# -----------------------------
# TEAM
# -----------------------------
with tab5:
    st.subheader("Team")
    st.write("""
- Disha Thorat  
(Add others if any)
""")

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
# SIMULATOR TAB 🔥
# -----------------------------
with tab6:

    st.subheader(f"Welcome {uname if uname else 'User'}")

    # GRAPH
    st.subheader("Simulation Results")

    np.random.seed(42)  # FIXED GRAPH
    data = simulate(100, immune, stress, treatment, start)

    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Melanocyte Level (%)")
    ax.set_title("Disease Progression")

    st.pyplot(fig)

    # GRAPH COMMENT ✅
    st.caption("Downward trend = depigmentation | Stable = treatment effect | Increase = recovery")

    # -----------------------------
    # ML MODEL
    # -----------------------------
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
    # RISK SCORE
    # -----------------------------
    risk_score = (immune*0.4 + stress*0.3 + (1-treatment)*0.3)*100

    # -----------------------------
    # RESULTS
    # -----------------------------
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
