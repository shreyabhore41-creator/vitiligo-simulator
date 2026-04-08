import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import pandas as pd

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title('Vitiligo Simulator')
st.sidebar.write('Adjust parameters below')

uname = st.sidebar.text_input('Enter your name:')

immune = st.sidebar.slider("Immune Attack", 0.0, 1.0, 0.3)
stress = st.sidebar.slider("Oxidative Stress", 0.0, 1.0, 0.2)
treatment = st.sidebar.slider("Treatment Strength", 0.0, 1.0, 0.4)
start = st.sidebar.slider("Treatment Start Time", 0, 100, 30)

notify = st.sidebar.checkbox('Enable notifications')

# -----------------------------
# HEADER
# -----------------------------
st.header(f"Welcome, {uname if uname else 'User'}")
st.write("Vitiligo progression simulation using ML + stochastic modeling")

if notify:
    st.success("Notifications enabled")

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
# HORIZONTAL TABS
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧬 Background", 
    "🎯 Objective", 
    "⚙️ Working", 
    "🧪 Applications", 
    "👩‍💻 Team"
])

# -----------------------------
# TAB 1: Background
# -----------------------------
with tab1:
    st.header("Background")
    st.write("""
Vitiligo is a skin condition characterized by loss of melanocytes, leading to depigmented patches. 
Understanding disease progression and treatment impact is challenging due to variability among patients.
This tool simulates vitiligo progression using a combination of stochastic modeling and machine learning.
    """)
    st.write("You can adjust immune attack, oxidative stress, treatment strength, and start time to simulate outcomes.")

# -----------------------------
# TAB 2: Objective
# -----------------------------
with tab2:
    st.header("Objective")
    st.write("""
- Predict the progression of melanocyte loss over time.
- Evaluate the effect of treatment timing and strength.
- Categorize patients into progression groups.
- Provide educational insights and risk scores.
- Generate downloadable PDF reports for records.
    """)

# -----------------------------
# TAB 3: Working
# -----------------------------
with tab3:
    st.header("Working")

    # Patient Profile
    st.subheader('Patient Profile')
    col1, col2 = st.columns(2)
    with col1:
        fname = st.text_input('First name')
    with col2:
        lname = st.text_input('Last name')

    # Simulation
    st.subheader("Simulation Results")
    data = simulate(100, immune, stress, treatment, start)
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Melanocyte Level (%)")
    ax.set_title("Disease Progression")
    st.pyplot(fig)
    st.caption("Downward trend = melanocyte loss. Stabilization = treatment effect.")

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

# -----------------------------
# TAB 4: Applications
# -----------------------------
with tab4:
    st.header("Applications")

    # ML Model
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

    # Clustering
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(X)
    cluster = kmeans.predict([[immune, stress, treatment, start]])[0]
    cluster_map = {0: "Slow progression group", 1: "Moderate progression group", 2: "Aggressive progression group"}
    st.subheader("Patient Category")
    st.write(cluster_map.get(cluster, "Unknown"))

    # Feature importance
    features = ['Immune', 'Stress', 'Treatment', 'Start']
    importance = model.feature_importances_
    df = pd.DataFrame({'Feature':features, 'Importance':importance})
    st.bar_chart(df.set_index('Feature'))

# -----------------------------
# TAB 5: Team
# -----------------------------
with tab5:
    st.header("Team & Reporting")
    st.write("""
- **Data Scientists**: Built the simulation and ML models.
- **Biologists**: Provided domain knowledge on vitiligo and melanocyte behavior.
- **ML Engineers**: Optimized prediction and clustering pipelines.
- **UI/UX Designers**: Ensured clarity and usability of the tool.
""")

    # PDF Download
    def create_pdf(text):
        doc = SimpleDocTemplate("report.pdf")
        styles = getSampleStyleSheet()
        content = []
        for line in text.split("\n"):
            content.append(Paragraph(line, styles["Normal"]))
            content.append(Spacer(1, 10))
        doc.build(content)

    report_text = f"""
Name: {fname} {lname}
Predicted Level: {pred:.2f}
Risk Score: {(immune * 0.4 + stress * 0.3 + (1 - treatment) * 0.3)*100:.2f}
Cluster: {cluster_map.get(cluster)}
"""
    create_pdf(report_text)
    with open("report.pdf", "rb") as f:
        st.download_button("📄 Download Report", f, file_name="vitiligo_report.pdf")

st.caption("This tool is for educational purposes only and not for medical diagnosis.")
