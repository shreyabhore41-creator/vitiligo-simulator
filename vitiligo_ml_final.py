import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import pandas as pd

# ✅ NEW PDF IMPORTS
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

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
# 💅 CRAZY UI
# -----------------------------
st.markdown("""
<style>
.stApp {
    background-image: linear-gradient(rgba(255,255,255,0.6), rgba(255,255,255,0.6)),
    url("https://images.unsplash.com/photo-1526045612212-70caf35c14df");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
.block-container {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(15px);
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.1);
}
h1, h2, h3 {
    color: #2c3e50;
    font-weight: 700;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fdfbfb 0%, #ebedee 100%);
}
.stButton button {
    background: linear-gradient(135deg, #a084ee, #6c63ff);
    color: white;
    border-radius: 12px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

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

# -----------------------------
# HEADER
# -----------------------------
st.markdown("## 🧬 Vitiligo AI Simulator")
st.caption("ML-powered dermatology simulation tool")
st.markdown("---")

# -----------------------------
# SIMULATION
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
# PROFILE
# -----------------------------
st.subheader("Patient Profile")
col1, col2 = st.columns(2)
fname = col1.text_input("First Name")
lname = col2.text_input("Last Name")

# -----------------------------
# GRAPH
# -----------------------------
st.markdown("---")
st.subheader("Simulation Results")

data = simulate(100, immune, stress, treatment, start)

fig, ax = plt.subplots()
ax.plot(data)
ax.set_xlabel("Time")
ax.set_ylabel("Melanocyte Level")
ax.set_title("Disease Progression")

st.pyplot(fig)
st.caption("Downward = depigmentation, stable = treatment effect")

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
# RISK SCORE
# -----------------------------
risk_score = (immune*0.4 + stress*0.3 + (1-treatment)*0.3)*100

# -----------------------------
# METRICS 💅
# -----------------------------
st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.metric("Melanocyte Level", f"{pred:.2f}")
col2.metric("Risk Score", f"{risk_score:.1f}")
col3.metric("Cluster", cluster_text)

# -----------------------------
# PDF FUNCTION 🔥
# -----------------------------
def create_pdf():
    fig.savefig("graph.png")

    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("<b>Vitiligo Simulation Report</b>", styles["Title"]))
    content.append(Spacer(1, 15))

    name = f"{fname} {lname}".strip()
    content.append(Paragraph(f"<b>Name:</b> {name}", styles["Normal"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"<b>Predicted Level:</b> {pred:.2f}", styles["Normal"]))
    content.append(Paragraph(f"<b>Risk Score:</b> {risk_score:.2f}", styles["Normal"]))
    content.append(Paragraph(f"<b>Cluster:</b> {cluster_text}", styles["Normal"]))

    content.append(Spacer(1, 15))
    content.append(Image("graph.png", width=5*inch, height=3*inch))

    doc.build(content)

create_pdf()

with open("report.pdf", "rb") as f:
    st.download_button("📄 Download Report", f)

# -----------------------------
# DISCLAIMER
# -----------------------------
st.caption("For educational purposes only")

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
# USER PROFILE
# -----------------------------
st.subheader('Patient Profile')
col1, col2 = st.columns(2)

with col1:
    fname = st.text_input('First name')
with col2:
    lname = st.text_input('Last name')

# -----------------------------
# SIMULATION GRAPH
# -----------------------------
st.subheader("Simulation Results")

data = simulate(100, immune, stress, treatment, start)

fig, ax = plt.subplots()
ax.plot(data)
ax.set_xlabel("Time Steps")
ax.set_ylabel("Melanocyte Level (%)")
ax.set_title("Disease Progression")

st.pyplot(fig)

st.caption("Downward trend = melanocyte loss (depigmentation). Stabilization = treatment effect.")

# -----------------------------
# COMPARISON
# -----------------------------
st.subheader("Treatment Comparison")

early = simulate(100, immune, stress, treatment, 10)
late = simulate(100, immune, stress, treatment, 60)

fig2, ax2 = plt.subplots()
ax2.plot(early, label="Early Treatment")
ax2.plot(late, label="Late Treatment")
ax2.legend()

st.pyplot(fig2)

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

st.subheader("Predicted Outcome")
st.write(f"Final Melanocyte Level: {pred:.2f}")
st.caption("Higher value = healthier pigmentation, lower = more depigmentation")

# -----------------------------
# CLUSTERING
# -----------------------------
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

cluster = kmeans.predict([[immune, stress, treatment, start]])[0]

cluster_map = {
    0: "Slow progression group",
    1: "Moderate progression group",
    2: "Aggressive progression group"
}

cluster_text = cluster_map.get(cluster)

st.subheader("Patient Category")
st.write(cluster_text)

# -----------------------------
# RISK SCORE
# -----------------------------
risk_score = (immune * 0.4 + stress * 0.3 + (1 - treatment) * 0.3) * 100

# -----------------------------
# PDF FUNCTION (UPDATED 🔥)
# -----------------------------
def create_pdf():
    fig.savefig("graph.png")  # save graph

    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("<b>Vitiligo Simulation Report</b>", styles["Title"]))
    content.append(Spacer(1, 15))

    name = f"{fname} {lname}".strip()
    content.append(Paragraph(f"<b>Name:</b> {name if name else 'Not provided'}", styles["Normal"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"<b>Predicted Level:</b> {pred:.2f}", styles["Normal"]))
    content.append(Paragraph(f"<b>Risk Score:</b> {risk_score:.2f}", styles["Normal"]))
    content.append(Paragraph(f"<b>Cluster:</b> {cluster_text}", styles["Normal"]))
    content.append(Spacer(1, 15))

    content.append(Paragraph("<b>Disease Progression Graph</b>", styles["Heading2"]))
    content.append(Spacer(1, 10))

    img = Image("graph.png", width=5*inch, height=3*inch)
    content.append(img)

    content.append(Spacer(1, 15))

    if pred > 70:
        text = "Condition is mild. Treatment is effective."
    elif pred > 40:
        text = "Moderate progression observed."
    else:
        text = "Severe depigmentation detected."

    content.append(Paragraph("<b>Interpretation:</b>", styles["Heading2"]))
    content.append(Paragraph(text, styles["Normal"]))

    doc.build(content)

# -----------------------------
# DOWNLOAD BUTTON
# -----------------------------
create_pdf()

with open("report.pdf", "rb") as f:
    st.download_button("📄 Download Full Report", f, file_name="vitiligo_report.pdf")

# -----------------------------
# DISCLAIMER
# -----------------------------
st.caption("This tool is for educational purposes only and not for medical diagnosis.")
