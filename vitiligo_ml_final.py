import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

# -----------------------------
# ??? SIDEBAR (like sir's style)
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
# ?? HEADER
# -----------------------------
st.header(f"Welcome, {uname if uname else 'User'}")
st.write("Vitiligo progression simulation using ML + stochastic modeling")

if notify:
    st.success("Notifications enabled")

# -----------------------------
# ?? SIMULATION FUNCTION
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
# ?? USER PROFILE SECTION
# -----------------------------
with st.container():
    st.subheader('Patient Profile')

    col1, col2 = st.columns(2)

    with col1:
        fname = st.text_input('First name')

    with col2:
        lname = st.text_input('Last name')


# ?? SIMULATION RESULTS
# -----------------------------
with st.container():
    st.subheader("Simulation Results")

    data = simulate(100, immune, stress, treatment, start)

    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_title("Disease Progression")

    st.pyplot(fig)

# -----------------------------
# ? COMPARISON
# -----------------------------
with st.container():
    st.subheader("Treatment Comparison")

    early = simulate(100, immune, stress, treatment, 10)
    late = simulate(100, immune, stress, treatment, 60)

    fig2, ax2 = plt.subplots()
    ax2.plot(early, label="Early Treatment")
    ax2.plot(late, label="Late Treatment")
    ax2.legend()

    st.pyplot(fig2)

# -----------------------------
# ?? ML MODEL
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

# -----------------------------
# ?? CLUSTERING
# -----------------------------
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

cluster = kmeans.predict([[immune, stress, treatment, start]])[0]

st.subheader("Patient Category")
st.write(f"Cluster Group: {cluster}")

# -----------------------------
# ?? INTERPRETATION
# -----------------------------
with st.container():
    st.subheader("Analysis")

    if pred > 70:
        st.success("Mild condition")
    elif pred > 40:
        st.warning("Moderate progression")
    else:
        st.error("Severe condition")


import pandas as pd

features = ['Immune', 'Stress', 'Treatment', 'Start']
importance = model.feature_importances_

df = pd.DataFrame({'Feature':features, 'Importance':importance})

st.bar_chart(df.set_index('Feature'))

if pred < 40:
    risk = "High Risk"
elif pred < 70:
    risk = "Moderate Risk"
else:
    risk = "Low Risk"

if pred > 70:
    insight = "Disease progression is controlled. Treatment is effective."
elif pred > 40:
    insight = "Moderate progression observed. Early intervention could improve outcome."
else:
    insight = "Rapid depigmentation detected. Stronger or earlier treatment required."

st.success(f"""
Summary:

- Predicted Melanocyte Level: {pred:.2f}
- Risk Level: {risk}

Interpretation:
{insight}
""")


