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
Vitiligo is a skin condition characterized by loss of melanocytes, causing depigmented patches.
This tool simulates vitiligo progression using stochastic modeling and machine learning.
""")

# -----------------------------
# TAB 2: Objective
# -----------------------------
with tab2:
    st.header("Objective")
    st.write("""
- Predict melanocyte loss over time
- Evaluate treatment effects
- Categorize patients by progression
- Provide educational insights and risk scores
- Generate PDF reports
""")

# -----------------------------
# TAB 3: Working (All simulation + ML + Risk + PDF)
# -----------------------------
with tab3:
    st.header("Working / Simulation")

    # Patient Profile
    col1, col2 = st.columns(2)
    with col1:
        fname = st.text_input('First name')
    with col2:
        lname = st.text_input('Last name')

    # Simulation function
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

    # Simulation plots
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

    # ML model
    X, y = [], []
    for _ in range(150):
        i = np.random.uniform(0,1)
        s = np.random.uniform(0,1)
        tr = np.random.uniform(0,1)
        stt = np.random.randint(0,100)
        sim_vals = simulate(100, i, s, tr, stt)
        X.append([i, s, tr, stt])
        y.append(sim_vals[-1])
    model = RandomForestRegressor()
    model.fit(X, y)
    pred = model.predict([[immune, stress, treatment, start]])[0]

    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(X)
    cluster = kmeans.predict([[immune, stress, treatment, start]])[0]
    cluster_map = {0: "Slow progression group", 1: "Moderate progression group", 2: "Aggressive progression group"}

    # Display ML outputs
    st.subheader("Predicted Outcome")
    st.write(f"Final Melanocyte Level: {pred:.2f}")
    st.caption("Higher value = healthier pigmentation")

    st.subheader("Patient Category")
    st.write(cluster_map.get(cluster, "Unknown"))

    st.subheader("Feature Importance")
    features = ['Immune', 'Stress', 'Treatment', 'Start']
    importance = model.feature_importances_
    df = pd.DataFrame({'Feature':features, 'Importance':importance})
    st.bar_chart(df.set_index('Feature'))

    # Risk Score
    st.subheader("Risk Score")
    risk_score = (immune * 0.4 + stress * 0.3 + (1 - treatment) * 0.3) * 100
    st.write(f"{risk_score:.2f} / 100")

    # PDF download
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
Risk Score: {risk_score:.2f}
Cluster: {cluster_map.get(cluster)}
"""
    create_pdf(report_text)
    with open("report.pdf", "rb") as f:
        st.download_button("📄 Download Report", f, file_name="vitiligo_report.pdf")

# -----------------------------
# TAB 4: Applications
# -----------------------------
with tab4:
    st.header("Applications of the Tool")
    st.write("""
- Educational: Understand vitiligo progression
- Research: Test treatment scenarios
- Clinical insights: Compare early vs late treatment effects
- Training: For medical students / dermatology trainees
""")

# -----------------------------
# TAB 5: Team
# -----------------------------
with tab5:
    st.header("Team Members")
    st.write("""DISHA THORAT

""")
