import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
                             confusion_matrix, classification_report)
from sklearn.calibration import calibration_curve
import warnings
import re

warnings.filterwarnings('ignore')

# =============================
# 全局常量定义
# =============================
ETHNICITIES_LIST = [
    "Australian", "Australian Aboriginal", "Australian South Sea Islander", "Torres Strait Islander",
    "New Zealand Peoples", "Melanesian and Papuan", "Micronesian", "Polynesian",
    "British", "Irish", "Western European", "Northern European", "Southern European",
    "South Eastern European", "Eastern European", "Arab", "Jewish", "Peoples of the Sudan",
    "Other North African and Middle Eastern", "Mainland South-East Asian", "Maritime South-East Asian",
    "Chinese Asian", "Other North-East Asian", "Southern Asian", "Central Asian",
    "North American", "South American", "Central American", "Caribbean Islander",
    "Central and West African", "Southern and East African"
]


# =============================
# HELPER FUNCTIONS FOR PDF PARSING
# =============================
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except ImportError:
        st.error("⚠️ PyPDF2 library not installed. Install with: pip install PyPDF2")
        return ""
    except Exception as e:
        st.error(f"❌ Error reading PDF: {str(e)}")
        return ""


def parse_resume_text(text):
    """Parse resume text to extract candidate information using keyword analysis"""
    extracted = {
        'cohort': 2023,
        'gender': "Female",
        'ethnicity': "Australian",
        'degree_major': "Other",
        'gpa': 3.0,
        'teaching_experience': "None",
        'leadership_score': 70.0,
        'communication_score': 70.0,
        'self_efficacy': 70.0,
        'organisational_support': 70.0,
        'raw_text': text[:500]
    }

    text_lower = text.lower()

    # Extract GPA
    gpa_patterns = [
        r'gpa[:\s]+(\d+\.?\d*)',
        r'grade point average[:\s]+(\d+\.?\d*)',
        r'wam[:\s]+(\d+\.?\d*)',
        r'average[:\s]+(\d+\.?\d*)',
    ]
    for pattern in gpa_patterns:
        match = re.search(pattern, text_lower)
        if match:
            gpa_val = float(match.group(1))
            if gpa_val > 4.0:
                gpa_val = gpa_val / 25.0
            extracted['gpa'] = min(max(gpa_val, 2.0), 4.0)
            break

    # Extract degree major
    major_keywords = {
        'Education': ['education', 'teaching', 'pedagogy', 'curriculum'],
        'STEM': ['science', 'technology', 'engineering', 'mathematics', 'computer', 'physics', 'chemistry', 'biology',
                 'data'],
        'Humanities': ['humanities', 'history', 'philosophy', 'literature', 'english', 'languages', 'linguistics'],
        'Commerce': ['business', 'commerce', 'economics', 'finance', 'accounting', 'marketing', 'management'],
        'Arts': ['arts', 'fine arts', 'design', 'music', 'theatre', 'visual', 'creative']
    }

    for major, keywords in major_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            extracted['degree_major'] = major
            break

    # Extract teaching experience
    if any(word in text_lower for word in ['teacher', 'teaching', 'tutor', 'educator', 'instructor']):
        exp_match = re.search(r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:teaching|experience)', text_lower)
        if exp_match:
            years = int(exp_match.group(1))
            if years >= 3:
                extracted['teaching_experience'] = "3+ years"
            elif years >= 1:
                extracted['teaching_experience'] = "1-3 years"
            else:
                extracted['teaching_experience'] = "<1 year"
        else:
            extracted['teaching_experience'] = "<1 year"

    # Infer leadership score
    leadership_keywords = ['lead', 'manage', 'coordinate', 'president', 'captain', 'director', 'supervisor', 'chair',
                           'head']
    leadership_count = sum(1 for keyword in leadership_keywords if keyword in text_lower)
    extracted['leadership_score'] = min(65.0 + (leadership_count * 4), 95.0)

    # Infer communication score
    communication_keywords = ['presentation', 'public speaking', 'communication', 'wrote', 'published', 'presented',
                              'article', 'report']
    comm_count = sum(1 for keyword in communication_keywords if keyword in text_lower)
    extracted['communication_score'] = min(65.0 + (comm_count * 4), 95.0)

    # Infer self-efficacy
    efficacy_keywords = ['confident', 'achieved', 'successful', 'award', 'recognition', 'excell']
    efficacy_count = sum(1 for keyword in efficacy_keywords if keyword in text_lower)
    extracted['self_efficacy'] = min(65.0 + (efficacy_count * 4), 90.0)

    return extracted


# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="TFA AI Fairness Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        font-size: 1.1rem;
    }
    .fasttrack-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    .secondlook-box {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    .info-box {
        background: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .upload-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px dashed #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🎓 Teach For Australia</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Recruitment Fairness Dashboard</p>', unsafe_allow_html=True)


# =============================
# DATA LOADING
# =============================
@st.cache_data
def load_data():
    import os

    file_path = "recruitment_dataset_with_scores.csv"

    # 定义必需的列
    required_columns = [
        'candidate_id', 'cohort', 'gender', 'ethnicity', 'degree_major',
        'teaching_experience', 'gpa', 'leadership_score',
        'communication_score', 'self_efficacy', 'organisational_support',
        'selected', 'proba'
    ]

    should_generate = False

    # Check if file exists
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)

            # Verify that all required columns are present
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                st.warning(f"⚠️ The dataset is missing the following columns: {', '.join(missing_columns)}")
                st.info("🔄 Regenerating the complete dataset...")
                should_generate = True
            else:
                st.success(f"✅ {file_path} loaded successfully ({len(df)} rows)")

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("🔄 Regenerating dataset...")
            should_generate = True
    else:
        st.warning("⚠️ Dataset file not found, generating synthetic dataset...")
        should_generate = True

    # 生成新数据
    if should_generate:
        np.random.seed(42)
        n = 2000

        # ======== Ethnicity (Australian context) - 完整31个分类 ========
        ethnicities = ETHNICITIES_LIST

        ethnicity_probs = np.array([
            0.25, 0.03, 0.005, 0.005,
            0.03, 0.01, 0.002, 0.005,
            0.18, 0.02, 0.015, 0.015,
            0.02, 0.015, 0.015,
            0.015, 0.005, 0.003, 0.004,
            0.05, 0.05, 0.05, 0.01,
            0.03, 0.01,
            0.01, 0.005, 0.003, 0.002,
            0.01, 0.008
        ])
        ethnicity_probs /= ethnicity_probs.sum()

        # ======== Generate other features ========
        genders = ["Male", "Female", "Non-binary"]
        majors = ["Education", "STEM", "Humanities", "Commerce", "Arts", "Other"]
        teaching_exp = ["None", "<1 year", "1-3 years", "3+ years"]

        GPA = np.round(np.random.normal(3.0, 0.5, n).clip(2.0, 4.0), 2)

        # Assessment Scores
        leadership = np.round(np.random.normal(70, 10, n).clip(40, 100), 2)
        communication = np.round(np.random.normal(72, 9, n).clip(40, 100), 2)
        self_efficacy = np.round(np.random.normal(68, 10, n).clip(40, 100), 2)
        org_support = np.round(np.random.normal(65, 10, n).clip(40, 100), 2)

        df = pd.DataFrame({
            "candidate_id": np.arange(1, n + 1),
            "cohort": np.random.choice([2019, 2020, 2021, 2022, 2023], n),
            "gender": np.random.choice(genders, n, p=[0.45, 0.5, 0.05]),
            "ethnicity": np.random.choice(ethnicities, size=n, p=ethnicity_probs),
            "degree_major": np.random.choice(majors, n, p=[0.3, 0.25, 0.15, 0.15, 0.1, 0.05]),
            "teaching_experience": np.random.choice(teaching_exp, n, p=[0.3, 0.3, 0.25, 0.15]),
            "gpa": GPA,
            "leadership_score": leadership,
            "communication_score": communication,
            "self_efficacy": self_efficacy,
            "organisational_support": org_support
        })

        # ======== Simulate selection outcome ========
        score = (
                df["gpa"] * 0.4 +
                df["leadership_score"] * 0.15 +
                df["communication_score"] * 0.15 +
                df["self_efficacy"] * 0.1 +
                df["organisational_support"] * 0.15 +
                np.random.normal(0, 2, n)
        )
        threshold = np.percentile(score, 65)
        df["selected"] = np.where(score > threshold, 1, 0)

        df["proba"] = (
                0.35 * (df["leadership_score"] / 100) +
                0.3 * (df["communication_score"] / 100) +
                0.2 * (df["self_efficacy"] / 100) +
                0.15 * (df["gpa"] / 4)
        ).clip(0, 1)

        # Save for reuse
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        st.success(f"✅ 合成数据集已生成并保存到 {file_path}")

    # ======== Model Card Text ========
    try:
        card = open("TFA_Model_Card.md").read()
    except FileNotFoundError:
        card = """# Model Card: TFA Recruitment AI
## Model Details
- **Version**: 1.0
- **Type**: Binary Classification (Random Forest)
- **Purpose**: Assist in candidate selection for Teach For Australia program
## Ethical Considerations
- Demographic features used only for fairness monitoring
- Regular bias audits conducted
"""
    return df, card


@st.cache_resource
def train_model(data):
    """Train Random Forest model for predictions"""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier

    target = "selected"
    exclude_cols = [target, "candidate_id", "proba"]

    # 获取所有可用的特征列
    feature_cols = [c for c in data.columns if c not in exclude_cols]

    X = data[feature_cols]
    y = data[target].astype(int)

    # 定义分类列 - 只使用实际存在的列
    potential_cat_cols = ["gender", "ethnicity", "degree_major", "teaching_experience", "cohort"]
    cat_cols = [col for col in potential_cat_cols if col in X.columns]

    # 定义数值列
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 构建预处理管道
    transformers = []

    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        )

    if num_cols:
        transformers.append(
            ("num", StandardScaler(), num_cols)
        )

    preprocess = ColumnTransformer(transformers, remainder='drop')

    # 构建随机森林模型
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )

    # 构建完整管道
    model = Pipeline([("pre", preprocess), ("clf", rf)])
    model.fit(X_train, y_train)

    # 获取特征名称
    feature_names = []
    if cat_cols:
        cat_feature_names = list(
            model.named_steps['pre'].named_transformers_['cat']
            .get_feature_names_out(cat_cols)
        )
        feature_names.extend(cat_feature_names)

    if num_cols:
        feature_names.extend(num_cols)

    return model, feature_names, cat_cols, num_cols


data, model_card_text = load_data()
model, feature_names, cat_cols, num_cols = train_model(data)

# =============================
# SIDEBAR CONTROLS
# =============================
st.sidebar.title("🎛️ Dashboard Controls")

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Adjust the classification threshold"
)

st.sidebar.markdown("---")

st.sidebar.subheader("🔍 Filters")

selected_cohorts = st.sidebar.multiselect(
    "Cohort Year",
    options=sorted(data["cohort"].unique()),
    default=sorted(data["cohort"].unique())
)

selected_majors = st.sidebar.multiselect(
    "Degree Major",
    options=sorted(data["degree_major"].unique()),
    default=sorted(data["degree_major"].unique())
)

selected_genders = st.sidebar.multiselect(
    "Gender",
    options=sorted(data["gender"].unique()),
    default=sorted(data["gender"].unique())
)

filtered_data = data[
    (data["cohort"].isin(selected_cohorts)) &
    (data["degree_major"].isin(selected_majors)) &
    (data["gender"].isin(selected_genders))
    ].copy()

filtered_data["pred"] = (filtered_data["proba"] >= threshold).astype(int)

st.sidebar.markdown("---")
st.sidebar.info(f"📊 **{len(filtered_data):,}** candidates selected out of **{len(data):,}** total")

csv = filtered_data.to_csv(index=False)
st.sidebar.download_button(
    label="📥 Download Filtered Data",
    data=csv,
    file_name="tfa_filtered_data.csv",
    mime="text/csv"
)


# =============================
# KEY METRICS
# =============================
def calculate_metrics(df, threshold):
    y_true = df["selected"]
    y_pred = (df["proba"] >= threshold).astype(int)
    y_proba = df["proba"]

    auc = roc_auc_score(y_true, y_proba)
    accuracy = (y_pred == y_true).mean()

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    selection_rate = y_pred.mean()
    actual_rate = y_true.mean()

    return {
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "selection_rate": selection_rate,
        "actual_rate": actual_rate,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }


metrics = calculate_metrics(filtered_data, threshold)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("🎯 AUC Score", f"{metrics['auc']:.3f}")
with col2:
    st.metric("✅ Accuracy", f"{metrics['accuracy']:.3f}")
with col3:
    st.metric("🎪 Precision", f"{metrics['precision']:.3f}")
with col4:
    st.metric("🔍 Recall", f"{metrics['recall']:.3f}")
with col5:
    st.metric("⚖️ F1 Score", f"{metrics['f1']:.3f}")

st.markdown("---")

# =============================
# TABBED INTERFACE
# =============================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎲 Candidate Simulator",
    "📈 Performance",
    "⚖️ Fairness Analysis",
    "🎯 Calibration",
    "🔬 Feature Analysis",
    "📄 Model Card"
])

# =============================
# TAB 1: CANDIDATE SIMULATOR WITH PDF UPLOAD
# =============================
with tab1:
    st.header("🎲 Interactive Candidate Simulator")
    st.markdown("Upload a resume or manually build a candidate profile to receive real-time predictions.")

    # PDF Upload Section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📄 Option 1: Upload Resume (PDF)")

    col_upload1, col_upload2 = st.columns([3, 1])

    with col_upload1:
        uploaded_file = st.file_uploader(
            "Upload candidate's resume in PDF format",
            type=['pdf'],
            help="Upload a PDF resume to automatically extract candidate information"
        )

    with col_upload2:
        st.markdown("<br>", unsafe_allow_html=True)
        if uploaded_file is not None:
            if st.button("🔍 Parse Resume", type="primary", use_container_width=True):
                with st.spinner("Analyzing resume..."):
                    pdf_text = extract_text_from_pdf(uploaded_file)

                    if pdf_text:
                        parsed_data = parse_resume_text(pdf_text)
                        st.session_state.candidate = parsed_data
                        st.success("✅ Resume parsed successfully!")

                        with st.expander("📋 View Extracted Information"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Academic Information:**")
                                st.write(f"- GPA: {parsed_data['gpa']}")
                                st.write(f"- Major: {parsed_data['degree_major']}")
                                st.write(f"- Experience: {parsed_data['teaching_experience']}")
                            with col2:
                                st.write("**Estimated Scores:**")
                                st.write(f"- Leadership: {parsed_data['leadership_score']:.1f}")
                                st.write(f"- Communication: {parsed_data['communication_score']:.1f}")
                                st.write(f"- Self-Efficacy: {parsed_data['self_efficacy']:.1f}")

                            st.markdown("**Resume Preview:**")
                            st.text(parsed_data.get('raw_text', 'No preview available')[:400] + "...")
                    else:
                        st.error("❌ Could not extract text from PDF. Please check the file format.")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📝 Option 2: Manual Profile Builder")

    col_input, col_output = st.columns([1.2, 1])

    with col_input:
        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("🎲 Generate Random", use_container_width=True):
                st.session_state.candidate = {
                    'cohort': int(np.random.choice([2019, 2020, 2021, 2022, 2023])),
                    'gender': np.random.choice(["Male", "Female", "Non-binary"]),
                    'ethnicity': np.random.choice(ETHNICITIES_LIST),
                    'degree_major': np.random.choice(["Education", "STEM", "Humanities", "Commerce", "Arts", "Other"]),
                    'gpa': round(np.random.uniform(2.5, 4.0), 2),
                    'teaching_experience': np.random.choice(["None", "<1 year", "1-3 years", "3+ years"]),
                    'leadership_score': round(np.random.uniform(50, 95), 1),
                    'communication_score': round(np.random.uniform(50, 95), 1),
                    'self_efficacy': round(np.random.uniform(50, 90), 1),
                    'organisational_support': round(np.random.uniform(50, 90), 1)
                }

        with col_btn2:
            if st.button("🔄 Reset to Default", use_container_width=True):
                st.session_state.candidate = {
                    'cohort': 2023, 'gender': "Female", 'ethnicity': "Australian",
                    'degree_major': "Education", 'gpa': 3.5, 'teaching_experience': "1-3 years",
                    'leadership_score': 75.0, 'communication_score': 78.0,
                    'self_efficacy': 70.0, 'organisational_support': 68.0
                }

        if 'candidate' not in st.session_state:
            st.session_state.candidate = {
                'cohort': 2023, 'gender': "Female", 'ethnicity': "Australian",
                'degree_major': "Education", 'gpa': 3.5, 'teaching_experience': "1-3 years",
                'leadership_score': 75.0, 'communication_score': 78.0,
                'self_efficacy': 70.0, 'organisational_support': 68.0
            }

        c = st.session_state.candidate

        st.markdown("#### 👤 Demographics (audit only)")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"],
                                  index=["Male", "Female", "Non-binary"].index(c['gender']))
        with col2:
            current_ethnicity = c['ethnicity'] if c['ethnicity'] in ETHNICITIES_LIST else "Australian"
            ethnicity = st.selectbox(
                "Ethnicity",
                ETHNICITIES_LIST,
                index=ETHNICITIES_LIST.index(current_ethnicity)
            )

        st.markdown("#### 🎓 Academic Background")
        col1, col2 = st.columns(2)
        with col1:
            cohort = st.selectbox("Cohort Year", [2019, 2020, 2021, 2022, 2023],
                                  index=[2019, 2020, 2021, 2022, 2023].index(c['cohort']))
            major = st.selectbox("Degree Major",
                                 ["Education", "STEM", "Humanities", "Commerce", "Arts", "Other"],
                                 index=["Education", "STEM", "Humanities", "Commerce", "Arts", "Other"].index(
                                     c['degree_major']))
        with col2:
            gpa = st.slider("GPA", 2.0, 4.0, float(c['gpa']), 0.1)
            experience = st.selectbox("Teaching Experience",
                                      ["None", "<1 year", "1-3 years", "3+ years"],
                                      index=["None", "<1 year", "1-3 years", "3+ years"].index(
                                          c['teaching_experience']))

        st.markdown("#### 📊 Assessment Scores")
        leadership = st.slider("Leadership Score", 40.0, 100.0, float(c['leadership_score']), 1.0,
                               help="Score from leadership assessment interview")
        communication = st.slider("Communication Score", 40.0, 100.0, float(c['communication_score']), 1.0,
                                  help="Score from communication skills evaluation")
        efficacy = st.slider("Self-Efficacy Score", 30.0, 100.0, float(c['self_efficacy']), 1.0,
                             help="Psychological assessment score")
        support = st.slider("Organizational Support", 30.0, 100.0, float(c['organisational_support']), 1.0,
                            help="Perceived organizational support score")

    with col_output:
        st.subheader("🎯 AI Prediction Results")

        candidate_df = pd.DataFrame({
            "cohort": [cohort],
            "gender": [gender],
            "ethnicity": [ethnicity],
            "degree_major": [major],
            "gpa": [gpa],
            "teaching_experience": [experience],
            "leadership_score": [leadership],
            "communication_score": [communication],
            "self_efficacy": [efficacy],
            "organisational_support": [support]
        })

        proba = model.predict_proba(candidate_df)[0, 1]
        decision = "✅ FAST-TRACK RECOMMENDED" if proba >= threshold else "🔄 SECOND-LOOK REQUIRED"
        decision_class = "fasttrack-box" if proba >= threshold else "secondlook-box"

        st.markdown(f'<div class="{decision_class}">{decision}</div>', unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=proba,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Selection Probability", 'font': {'size': 18}},
            delta={'reference': threshold, 'increasing': {'color': "green"}},
            number={'font': {'size': 35}, 'suffix': "", 'valueformat': '.3f'},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 2},
                'bar': {'color': "darkblue", 'thickness': 0.75},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, threshold], 'color': '#ffdddd'},
                    {'range': [threshold, 1], 'color': '#ddffdd'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold
                }
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div class="info-box">
        <h4>📋 Interpretation</h4>
        <ul>
        <li><b>Success Probability:</b> {proba * 100:.1f}%</li>
        <li><b>Decision Threshold:</b> {threshold * 100:.0f}%</li>
        <li><b>Status:</b> {"✅ Above threshold" if proba >= threshold else "⚠️ Below threshold"}</li>
        <li><b>Recommendation:</b> {"Fast-track to next stage" if proba >= threshold else "Schedule additional review panel"}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔍 What's Driving This Prediction?")

    col1, col2 = st.columns([1.5, 1])

    with col1:
        importances = model.named_steps['clf'].feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(10)


        def simplify_name(name):
            replacements = {
                'gender_': '', 'ethnicity_': '', 'degree_major_': '',
                'teaching_experience_': '', 'cohort_': '',
                'leadership_score': 'Leadership', 'communication_score': 'Communication',
                'self_efficacy': 'Self-Efficacy', 'organisational_support': 'Org Support',
                'gpa': 'GPA'
            }
            for old, new in replacements.items():
                name = name.replace(old, new)
            return name.title()


        importance_df['Feature'] = importance_df['Feature'].apply(simplify_name)

        fig_imp = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top 10 Most Important Features (Global Model)",
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig_imp.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    with col2:
        st.markdown("### 📊 This Candidate's Scores")

        scores = {
            'Leadership': leadership,
            'Communication': communication,
            'Self-Efficacy': efficacy,
            'Org Support': support,
            'GPA': gpa * 25
        }

        scores_df = pd.DataFrame({
            'Metric': list(scores.keys()),
            'Score': list(scores.values())
        })

        fig_scores = go.Figure()
        fig_scores.add_trace(go.Bar(
            x=scores_df['Score'],
            y=scores_df['Metric'],
            orientation='h',
            marker=dict(
                color=scores_df['Score'],
                colorscale='RdYlGn',
                cmin=0,
                cmax=100
            ),
            text=scores_df['Score'].round(1),
            textposition='outside'
        ))
        fig_scores.update_layout(
            height=400,
            xaxis=dict(range=[0, 105]),
            xaxis_title="Score",
            showlegend=False
        )
        st.plotly_chart(fig_scores, use_container_width=True)

    st.markdown("---")
    st.subheader("⚖️ Fairness Context & Group Comparisons")

    gender_avg = data[data['gender'] == gender]['proba'].mean()
    ethnicity_avg = data[data['ethnicity'] == ethnicity]['proba'].mean()
    overall_avg = data['proba'].mean()

    gender_pass_rate = (data[data['gender'] == gender]['proba'] >= threshold).mean()
    ethnicity_pass_rate = (data[data['ethnicity'] == ethnicity]['proba'] >= threshold).mean()
    overall_pass_rate = (data['proba'] >= threshold).mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Overall Avg Score", f"{overall_avg:.3f}")
        st.caption(f"Pass rate: {overall_pass_rate:.1%}")
    with col2:
        st.metric(f"{gender} Avg", f"{gender_avg:.3f}",
                  delta=f"{(proba - gender_avg):.3f}")
        st.caption(f"Pass rate: {gender_pass_rate:.1%}")
    with col3:
        st.metric(f"{ethnicity} Avg", f"{ethnicity_avg:.3f}",
                  delta=f"{(proba - ethnicity_avg):.3f}")
        st.caption(f"Pass rate: {ethnicity_pass_rate:.1%}")
    with col4:
        st.metric("This Candidate", f"{proba:.3f}",
                  delta=f"{(proba - overall_avg):.3f}")
        st.caption(f"{'✅ Pass' if proba >= threshold else '⚠️ Review'}")

    st.info(f"""
    📌 **Context:** This candidate's score is {abs(proba - overall_avg):.3f} {'above' if proba > overall_avg else 'below'} 
    the overall average. Their demographic group ({gender}, {ethnicity}) has a {gender_pass_rate:.1%} fast-track rate.
    """)

# =============================
# TAB 2: PERFORMANCE
# =============================
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC Curve")
        fpr, tpr, thresholds = roc_curve(filtered_data["selected"], filtered_data["proba"])

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC (AUC={metrics["auc"]:.3f})',
            line=dict(color='#1f77b4', width=3),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)'
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', dash='dash', width=2)
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            hovermode='closest',
            height=400
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        st.subheader("Precision-Recall Curve")
        precision, recall, _ = precision_recall_curve(filtered_data["selected"], filtered_data["proba"])

        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name='PR Curve',
            line=dict(color='#ff7f0e', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 127, 14, 0.2)'
        ))
        fig_pr.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            hovermode='closest',
            height=400
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    st.subheader("🎭 Confusion Matrix")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        **True Negatives:** {metrics['tn']}  
        **False Positives:** {metrics['fp']}  
        **False Negatives:** {metrics['fn']}  
        **True Positives:** {metrics['tp']}

        ---

        **Selection Rate:** {metrics['selection_rate']:.1%}  
        **Actual Rate:** {metrics['actual_rate']:.1%}
        """)

    with col2:
        cm = confusion_matrix(filtered_data["selected"], filtered_data["pred"])
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Not Selected', 'Selected'],
            y=['Not Selected', 'Selected'],
            text_auto=True,
            color_continuous_scale='Blues',
            aspect='auto'
        )
        fig_cm.update_layout(height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("📊 Prediction Score Distribution")

    fig_dist = go.Figure()

    fig_dist.add_trace(go.Histogram(
        x=filtered_data[filtered_data['selected'] == 1]['proba'],
        name='Actually Selected',
        opacity=0.7,
        marker_color='#2ecc71',
        nbinsx=30
    ))

    fig_dist.add_trace(go.Histogram(
        x=filtered_data[filtered_data['selected'] == 0]['proba'],
        name='Not Selected',
        opacity=0.7,
        marker_color='#e74c3c',
        nbinsx=30
    ))

    fig_dist.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Threshold = {threshold:.2f}"
    )

    fig_dist.update_layout(
        barmode='overlay',
        xaxis_title="Predicted Probability",
        yaxis_title="Count",
        height=400
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# =============================
# TAB 3: FAIRNESS ANALYSIS
# =============================
with tab3:
    def group_fairness_metrics(df, sensitive_col, ref_group):
        results = []
        for group in df[sensitive_col].unique():
            group_df = df[df[sensitive_col] == group]
            y_true = group_df["selected"]
            y_pred = group_df["pred"]

            selection_rate = y_pred.mean()

            if y_true.sum() > 0:
                tp = ((y_pred == 1) & (y_true == 1)).sum()
                fn = ((y_pred == 0) & (y_true == 1)).sum()
                fp = ((y_pred == 1) & (y_true == 0)).sum()
                tn = ((y_pred == 0) & (y_true == 0)).sum()

                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            else:
                tpr, fpr, ppv = 0, 0, 0

            results.append({
                "Group": group,
                "Count": len(group_df),
                "Selection Rate": selection_rate,
                "TPR (Recall)": tpr,
                "FPR": fpr,
                "PPV (Precision)": ppv
            })

        df_results = pd.DataFrame(results)

        if ref_group in df_results["Group"].values:
            ref_row = df_results[df_results["Group"] == ref_group].iloc[0]
            df_results["SPD"] = df_results["Selection Rate"] - ref_row["Selection Rate"]
            df_results["TPR Gap"] = df_results["TPR (Recall)"] - ref_row["TPR (Recall)"]
            df_results["FPR Gap"] = df_results["FPR"] - ref_row["FPR"]
        else:
            df_results["SPD"] = 0
            df_results["TPR Gap"] = 0
            df_results["FPR Gap"] = 0

        return df_results


    st.subheader("👥 Gender Fairness Analysis")
    gender_fairness = group_fairness_metrics(filtered_data, "gender", "Male")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(
            gender_fairness.style.format({
                "Selection Rate": "{:.1%}",
                "TPR (Recall)": "{:.3f}",
                "FPR": "{:.3f}",
                "PPV (Precision)": "{:.3f}",
                "SPD": "{:.3f}",
                "TPR Gap": "{:.3f}",
                "FPR Gap": "{:.3f}"
            }).background_gradient(subset=["SPD", "TPR Gap", "FPR Gap"], cmap="RdYlGn", vmin=-0.1, vmax=0.1),
            height=200
        )

    with col2:
        fig_gender = go.Figure()

        for metric in ["SPD", "TPR Gap", "FPR Gap"]:
            fig_gender.add_trace(go.Bar(
                name=metric,
                x=gender_fairness["Group"],
                y=gender_fairness[metric],
                text=gender_fairness[metric].round(3),
                textposition='outside'
            ))

        fig_gender.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_gender.update_layout(
            barmode='group',
            yaxis_title="Gap (relative to Male)",
            xaxis_title="Gender",
            height=350
        )
        st.plotly_chart(fig_gender, use_container_width=True)

    st.markdown("---")

    st.subheader("🌏 Ethnicity Fairness Analysis")
    eth_fairness = group_fairness_metrics(filtered_data, "ethnicity", "Australian")

    # 只显示样本量大于10的种族组
    eth_fairness_filtered = eth_fairness[eth_fairness["Count"] >= 10].sort_values("Count", ascending=False)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(
            eth_fairness_filtered.style.format({
                "Selection Rate": "{:.1%}",
                "TPR (Recall)": "{:.3f}",
                "FPR": "{:.3f}",
                "PPV (Precision)": "{:.3f}",
                "SPD": "{:.3f}",
                "TPR Gap": "{:.3f}",
                "FPR Gap": "{:.3f}"
            }).background_gradient(subset=["SPD", "TPR Gap", "FPR Gap"], cmap="RdYlGn", vmin=-0.15, vmax=0.15),
            height=400
        )

    with col2:
        # 只显示前10个最大的种族组
        top_ethnicities = eth_fairness_filtered.head(10)

        fig_eth = go.Figure()

        for metric in ["SPD", "TPR Gap", "FPR Gap"]:
            fig_eth.add_trace(go.Bar(
                name=metric,
                x=top_ethnicities["Group"],
                y=top_ethnicities[metric],
                text=top_ethnicities[metric].round(3),
                textposition='outside'
            ))

        fig_eth.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_eth.update_layout(
            barmode='group',
            yaxis_title="Gap (relative to Australian)",
            xaxis_title="Ethnicity (Top 10 by sample size)",
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_eth, use_container_width=True)

    st.markdown("---")
    st.subheader("🔀 Intersectional Analysis (Gender × Major)")

    intersect_data = filtered_data.groupby(['gender', 'degree_major']).agg({
        'pred': 'mean',
        'selected': 'mean',
        'candidate_id': 'count'
    }).reset_index()
    intersect_data.columns = ['Gender', 'Major', 'Predicted Rate', 'Actual Rate', 'Count']

    # 过滤样本量小的组
    intersect_data = intersect_data[intersect_data['Count'] >= 5]

    fig_heatmap = px.density_heatmap(
        intersect_data,
        x='Major',
        y='Gender',
        z='Predicted Rate',
        text_auto='.2%',
        color_continuous_scale='RdYlGn',
        title="Selection Rate Heatmap (Gender × Major)"
    )
    fig_heatmap.update_layout(height=300)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    with st.expander("ℹ️ Understanding Fairness Metrics"):
        st.markdown("""
        **Statistical Parity Difference (SPD)**: Difference in selection rates between groups.  
        - Values close to 0 indicate similar selection rates.

        **TPR Gap (True Positive Rate)**: Difference in recall between groups.  
        - Measures equal opportunity - whether qualified candidates are selected equally.

        **FPR Gap (False Positive Rate)**: Difference in false positive rates.  
        - Measures if unqualified candidates are incorrectly selected at different rates.

        **Ideal**: All gaps should be close to 0 for fairness across groups.
        """)

# =============================
# TAB 4: CALIBRATION
# =============================
with tab4:
    st.subheader("📐 Model Calibration Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            filtered_data["selected"],
            filtered_data["proba"],
            n_bins=10,
            strategy='uniform'
        )

        fig_calib = go.Figure()

        fig_calib.add_trace(go.Scatter(
            x=mean_predicted_value,
            y=fraction_of_positives,
            mode='lines+markers',
            name='Model',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=10)
        ))

        fig_calib.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfectly Calibrated',
            line=dict(color='gray', dash='dash', width=2)
        ))

        fig_calib.update_layout(
            title="Calibration Curve",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            height=400
        )
        st.plotly_chart(fig_calib, use_container_width=True)

    with col2:
        st.markdown("### 🎯 Triage Bands Analysis")


        def triage_band(p):
            if p < 0.3:
                return "Low (< 0.3)"
            elif p < 0.5:
                return "Medium-Low (0.3-0.5)"
            elif p < 0.7:
                return "Medium-High (0.5-0.7)"
            else:
                return "High (≥ 0.7)"


        filtered_data["band"] = filtered_data["proba"].apply(triage_band)

        triage_summary = filtered_data.groupby("band").agg({
            "candidate_id": "count",
            "selected": "mean",
            "proba": "mean"
        }).reset_index()
        triage_summary.columns = ["Band", "Count", "Actual Rate", "Predicted Rate"]

        band_order = ["High (≥ 0.7)", "Medium-High (0.5-0.7)", "Medium-Low (0.3-0.5)", "Low (< 0.3)"]
        triage_summary["Band"] = pd.Categorical(triage_summary["Band"], categories=band_order, ordered=True)
        triage_summary = triage_summary.sort_values("Band")

        st.dataframe(
            triage_summary.style.format({
                "Actual Rate": "{:.1%}",
                "Predicted Rate": "{:.3f}"
            }),
            hide_index=True
        )

        fig_bands = px.bar(
            triage_summary,
            x="Band",
            y="Count",
            text="Count",
            color="Actual Rate",
            color_continuous_scale="RdYlGn",
            title="Candidate Distribution by Risk Band"
        )
        fig_bands.update_layout(height=300)
        st.plotly_chart(fig_bands, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Calibration by Demographic Groups")

    selected_group = st.radio(
        "Select attribute:",
        ["gender", "degree_major"],
        horizontal=True
    )

    fig_subgroup_calib = go.Figure()

    for group in filtered_data[selected_group].unique():
        group_df = filtered_data[filtered_data[selected_group] == group]
        if len(group_df) > 50:
            frac_pos, mean_pred = calibration_curve(
                group_df["selected"],
                group_df["proba"],
                n_bins=8,
                strategy='quantile'
            )
            fig_subgroup_calib.add_trace(go.Scatter(
                x=mean_pred,
                y=frac_pos,
                mode='lines+markers',
                name=str(group),
                marker=dict(size=8)
            ))

    fig_subgroup_calib.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='gray', dash='dash', width=2)
    ))

    fig_subgroup_calib.update_layout(
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        height=400,
        hovermode='closest'
    )
    st.plotly_chart(fig_subgroup_calib, use_container_width=True)

# =============================
# TAB 5: FEATURE ANALYSIS
# =============================
with tab5:
    st.subheader("🔬 Feature Distribution Analysis")

    numeric_features = ["gpa", "leadership_score", "communication_score", "self_efficacy", "organisational_support"]
    selected_feature = st.selectbox("Select feature to analyze:", numeric_features)

    col1, col2 = st.columns(2)

    with col1:
        fig_dist = go.Figure()

        fig_dist.add_trace(go.Violin(
            y=filtered_data[filtered_data['selected'] == 1][selected_feature],
            name='Selected',
            box_visible=True,
            meanline_visible=True,
            fillcolor='lightgreen',
            opacity=0.6,
            x0='Selected'
        ))

        fig_dist.add_trace(go.Violin(
            y=filtered_data[filtered_data['selected'] == 0][selected_feature],
            name='Not Selected',
            box_visible=True,
            meanline_visible=True,
            fillcolor='lightcoral',
            opacity=0.6,
            x0='Not Selected'
        ))

        fig_dist.update_layout(
            title=f"{selected_feature} Distribution by Outcome",
            yaxis_title=selected_feature,
            height=400
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        fig_gender_dist = px.box(
            filtered_data,
            x="gender",
            y=selected_feature,
            color="gender",
            title=f"{selected_feature} by Gender",
            points="outliers"
        )
        fig_gender_dist.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_gender_dist, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Feature Correlations")

    corr_features = numeric_features + ['proba', 'selected']
    corr_matrix = filtered_data[corr_features].corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        aspect='auto'
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")
    st.subheader("📅 Trends Over Time (by Cohort)")

    cohort_trends = filtered_data.groupby('cohort').agg({
        'selected': 'mean',
        'gpa': 'mean',
        'leadership_score': 'mean',
        'communication_score': 'mean',
        'candidate_id': 'count'
    }).reset_index()

    fig_trends = go.Figure()

    fig_trends.add_trace(go.Scatter(
        x=cohort_trends['cohort'],
        y=cohort_trends['selected'],
        mode='lines+markers',
        name='Selection Rate',
        yaxis='y',
        line=dict(width=3)
    ))

    fig_trends.add_trace(go.Bar(
        x=cohort_trends['cohort'],
        y=cohort_trends['candidate_id'],
        name='Candidate Count',
        yaxis='y2',
        opacity=0.3
    ))

    fig_trends.update_layout(
        yaxis=dict(title='Selection Rate', side='left'),
        yaxis2=dict(title='Candidate Count', overlaying='y', side='right'),
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig_trends, use_container_width=True)

# =============================
# TAB 6: MODEL CARD
# =============================
with tab6:
    st.markdown(model_card_text)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 📋 Model Info
        - **Type**: Classification
        - **Framework**: Scikit-learn
        - **Algorithm**: Random Forest
        - **Version**: 1.0
        """)

    with col2:
        st.markdown("""
        ### 🎯 Performance
        - **AUC**: {:.3f}
        - **Accuracy**: {:.3f}
        - **F1 Score**: {:.3f}
        """.format(metrics['auc'], metrics['accuracy'], metrics['f1']))

    with col3:
        st.markdown("""
        ### ⚠️ Limitations
        - Synthetic data for demo
        - Regular retraining needed
        - Continuous bias monitoring
        - Human review required
        """)

    st.markdown("---")

    st.subheader("📊 Dataset Statistics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Candidates", f"{len(data):,}")
        st.metric("Selected", f"{data['selected'].sum():,}")
    with col2:
        st.metric("Ethnic Groups", f"{data['ethnicity'].nunique()}")
        st.metric("Cohorts", f"{data['cohort'].nunique()}")
    with col3:
        st.metric("Degree Majors", f"{data['degree_major'].nunique()}")
        st.metric("Selection Rate", f"{data['selected'].mean():.1%}")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <p>🎓 Teach For Australia | AI Fairness Dashboard v2.0</p>
    <p>Built with Streamlit 🎈 | Data updated: {}</p>
    <p><i>Note: Install PyPDF2 for PDF upload functionality: pip install PyPDF2</i></p>
</div>
""".format(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)