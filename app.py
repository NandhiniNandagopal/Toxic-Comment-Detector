import streamlit as st
import joblib
import pandas as pd
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Toxic Comment Detection",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🛡️ Toxic Comment Detection System")

st.write(
    "An NLP-based multi-label toxicity detection system "
    "using TF-IDF and Support Vector Machine (SVM)."
)


# ============================================================
# TOXICITY CATEGORIES
# ============================================================

categories = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


display_names = {
    "toxic": "Toxic",
    "severe_toxic": "Severe Toxic",
    "obscene": "Obscene",
    "threat": "Threat",
    "insult": "Insult",
    "identity_hate": "Identity Hate"
}


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {}
    vectorizers = {}

    for category in categories:

        model_path = (
            "models/"
            + category
            + "_model.pkl"
        )

        vectorizer_path = (
            "models/"
            + category
            + "_vectorizer.pkl"
        )

        models[category] = joblib.load(
            model_path
        )

        vectorizers[category] = joblib.load(
            vectorizer_path
        )

    return models, vectorizers


models, vectorizers = load_models()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 About")

st.sidebar.write(
    "This application detects different types "
    "of toxic comments using six independent "
    "SVM classifiers."
)

st.sidebar.write("")

st.sidebar.write("**Technologies Used:**")

st.sidebar.write("• Python")
st.sidebar.write("• Scikit-learn")
st.sidebar.write("• TF-IDF")
st.sidebar.write("• Linear SVM")
st.sidebar.write("• Streamlit")


# ============================================================
# USER INPUT
# ============================================================

st.subheader("💬 Enter a Comment")

comment = st.text_area(
    "Type or paste a comment below:",
    height=150,
    placeholder="Enter your comment here..."
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Comment",
    use_container_width=True
):

    if comment.strip() == "":

        st.warning(
            "Please enter a comment before analyzing."
        )

    else:

        predictions = {}


        # ====================================================
        # MAKE PREDICTIONS
        # ====================================================

        for category in categories:

            model = models[category]

            vectorizer = vectorizers[category]


            comment_tfidf = vectorizer.transform(
                [comment]
            )


            prediction = model.predict(
                comment_tfidf
            )[0]


            predictions[category] = prediction


        # ====================================================
        # DISPLAY OVERALL RESULT
        # ====================================================

        st.subheader("📊 Prediction Result")


        detected = []

        for category in categories:

            if predictions[category] == 1:

                detected.append(category)


        if "toxic" in detected:

            st.error(
                "🚨 Toxic Comment Detected"
            )

        else:

            st.success(
                "✅ Non-Toxic Comment"
            )


        # ====================================================
        # CATEGORY RESULTS
        # ====================================================

        st.subheader("🔎 Toxicity Categories")


        col1, col2, col3 = st.columns(3)


        columns = [
            col1,
            col2,
            col3,
            col1,
            col2,
            col3
        ]


        for i, category in enumerate(categories):

            with columns[i]:

                name = display_names[category]


                if predictions[category] == 1:

                    st.error(
                        "🔴 " + name + "\n\nDetected"
                    )

                else:

                    st.success(
                        "🟢 " + name + "\n\nNot Detected"
                    )


        # ====================================================
        # DETECTED CATEGORIES
        # ====================================================

        st.subheader("📋 Detected Categories")


        if len(detected) > 0:

            for category in detected:

                st.write(
                    "🔴 "
                    + display_names[category]
                )

        else:

            st.write(
                "No toxic category detected."
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.subheader("📈 Model Performance")


results_file = "results/model_results.csv"


if os.path.exists(results_file):

    results_df = pd.read_csv(
        results_file
    )


    st.dataframe(
        results_df,
        use_container_width=True
    )


    st.subheader(
        "SVM Performance Comparison"
    )


    chart_data = results_df.set_index(
        "Category"
    )[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    ]


    st.bar_chart(
        chart_data
    )


# ============================================================
# GRAPHS
# ============================================================

st.subheader("📊 Generated Graphs")


col1, col2 = st.columns(2)


with col1:

    graph1 = (
        "results/category_distribution.png"
    )

    if os.path.exists(graph1):

        st.image(
            graph1,
            caption="Toxicity Category Distribution"
        )


with col2:

    graph2 = (
        "results/performance_graph.png"
    )

    if os.path.exists(graph2):

        st.image(
            graph2,
            caption="SVM Model Performance"
        )


graph3 = "results/f1_score_graph.png"


if os.path.exists(graph3):

    st.image(
        graph3,
        caption="F1 Score of SVM Models"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Toxic Comment Detection using NLP, TF-IDF and SVM"
)
