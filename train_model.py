import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. LOAD DATASET
# ============================================================
print("\n==========================================")
print("TOXIC COMMENT DETECTION USING SVM")
print("==========================================")
print("\nLoading dataset...")
df = pd.read_csv("train.csv")
print("\nDataset loaded successfully!")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

# ============================================================
# 2. DISPLAY DATASET INFORMATION
# ============================================================
print("\nDataset columns:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# ============================================================
# 3. DEFINE TOXICITY CATEGORIES
# ============================================================
categories = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# ============================================================
# 4. SELECT REQUIRED COLUMNS
# ============================================================
df = df[["comment_text"] + categories]

# ============================================================
# 5. REMOVE MISSING VALUES
# ============================================================
print("\nChecking missing values...")
print(df.isnull().sum())
df = df.dropna(subset=["comment_text"])
print("\nDataset after removing missing comments:")
print("Number of rows:", len(df))

# ============================================================
# 6. CREATE FOLDERS
# ============================================================
if not os.path.exists("models"):
    os.makedirs("models")
if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists("results/confusion_matrices"):
    os.makedirs("results/confusion_matrices")
print("\nRequired folders created.")

# ============================================================
# 7. DATASET DISTRIBUTION
# ============================================================
print("\n==========================================")
print("TOXICITY CATEGORY DISTRIBUTION")
print("==========================================")
category_counts = {}
for category in categories:
    count = df[category].sum()
    category_counts[category] = count
    print(category, ":", count)

# ============================================================
# 8. PLOT CATEGORY DISTRIBUTION
# ============================================================
plt.figure(figsize=(10, 6))
plt.bar(
    category_counts.keys(),
    category_counts.values()
)
plt.title("Toxicity Category Distribution")
plt.xlabel("Toxicity Category")
plt.ylabel("Number of Comments")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/category_distribution.png")
plt.close()
print("\nCategory distribution graph saved.")

# ============================================================
# 9. STORE MODEL RESULTS
# ============================================================
results = []

# ============================================================
# 10. TRAIN EACH SVM MODEL
# ============================================================
for category in categories:
    print("\n")
    print("==========================================")
    print("TRAINING:", category.upper())
    print("==========================================")

    # INPUT
    X = df["comment_text"].astype(str)

    # TARGET
    y = df[category]

    print("\nPositive samples:", y.sum())
    print("Negative samples:", len(y) - y.sum())

    # TRAIN TEST SPLIT
    print("\nSplitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # ========================================================
    # 11. TF-IDF
    # ========================================================
    print("\nCreating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Number of TF-IDF features:", X_train_tfidf.shape[1])

    # ========================================================
    # 12. SVM MODEL
    # ========================================================
    print("\nTraining SVM...")
    model = LinearSVC(
        class_weight="balanced",
        random_state=42
    )
    model.fit(
        X_train_tfidf,
        y_train
    )
    print("SVM training completed.")

    # ========================================================
    # 13. PREDICTION
    # ========================================================
    print("\nMaking predictions...")
    y_pred = model.predict(
        X_test_tfidf
    )

    # ========================================================
    # 14. EVALUATION
    # ========================================================
    accuracy = accuracy_score(
        y_test,
        y_pred
    )
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # ========================================================
    # 15. PRINT RESULTS
    # ========================================================
    print("\n------------------------------------------")
    print("MODEL PERFORMANCE")
    print("------------------------------------------")
    print(
        "Accuracy  :",
        round(accuracy * 100, 2),
        "%"
    )
    print(
        "Precision :",
        round(precision * 100, 2),
        "%"
    )
    print(
        "Recall    :",
        round(recall * 100, 2),
        "%"
    )
    print(
        "F1 Score  :",
        round(f1 * 100, 2),
        "%"
    )

    # ========================================================
    # 16. SAVE RESULTS
    # ========================================================
    results.append({
        "Category": category,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1 Score": round(f1 * 100, 2)
    })

    # ========================================================
    # 17. CONFUSION MATRIX
    # ========================================================
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nConfusion Matrix:")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Non-Toxic",
            "Toxic"
        ]
    )

    display.plot()
    plt.title("Confusion Matrix - " + category)
    plt.tight_layout()
    plt.savefig(
        "results/confusion_matrices/"
        + category
        + "_confusion_matrix.png"
    )
    plt.close()

    # ========================================================
    # 18. SAVE MODEL
    # ========================================================
    model_file = (
        "models/"
        + category
        + "_model.pkl"
    )

    vectorizer_file = (
        "models/"
        + category
        + "_vectorizer.pkl"
    )

    joblib.dump(
        model,
        model_file
    )

    joblib.dump(
        vectorizer,
        vectorizer_file
    )

    print("\nModel saved:", model_file)
    print("Vectorizer saved:", vectorizer_file)

# ============================================================
# 19. CREATE RESULTS DATAFRAME
# ============================================================
results_df = pd.DataFrame(results)

print("\n")
print("==========================================")
print("FINAL MODEL RESULTS")
print("==========================================")
print(results_df)

# ============================================================
# 20. SAVE RESULTS TO CSV
# ============================================================
results_df.to_csv(
    "results/model_results.csv",
    index=False
)

print("\nResults saved to results/model_results.csv")

# ============================================================
# 21. PERFORMANCE GRAPH
# ============================================================
x = np.arange(len(categories))
width = 0.2

plt.figure(figsize=(12, 7))

plt.bar(
    x - width * 1.5,
    results_df["Accuracy"],
    width,
    label="Accuracy"
)

plt.bar(
    x - width / 2,
    results_df["Precision"],
    width,
    label="Precision"
)

plt.bar(
    x + width / 2,
    results_df["Recall"],
    width,
    label="Recall"
)

plt.bar(
    x + width * 1.5,
    results_df["F1 Score"],
    width,
    label="F1 Score"
)

plt.xlabel("Toxicity Category")
plt.ylabel("Score (%)")
plt.title("SVM Model Performance Comparison")

plt.xticks(
    x,
    categories,
    rotation=45
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "results/performance_graph.png"
)

plt.close()
print("\nPerformance graph saved.")

# ============================================================
# 22. F1 SCORE GRAPH
# ============================================================
plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Category"],
    results_df["F1 Score"]
)

plt.xlabel("Toxicity Category")
plt.ylabel("F1 Score (%)")
plt.title("F1 Score of SVM Models")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "results/f1_score_graph.png"
)

plt.close()
print("F1 score graph saved.")

# ============================================================
# 23. FINAL MESSAGE
# ============================================================
print("\n")
print("==========================================")
print("TRAINING COMPLETED SUCCESSFULLY!")
print("==========================================")

print("\nGenerated files:")
print("\nModels:")
print("✓ Six SVM models")
print("✓ Six TF-IDF vectorizers")

print("\nResults:")
print("✓ model_results.csv")
print("✓ performance_graph.png")
print("✓ f1_score_graph.png")
print("✓ category_distribution.png")
print("✓ Six confusion matrix graphs")

print("\nNext step:")
print("We will create a separate prediction program")
print("where you can enter your own comment.")

print("\n==========================================")
