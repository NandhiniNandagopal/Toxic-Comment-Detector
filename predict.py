import joblib


# ============================================================
# 1. TOXICITY CATEGORIES
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
# 2. DISPLAY NAMES
# ============================================================

display_names = {
    "toxic": "Toxic",
    "severe_toxic": "Severe Toxic",
    "obscene": "Obscene",
    "threat": "Threat",
    "insult": "Insult",
    "identity_hate": "Identity Hate"
}


# ============================================================
# 3. LOAD MODELS
# ============================================================

print("Loading trained models...")

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

    models[category] = joblib.load(model_path)

    vectorizers[category] = joblib.load(
        vectorizer_path
    )


print("All models loaded successfully!")


# ============================================================
# 4. USER INPUT
# ============================================================

print("\n")
print("==========================================")
print("TOXIC COMMENT DETECTION SYSTEM")
print("==========================================")

print("\nEnter a comment to analyze.")

comment = input("\nComment: ")


# ============================================================
# 5. CHECK EMPTY INPUT
# ============================================================

if comment.strip() == "":

    print("\nPlease enter a valid comment.")

else:

    predictions = {}


    # ========================================================
    # 6. PREDICT EACH CATEGORY
    # ========================================================

    for category in categories:

        model = models[category]

        vectorizer = vectorizers[category]


        # Convert comment to TF-IDF

        comment_tfidf = vectorizer.transform(
            [comment]
        )


        # Predict

        prediction = model.predict(
            comment_tfidf
        )[0]


        predictions[category] = prediction


    # ========================================================
    # 7. DISPLAY RESULTS
    # ========================================================

    print("\n")
    print("==========================================")
    print("PREDICTION RESULT")
    print("==========================================")


    detected = []


    for category in categories:

        if predictions[category] == 1:

            detected.append(category)


    # ========================================================
    # 8. OVERALL RESULT
    # ========================================================

    if "toxic" in detected:

        print("\n🚨 TOXIC COMMENT DETECTED")

    else:

        print("\n✅ NON-TOXIC COMMENT")


    # ========================================================
    # 9. CATEGORY RESULTS
    # ========================================================

    print("\nCategory Results:")
    print("------------------------------------------")


    for category in categories:

        name = display_names[category]


        if predictions[category] == 1:

            print(
                "🔴",
                name,
                ": YES"
            )

        else:

            print(
                "🟢",
                name,
                ": NO"
            )


    # ========================================================
    # 10. DETECTED CATEGORIES
    # ========================================================

    print("\n")
    print("==========================================")
    print("DETECTED TOXICITY")
    print("==========================================")


    if len(detected) > 0:

        for category in detected:

            print(
                "•",
                display_names[category]
            )

    else:

        print("No toxic category detected.")


    print("\n==========================================")
    print("Analysis completed.")
    print("==========================================")
