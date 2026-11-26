
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("student_placement_best_model.joblib")


# Average profile of placed students
placed_averages = {
    "CGPA": 7.10,
    "SSC_Percentage": 71.38,
    "HSC_Percentage": 70.55,
    "Internships": 2.02,
    "Projects": 3.02,
    "Certifications": 2.49,
    "Communication_Skill": 5.62,
    "Technical_Skill": 5.43,
    "Aptitude_Score": 65.60,
    "Coding_Score": 60.03,
    "Attendance_Percentage": 82.07
}


display_names = {
    "CGPA": "CGPA",
    "SSC_Percentage": "SSC Percentage",
    "HSC_Percentage": "HSC Percentage",
    "Internships": "Internships",
    "Projects": "Projects",
    "Certifications": "Certifications",
    "Communication_Skill": "Communication Skill",
    "Technical_Skill": "Technical Skill",
    "Aptitude_Score": "Aptitude Score",
    "Coding_Score": "Coding Score",
    "Attendance_Percentage": "Attendance"
}


recommendations = {
    "CGPA": "Focus on maintaining or improving your academic performance.",
    "SSC_Percentage": "Strengthen your academic fundamentals and core concepts.",
    "HSC_Percentage": "Improve your academic foundation through regular revision.",
    "Internships": "Gain more practical industry exposure through internships.",
    "Projects": "Build more real-world projects and add them to your portfolio.",
    "Certifications": "Complete relevant technical certifications.",
    "Communication_Skill": "Practice communication through presentations, discussions and mock interviews.",
    "Technical_Skill": "Strengthen your technical skills through coding and practical projects.",
    "Aptitude_Score": "Practice quantitative aptitude, logical reasoning and verbal ability.",
    "Coding_Score": "Practice programming problems regularly on coding platforms.",
    "Attendance_Percentage": "Maintain consistent attendance and participation."
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Collect input values
    data = {
        "CGPA": float(request.form["CGPA"]),
        "SSC_Percentage": float(request.form["SSC_Percentage"]),
        "HSC_Percentage": float(request.form["HSC_Percentage"]),
        "Internships": int(request.form["Internships"]),
        "Projects": int(request.form["Projects"]),
        "Certifications": int(request.form["Certifications"]),
        "Communication_Skill": float(request.form["Communication_Skill"]),
        "Technical_Skill": float(request.form["Technical_Skill"]),
        "Aptitude_Score": float(request.form["Aptitude_Score"]),
        "Coding_Score": float(request.form["Coding_Score"]),
        "Attendance_Percentage": float(request.form["Attendance_Percentage"]),
        "Branch": request.form["Branch"],
        "Gender": request.form["Gender"]
    }

    # Create DataFrame
    input_df = pd.DataFrame([data])

    # Prediction
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    not_placed_probability = probabilities[0] * 100
    placed_probability = probabilities[1] * 100

    if prediction == 1:
        status = "PLACED"
    else:
        status = "NOT PLACED"

    # -----------------------------
    # Gap and strength analysis
    # -----------------------------

    strengths = []
    gaps = []

    for feature, average in placed_averages.items():

        student_value = data[feature]

        difference = student_value - average

        item = {
            "feature": display_names[feature],
            "student_value": round(student_value, 2),
            "average": round(average, 2),
            "difference": round(difference, 2)
        }

        if difference >= 0:
            strengths.append(item)
        else:
            gaps.append(item)

    # Sort gaps by largest negative difference
    gaps = sorted(gaps, key=lambda x: x["difference"])

    # Sort strengths by largest positive difference
    strengths = sorted(
        strengths,
        key=lambda x: x["difference"],
        reverse=True
    )

    # Top areas for improvement
    improvement_areas = gaps[:5]

    # Recommendations
    recommendation_list = []

    for item in improvement_areas:
        for key, name in display_names.items():

            if name == item["feature"]:
                recommendation_list.append({
                    "area": item["feature"],
                    "recommendation": recommendations[key]
                })
                break

    # -----------------------------
    # Profile readiness score
    # -----------------------------

    numeric_features = list(placed_averages.keys())

    above_average_count = sum(
        1
        for feature in numeric_features
        if data[feature] >= placed_averages[feature]
    )

    readiness_score = round(
        (above_average_count / len(numeric_features)) * 100,
        1
    )

    return render_template(
        "result.html",
        status=status,
        placed_probability=round(placed_probability, 2),
        not_placed_probability=round(not_placed_probability, 2),
        data=data,
        student_profile=data,
        strengths=strengths,
        gaps=gaps,
        improvement_areas=improvement_areas,
        recommendations=recommendation_list,
        readiness_score=readiness_score
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
