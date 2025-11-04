from model_training.config_model import model_lg, model_dt
import pandas as pd

def get_incident_risk_list(student_incident_list):
    print(student_incident_list)
    return []
    """
    results = []

    for student in student_incident_list:
        student_id = student.get("student_id", "unknown")

        # Run logistic model prediction
        pred_class, confidence, remark = get_student_reoffend_status(student)

        # Count how many incident types are repeated (>=2)
        repeat_count = student.get('repeated_incident_count')        

        results.append({
            "student_id": student_id,
            "repeat_count": repeat_count,
            "risk_score": round(float(confidence), 3),
            "ongoing_incident_count": student.get('ongoing_incident_count'),
            "incident_prediction": int(pred_class),
            "risk_label": (
                "High Risk" if confidence >= 0.7 else
                "Moderate Risk" if confidence >= 0.4 else
                "Low Risk"
            )
        })

    return results
    """

def get_student_reoffend_status(student_data):
    """
    Predicts if a single student is likely to have another incident.
    Uses the trained logistic model (model_lg).
    """
    # Extract relevant inputs safely (default to 0 if missing)
    repeated_incident_count = student_data.get("repeated_incident_count", 0)
    major_count = student_data.get("major_count", 0)
    minor_count = student_data.get("minor_count", 0)
    ongoing_incident_count = student_data.get("ongoing_incident_count", 0)


    # Predict using the logistic model
    pred_class, confidence, remark = model_lg.predict(
        repeated_incident_count=repeated_incident_count,
        major_count=major_count,
        minor_count=minor_count,
        ongoing_incident_count=ongoing_incident_count
    )

    return pred_class, confidence, remark

def get_student_incident_risk_level(student_data):
    # Filter features to match the model columns
    repeated_incident_count = float(student_data.get("repeated_incident_count", 0))
    major_count = float(student_data.get("major_count", 0))
    minor_count = float(student_data.get("minor_count", 0))

    # 🧠 Run prediction using your trained decision tree model
    predicted_class, predicted_prob, remark = model_dt.predict(
        repeated_incident_count=repeated_incident_count,
        major_count=major_count,
        minor_count=minor_count
    )

    # 🧾 Return structured result
    return {
        "student_id": student_data.get("student_id"),
        "repeated_incident_count": repeated_incident_count,
        "major_count": major_count,
        "minor_count": minor_count,
        "ongoing_incident_count": float(student_data.get("ongoing_incident_count", 0)),
        "risk_score": round(float(predicted_prob), 3),
        "incident": int(predicted_class),
        "remark": remark
    }
    
def get_student_incident_remarks(student_data):
    student_data = student_data['student'][0]
    
    repeated_incident_count = float(student_data.get("repeated_incident_count", 0))
    major_count = float(student_data.get("major_count", 0))
    minor_count = float(student_data.get("minor_count", 0))
    ongoing_count = student_data.get("ongoing_incident_count", 0)

    # 🧠 Run prediction using your trained decision tree model and logistic regression
    predicted_class_lg, predicted_prob_lg, remark_lg = get_student_reoffend_status(student_data)
    
    # 🧾 Return structured result
    return {
        "student_id": student_data.get("student_id"),
        "repeated_incident_count": repeated_incident_count,
        "major_count": major_count,
        "minor_count": minor_count,
        "ongoing_incident_count": ongoing_count,
        "risk_score": round(float(predicted_prob_lg), 3),
        "risk_label": remark_lg,
        "incident": int(predicted_class_lg),
        "remark": remark_lg
    }
