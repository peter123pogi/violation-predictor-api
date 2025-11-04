import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score
)

# -------------------------------------------------------------------
# 🔧 Utility: Convert NumPy → Native Python for JSON serialization
# -------------------------------------------------------------------
def make_json_safe(data):
    """Recursively convert NumPy and pandas types into JSON-serializable Python types."""
    if isinstance(data, dict):
        return {k: make_json_safe(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_json_safe(v) for v in data]
    elif isinstance(data, (np.integer, np.int64)):
        return int(data)
    elif isinstance(data, (np.floating, np.float64)):
        return float(data)
    elif pd.isna(data):
        return None
    else:
        return data

def assign_action(row):
    """Rule-based disciplinary action levels (1–11) based on severity and frequency."""
    severity = row.get("avg_severity", 0)
    freq = row.get("frequency", 0)

    if severity < 0.2 and freq <= 1:
        return "1"   # Verbal Warning
    elif severity < 0.3 and freq <= 2:
        return "2"   # Written Warning
    elif severity < 0.4 and freq <= 3:
        return "3"   # Conference with Parents/Guardians
    elif severity < 0.5 and freq <= 3:
        return "4"   # Restitution
    elif severity < 0.55 and freq <= 4:
        return "5"   # Confiscation
    elif severity < 0.6 and freq <= 4:
        return "6"   # Demerit
    elif severity < 0.7:
        return "7"   # One-day Suspension
    elif severity < 0.75:
        return "8"   # Two-day Suspension
    elif severity < 0.8:
        return "9"   # Exclusion (temporary removal from class)
    elif severity < 0.9:
        return "10"  # Dismissal / Non-readmission
    else:
        return "11"  # Expulsion

# -------------------------------------------------------------------
# 🧠 Violation Predictor Class (with Auto-Computed behavior_ratio)
# -------------------------------------------------------------------
class ViolationPredictor:
    def __init__(self, csv_file, dataset_type='training'):
        self.logistic_model = LogisticRegression(max_iter=1000)
        self.tree_model = DecisionTreeClassifier()
        self.encoder = LabelEncoder()
        self.features = None
        self.student_data = None
        self.train_models(csv_file, dataset_type)

    # ---------------------------------------------------------------
    # 🧩 Data Preparation & Training
    # ---------------------------------------------------------------
    def load_and_prepare(self, data: pd.DataFrame):
        """Prepare dataset for training both models."""
        df = data.copy()
        if "next_repeated_violation" not in df.columns:
            raise ValueError("Missing column: 'next_repeated_violation'")

        # Encode violation labels for Decision Tree
        df["violation_encoded"] = self.encoder.fit_transform(df["next_repeated_violation"])

        X = df[[
            "frequency",
            "repeated_major_count",
            "repeated_minor_count",
            "ongoing_incident_count",
            "total_incidents",
            "avg_severity",
            "is_major",
            "no_violation_count",
            "behavior_ratio"
        ]]
        y_risk = df["will_reoffend_same_violation"]

        # Assign and encode disciplinary actions
        df["disciplinary_action_level"] = df.apply(assign_action, axis=1)
        y_violation = df["disciplinary_action_level"]

        self.features = X.columns
        self.student_data = df
        return X, y_risk, y_violation
        
    def train_models(self, csv_file, dataset_type):
        type = f'{dataset_type}/' if dataset_type != '' else ''
        
        X, y_risk, y_violation = self.load_and_prepare(pd.read_csv(f"./dataset/{type}{csv_file}.csv"))

        X_train, X_test, y_risk_train, y_risk_test = train_test_split(
            X, y_risk, test_size=0.2, random_state=42
        )
        _, _, y_violation_train, y_violation_test = train_test_split(
            X, y_violation, test_size=0.2, random_state=42
        )

        self.logistic_model.fit(X_train, y_risk_train)
        self.tree_model.fit(X_train, y_violation_train)
        
        """
        print("🔹 Logistic Regression — Reoffense Risk")
        print("Logistic Regression Accuracy: ", accuracy_score(y_risk_test, self.logistic_model.predict(X_test)))
        print()
        print(classification_report(y_risk_test, self.logistic_model.predict(X_test)))
        print()
        print("🔹 Decision Tree — Fair Disciplinary Action Prediction")
        print("Decision Tree Accuracy: ", accuracy_score(y_violation_test, self.tree_model.predict(X_test)))
        print()
        print(classification_report(y_violation_test, self.tree_model.predict(X_test)))
        """
        
        

    # ---------------------------------------------------------------
    # ⚙️ Auto Compute Features
    # ---------------------------------------------------------------
    def _auto_compute_avg_severity(self, student: dict) -> float:
        major = student.get("repeated_major_count", 0)
        minor = student.get("repeated_minor_count", 0)
        ongoing = student.get("ongoing_incident_count", 0)
        total = student.get("total_incidents", 0)
        freq = student.get("frequency", 0)
        if total == 0 and (major + minor + ongoing) == 0:
            return 0.0
        avg = ((2 * major) + (1 * minor) + (1.5 * ongoing) + (0.3 * freq)) / (total + 1)
        return round(min(avg, 1.0), 2)

    def _auto_compute_behavior_ratio(self, student: dict) -> float:
        total = student.get("total_incidents", 0)
        no_v = student.get("no_violation_count", 0)
        ratio = no_v / (total + no_v + 1)
        return round(ratio, 3)

    # ---------------------------------------------------------------
    # 🧠 Generate Risk Watchlist
    # ---------------------------------------------------------------
    def generate_watchlist(self, input_data=None):
        """Generate a risk-ranked student watchlist with balanced correction."""
        if input_data is not None:
            if isinstance(input_data, dict):
                X_full = pd.DataFrame([input_data])
            elif isinstance(input_data, list) and all(isinstance(i, dict) for i in input_data):
                X_full = pd.DataFrame(input_data)
            elif isinstance(input_data, pd.DataFrame):
                X_full = input_data.copy()
            else:
                raise TypeError("input_data must be a dict, list of dicts, or DataFrame.")
        else:
            if self.student_data is None:
                raise ValueError("No data provided and no training data available.")
            X_full = self.student_data[self.features]

        # Fill and compute missing values
        for feature in self.features:
            if feature not in X_full.columns:
                X_full[feature] = 0
        for idx, row in X_full.iterrows():
            s = row.to_dict()
            X_full.at[idx, "avg_severity"] = float(self._auto_compute_avg_severity(s))
            X_full.at[idx, "behavior_ratio"] = float(self._auto_compute_behavior_ratio(s))

        has_id = "student_id" in X_full.columns
        valid = X_full[self.features]
        raw_probs = self.logistic_model.predict_proba(valid)[:, 1]
        watchlist = []

        for i, idx in enumerate(valid.index):
            row = X_full.loc[idx]
            base = float(raw_probs[i])
            br = float(row["behavior_ratio"])
            ti = float(row["total_incidents"])

            # ✅ Balanced correction (behavior dominates)
            incident_factor = 1 + min(ti / 300, 0.25)  # cap at +25%
            behavior_factor = 1 - (0.9 * br)           # stronger reward for good behavior
            correction = behavior_factor * incident_factor
            adjusted = min(max(base * correction, 0), 1)

            risk = (
                "Low" if adjusted < 0.33
                else "Medium" if adjusted < 0.55
                else "High" if adjusted < 0.75
                else "Very High" if adjusted < 0.9
                else "Critical"
            )

            watchlist.append({
                "student_id": str(row["student_id"]) if has_id else f"Student_{i+1}",
                "probability_of_reoffense": round(adjusted, 4),
                "frequency": int(row["frequency"]),
                "is_major": int(row["is_major"]),
                "no_violation_count": int(row["no_violation_count"]),
                "behavior_ratio": br,
                "avg_severity": float(row["avg_severity"]),
                "risk_level": risk,
                "ongoing_incident_count": int(row["ongoing_incident_count"]),
                "total_incident": int(row["total_incidents"]),
                "last_incident": row["last_incident"],
                "note": "Predicted successfully",
            })

        order = {"Critical": 5, "Very High": 4, "High": 3, "Medium": 2, "Low": 1, "No Data": 0}
        sorted_watchlist = sorted(
            watchlist,
            key=lambda e: (order.get(e["risk_level"], 0), e["probability_of_reoffense"]),
            reverse=True,
        )
        return make_json_safe(sorted_watchlist)

    # ---------------------------------------------------------------
    # 🎯 Predict Single Student Risk (also using balanced correction)
    # ---------------------------------------------------------------
    def predict_reoffense_risk(self, student_input):
        if isinstance(student_input, dict):
            X_single = pd.DataFrame([student_input])
        elif isinstance(student_input, pd.DataFrame):
            X_single = student_input.copy()
        else:
            raise TypeError("student_input must be a dict or DataFrame.")

        for feature in self.features:
            if feature not in X_single.columns:
                X_single[feature] = 0
        for idx, row in X_single.iterrows():
            s = row.to_dict()
            X_single.at[idx, "avg_severity"] = float(self._auto_compute_avg_severity(s))
            X_single.at[idx, "behavior_ratio"] = float(self._auto_compute_behavior_ratio(s))

        X_single = X_single[self.features]
        base = float(self.logistic_model.predict_proba(X_single)[0][1])
        br = float(X_single.iloc[0]["behavior_ratio"])
        ti = float(X_single.iloc[0]["total_incidents"])

        # ✅ Balanced correction
        incident_factor = 1 + min(ti / 300, 0.25)
        behavior_factor = 1 - (0.9 * br)
        correction = behavior_factor * incident_factor
        adjusted = min(max(base * correction, 0), 1)

        if adjusted >= 0.85:
            risk_level = "Critical"
        elif adjusted >= 0.70:
            risk_level = "Very High"
        elif adjusted >= 0.55:
            risk_level = "High"
        elif adjusted >= 0.40:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        recommendation = self._generate_recommendation(risk_level)

        return make_json_safe({
            "student_id": str(student_input.get("student_id", "Unknown")),
            "probability_of_reoffense": round(adjusted, 4),
            "risk_level": risk_level,
            "no_violation_count": int(X_single.iloc[0]["no_violation_count"]),
            "behavior_ratio": round(float(X_single.iloc[0]["behavior_ratio"]), 3),
            "computed_avg_severity": float(X_single.iloc[0]["avg_severity"]),
            "note": "Predicted successfully",
            "recommendation": recommendation
        })

    # ---------------------------------------------------------------
    # 🔮 Predict All Likely Violations (with recommendations)
    # ---------------------------------------------------------------
    def predict_likely_violations_from_input(self, input_data):
        """Predict likely future violations dynamically and provide detailed recommendations."""
        if isinstance(input_data, dict):
            X_new = pd.DataFrame([input_data])
        elif isinstance(input_data, list) and all(isinstance(i, dict) for i in input_data):
            X_new = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            X_new = input_data.copy()
        else:
            raise TypeError("input_data must be a dict, list of dicts, or DataFrame.")

        for feature in self.features:
            if feature not in X_new.columns:
                X_new[feature] = 0

        for idx, row in X_new.iterrows():
            s = row.to_dict()
            X_new.at[idx, "avg_severity"] = float(self._auto_compute_avg_severity(s))
            X_new.at[idx, "behavior_ratio"] = float(self._auto_compute_behavior_ratio(s))

        X_new = X_new[self.features]
        results = []

        for i in range(len(X_new)):
            row = X_new.iloc[i].to_dict()
            probs = self.tree_model.predict_proba(X_new.iloc[[i]])[0]
            violations = self.encoder.inverse_transform(np.arange(len(probs)))
            ranked = (
                pd.DataFrame({"violation": violations, "probability": probs})
                .sort_values(by="probability", ascending=False)
                .reset_index(drop=True)
                .to_dict(orient="records")
            )

            avg = float(row["avg_severity"])
            ratio = float(row["behavior_ratio"])
            top_v = ranked[0]["violation"]

            # 🔹 Use model probability instead of avg_severity thresholds
            prob_top = ranked[0]["probability"]

            if prob_top >= 0.85:
                rec = (
                    f"The student shows a **critical likelihood** of repeating '{top_v}' "
                    "(prediction confidence: {:.1f}%). Immediate multidisciplinary "
                    "intervention is required — guidance counselor, adviser, and guardians "
                    "should collaborate on an urgent behavior recovery plan.".format(prob_top * 100)
                )
            elif prob_top >= 0.70:
                rec = (
                    f"The student exhibits a **very high probability** ({prob_top*100:.1f}%) "
                    f"of repeating '{top_v}'. Prompt counseling and corrective measures "
                    "are strongly advised to prevent recurrence."
                )
            elif prob_top >= 0.55:
                rec = (
                    f"There is a **high likelihood** ({prob_top*100:.1f}%) of repeating '{top_v}'. "
                    "Behavior should be closely monitored, and mentoring sessions initiated."
                )
            elif prob_top >= 0.35:
                rec = (
                    f"The predicted chance of repeating '{top_v}' is **moderate** "
                    f"({prob_top*100:.1f}%). Maintain active supervision and reinforce "
                    "positive conduct through consistent encouragement."
                )
            else:
                rec = (
                    f"The likelihood of repeating '{top_v}' is **low** ({prob_top*100:.1f}%), "
                    "indicating ongoing improvement. Continue to promote positive behavior."
                )


            results.append({
                "student_id": i + 1,
                "top_violations": ranked,
                "computed_avg_severity": avg,
                "computed_behavior_ratio": ratio,
                "note": "Predicted successfully",
                "recommendation": rec
            })

        return make_json_safe(results)
    
    # ---------------------------------------------------------------
    # ⚖️ Predict Disciplinary Action for a Student
    # ---------------------------------------------------------------
        # ---------------------------------------------------------------
    # ⚖️ Predict Disciplinary Action (Fair Action Justified by Reoffense Probability)
    # ---------------------------------------------------------------
    def predict_disciplinary_action(self, student):
        """
        Predicts a fair disciplinary action level for a student using the Decision Tree model,
        justified by the reoffense probability estimated from the Logistic Regression model.
        """
        if isinstance(student, dict):
            student_df = pd.DataFrame([student])
        else:
            student_df = student.copy()

        # Ensure all features exist
        for feature in self.features:
            if feature not in student_df.columns:
                student_df[feature] = 0

        # Auto-compute indicators
        for idx, row in student_df.iterrows():
            s = row.to_dict()
            student_df.at[idx, "avg_severity"] = self._auto_compute_avg_severity(s)
            student_df.at[idx, "behavior_ratio"] = self._auto_compute_behavior_ratio(s)

        X_single = student_df[self.features]

        # 🔹 Get reoffense risk probability
        risk_prob = self.predict_reoffense_risk(student)["probability_of_reoffense"]
        predicted_code = self.tree_model.predict(X_single)[0]

        disciplinary_actions = {
            "1": "Warning",
            "2": "Oral Warning",
            "3": "Written Warning",
            "4": "Conference With Parents/Guardians",
            "5": "Restitution",
            "6": "Confiscation",
            "7": "Demerit",
            "8": "Suspension",
            "9": "Exclusion",
            "10": "Dismissal/Non-readmission",
            "11": "Expulsion"
        }

        action_name = disciplinary_actions.get(predicted_code, "Unknown")

        # ---------------------------------------------------------------
        # 🧭 Fair Action Logic (low-risk override & undecided handling)
        # ---------------------------------------------------------------
        if risk_prob < 0.2 and predicted_code in ["8", "9", "10", "11"]:
            adjusted_action = "6"  # downgrade to demerit
            action_name = disciplinary_actions[adjusted_action]
            remark = (
                f"The predicted reoffense risk is very low ({risk_prob*100:.1f}%), "
                "but the model suggests a severe penalty. In fairness, a demerit is more appropriate. "
                "This approach maintains accountability while recognizing that the student’s conduct "
                "does not warrant heavy disciplinary action. Such balance reflects the principle of "
                "proportionate discipline—firm but fair."
            )

        elif 0.2 <= risk_prob < 0.4 and predicted_code in ["7", "8", "9", "10"]:
            adjusted_action = "5"  # downgrade to one-day suspension
            action_name = disciplinary_actions[adjusted_action]
            remark = (
                f"The reoffense probability is relatively low ({risk_prob*100:.1f}%), "
                "yet the model recommends a severe consequence. A lighter measure such as a one-day suspension "
                "with counseling maintains fairness—acknowledging the violation but providing space for growth."
            )

        elif 0.4 <= risk_prob <= 0.6 and predicted_code in ["8", "9", "10"]:
            adjusted_action = "Undecided"
            action_name = "Manual Review Required"
            remark = (
                f"The student's reoffense probability ({risk_prob*100:.1f}%) falls in a moderate range, "
                "while the model indicates a severe disciplinary level. Further review is advised to ensure "
                "fair and context-sensitive judgment. The case should be assessed by the guidance office "
                "and disciplinary committee before action is finalized."
            )

        elif risk_prob > 0.9 and predicted_code in ["1", "2", "3", "4"]:
            adjusted_action = "8"  # escalate to suspension
            action_name = disciplinary_actions[adjusted_action]
            remark = (
                f"With an exceptionally high reoffense risk ({risk_prob*100:.1f}%), a suspension is a fair and justified response. "
                "This escalation protects the school community while emphasizing rehabilitation and accountability. "
                "It ensures discipline remains both corrective and protective in nature."
            )

        else:
            adjusted_action = predicted_code
            remark = self._generate_action_remark(predicted_code, risk_prob)

        return make_json_safe({
            "predicted_action_level": adjusted_action,
            "recommended_action": action_name,
            "reoffense_probability": round(risk_prob, 3),
            "advisory_remark": remark,
            "note": "Prediction completed successfully."
        })
        
    # ---------------------------------------------------------------
    # 💬 Generate Action Remarks (Contextual Justifications)
    # ---------------------------------------------------------------
    def _generate_action_remark(self, code, prob):
        p = prob * 100
        if code == "1":
            return f"Low risk ({p:.1f}%). A fair, minimal response — simple warning for awareness."
        elif code == "2":
            return f"Moderate risk ({p:.1f}%). Restitution promotes responsibility and fairness."
        elif code == "4":
            return f"High risk ({p:.1f}%). Demerit maintains accountability with room for growth."
        elif code == "5":
            return f"Elevated risk ({p:.1f}%). Suspension is proportionate and reform-oriented."
        elif code == "6":
            return f"High risk ({p:.1f}%). Exclusion maintains fairness by ensuring campus safety."
        elif code == "7":
            return f"Very high risk ({p:.1f}%). Dismissal is justified for repeated defiance."
        elif code == "8":
            return f"Critical risk ({p:.1f}%). Expulsion fair only after exhaustive intervention."
        else:
            return f"Risk {p:.1f}% — further review advised for fair judgment."
    # ---------------------------------------------------------------
    # 🧩 Recommendation Summary
    # ---------------------------------------------------------------
    def _generate_recommendation(self, risk_level):
        if risk_level == "Critical":
            return (
                "⚠️The student demonstrates an exceptionally high probability of reoffending, "
                "indicating persistent or escalating behavioral problems that can no longer be addressed through routine interventions. "
                "As the prefect, you are strongly advised to take immediate and formal action by reporting the case directly to the "
                "Guidance Office and the Discipline Committee. It is essential to arrange an emergency case conference that includes "
                "the student, class adviser, guidance counselor, and, if possible, the student’s parents or guardians. "
                "During this meeting, a structured Behavior Recovery Plan should be drafted—outlining clear behavioral goals, "
                "consequences for further infractions, and scheduled progress assessments. You must document every detail of the student’s "
                "conduct, including previous infractions and intervention attempts, to establish a consistent behavioral record. "
                "Furthermore, it is crucial that you follow up weekly with teachers to evaluate progress and ensure that the intervention "
                "remains effective. Your leadership and coordination at this stage are critical in preventing further escalation and "
                "guiding the student toward behavioral restoration."
            )

        elif risk_level == "Very High":
            return (
                "The student’s behavioral pattern indicates a strong and recurring tendency to commit violations, "
                "often showing minimal response to previous corrective measures. As the prefect, you should act decisively yet thoughtfully. "
                "Begin by holding a confidential and structured one-on-one discussion with the student to identify emotional, academic, "
                "or environmental factors that may be contributing to the misconduct. Prepare a written summary of your observations and "
                "forward it to the Guidance Office for further evaluation. You are advised to recommend that the student participate in a "
                "series of focused counseling or mentoring sessions aimed at behavioral reflection and self-awareness. "
                "Teachers should be informed discreetly to maintain coordinated classroom observation, ensuring that behavioral changes "
                "are consistently monitored and recorded. You are also encouraged to maintain close communication with both the guidance counselor "
                "and class adviser, ensuring that all behavioral progress is tracked and reviewed regularly. If necessary, propose temporary "
                "restrictions or behavioral watchlist inclusion to emphasize accountability while promoting rehabilitation through mentorship and reflection."
            )

        elif risk_level == "High":
            return (
                "The student presents a high probability of reoffending, suggesting the presence of consistent behavioral "
                "tendencies that require structured attention and correction. As the prefect, your responsibility is to act promptly and with precision. "
                "You should first document the student’s current and past violations comprehensively, then submit a disciplinary report to the Guidance Office "
                "and request that the case be reviewed for counseling intervention. Coordinate with the class adviser to ensure that the student’s classroom "
                "behavior is regularly monitored, and that any observed triggers or patterns are immediately reported. It is advisable to meet briefly with the "
                "student to remind them of school policies and the consequences of repeated misconduct, emphasizing the importance of self-discipline and accountability. "
                "You may also propose a short-term behavioral improvement plan, including weekly reflection forms or conduct evaluations, to help the student "
                "actively engage in their own correction. Maintaining consistency and visibility in your supervision will demonstrate fairness and reinforce "
                "the seriousness of behavioral expectations within the school environment."
            )

        elif risk_level == "Medium":
            return (
                "The student exhibits moderate behavioral concerns that, if left unaddressed, could gradually escalate "
                "into more serious misconduct. As the prefect, you should adopt a preventive approach by maintaining open communication with the student "
                "to provide encouragement, guidance, and gentle reminders about school discipline policies. Document any recurring patterns of misbehavior, "
                "no matter how minor, and share your observations with the class adviser for early guidance intervention. "
                "You may recommend that the student participate in leadership or co-curricular programs that build responsibility and teamwork, "
                "as involvement in structured activities often helps redirect energy toward positive engagement. Additionally, you should check in periodically "
                "with the student and teachers to track progress and reinforce accountability. By combining consistent observation with positive reinforcement, "
                "you can help stabilize the student’s conduct and prevent further behavioral regression."
            )

        else:  # Low
            return (
                "The student currently demonstrates positive and stable behavior with minimal risk of reoffending. "
                "As the prefect, your role shifts from corrective enforcement to supportive mentorship. You are encouraged to maintain regular interaction "
                "with the student to acknowledge and reinforce their consistent good behavior. Consider nominating the student for participation in "
                "student leadership initiatives, peer mentoring programs, or academic support roles, as these opportunities will strengthen their "
                "sense of responsibility and serve as positive examples to others. It is still advisable to maintain light observation to ensure that "
                "the student remains on a steady behavioral path, but no disciplinary intervention is required at this time. Instead, prioritize recognition, "
                "encouragement, and reinforcement of good conduct to promote long-term behavioral maturity and moral integrity."
            )