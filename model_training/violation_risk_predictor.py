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

            
# -------------------------------------------------------------------
# 🧠 Logistic Regression–Only Violation Predictor
# -------------------------------------------------------------------
class ViolationRiskPredictor:
    def __init__(self, csv_file, dataset_type='training'):
        self.logistic_model = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.features = None
        self.student_data = None
        self.train_model(csv_file, dataset_type)

    # ---------------------------------------------------------------
    # 🧩 Data Preparation & Training
    # ---------------------------------------------------------------
    def load_and_prepare(self, data: pd.DataFrame):
        """Prepare dataset with simplified features focused on violation behavior."""
        df = data.copy()
        if "will_reoffend_another_violation" not in df.columns:
            raise ValueError("Missing column: 'will_reoffend_another_violation'")

        # --- Feature Selection (Code B style) ---
        required = ["total_violations", "total_repeated_violations", "total_no_violation_in_cases"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        X = df[required]
        y = df["will_reoffend_another_violation"]

        self.features = X.columns
        self.student_data = df
        return X, y

        
    def train_model(self, csv_file, dataset_type):
        """Train logistic regression model from CSV dataset."""
        path_prefix = f'{dataset_type}/' if dataset_type else ''
        data_path = f"./dataset/{path_prefix}{csv_file}.csv"

        X, y = self.load_and_prepare(pd.read_csv(data_path))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.logistic_model.fit(X_train, y_train)

        # Evaluate performance
        y_pred = self.logistic_model.predict(X_test)
        y_prob = self.logistic_model.predict_proba(X_test)[:, 1]

        print("=== Logistic Regression Performance ===")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("ROC-AUC:", roc_auc_score(y_test, y_prob))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # ---------------------------------------------------------------
    # ⚙️ Auto Compute Final Features for Single Input
    # ---------------------------------------------------------------
    def _auto_compute_features(self, student: dict) -> dict:
        """
        Prepare single-student features consistent with Code B logistic model.
        No proportional metrics, only direct violation-count features.
        """
        return {
            "total_violations": student.get("total_violations", 0),
            "total_repeated_violations": student.get("total_repeated_violations", 0),
            "total_no_violation_in_cases": student.get("total_no_violations", 0),
        }


    # ---------------------------------------------------------------
    # 🎯 Predict Reoffense Risk for a Student
    # ---------------------------------------------------------------
    def predict_reoffense_risk(self, student_input):
        """Predict reoffense probability for a single student (Code B logic)."""
        if not isinstance(student_input, dict):
            raise TypeError("student_input must be a dict.")

        # Prepare features for prediction
        computed = self._auto_compute_features(student_input)
        X_single = pd.DataFrame([computed])

        # Predict probability directly
        prob = float(self.logistic_model.predict_proba(X_single[self.features])[0][1])
        
        # Special case: if ALL counts are zero, do NOT use the logistic model
        if (
            student_input.get("total_violations", 0) == 0 and
            student_input.get("total_repeated_violations", 0) == 0 and
            student_input.get("total_no_violations", 0) == 0
        ):
            return make_json_safe({
                "student_id": str(student_input.get("student_id", "Unknown")),
                "total_incidents": student_input.get("total_incidents", 0),
                "probability_of_reoffense": 0.0,
                "probability_raw": 0.0,
                "risk_level": "Low",
                "total_violations": 0,
                "total_repeated_violations": 0,
                "total_no_violations": 0,
                "note": "Auto-classified: No recorded behavior data"
            })

        # Risk levels (Code B threshold logic)
        if prob < 0.30:
            risk = "Low"
        elif prob < 0.60:
            risk = "Moderate"
        elif prob < 0.85:
            risk = "High"
        else:
            risk = "Critical"
            
        return make_json_safe({
            "student_id": str(student_input.get("student_id", "Unknown")),
            "total_incidents": student_input.get("total_incidents", 0),
            "probability_of_reoffense": round(prob, 4),
            "probability_raw": round(prob, 4),
            "risk_level": risk,
            "total_violations": student_input.get("total_violations", 0),
            "total_repeated_violations": student_input.get("total_repeated_violations", 0),
            "total_no_violations": computed['total_no_violation_in_cases'],
            "note": "Predicted successfully"
        })

        
    # ---------------------------------------------------------------
    # 📊 Get Student Risk Monitoring (Batch Watchlist)
    # ---------------------------------------------------------------
    def get_student_risk_monitoring(self, input_data):
        """
        Generate a risk monitoring table for multiple students.
        Accepts: list of dicts or a DataFrame.
        Returns: list of ranked student risks.
        """
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list) and all(isinstance(i, dict) for i in input_data):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError("input_data must be a dict, list of dicts, or DataFrame.")

        results = []
        for i, row in df.iterrows():
            student = row.to_dict()
            prediction = self.predict_reoffense_risk(student)
            results.append(prediction)

        # Sort by probability (descending)
        sorted_results = sorted(results, key=lambda x: x["probability_of_reoffense"], reverse=True)

        return make_json_safe(sorted_results)
    
    # ---------------------------------------------------------------
    # 🧩 Generate Insight (Supportive Behavioral Observation Summary)
    # ---------------------------------------------------------------
    def generate_insight(self, student_result):
        """
        Returns:
        - insight_summary (prefect)
        - insight_summary_student (student)
        """

        # ---------------------------------------------------------------
        # 🧩 Insight for Prefect (Very Long, Formal, Count-Based)
        # ---------------------------------------------------------------
        def generate_insight_prefect(student_result):

            total_incidents = student_result.get("total_incidents", 0)
            total_violations = student_result.get("total_violations", 0)
            total_repeated = student_result.get("total_repeated_violations", 0)
            total_no_violation = student_result.get("total_no_violations", 0)
            risk_level = (student_result.get("risk_level") or "Low").capitalize()

            repeat_rate = total_repeated / (total_violations + 1)
            positive_rate = total_no_violation / (total_incidents + 1)

            # --- Risk framing with expanded formal narrative ---
            risk_map = {
                "Low": (
                    f"The student is assessed to be at a **LOW RISK** of committing future violations. "
                    f"Across a total of **{total_incidents} recorded incidents**, only **{total_violations} led to actual violations**, "
                    f"which reflects a generally stable behavioral pattern with minimal indicators of potential escalation. "
                    f"This level of risk typically suggests that the student is able to regulate their behavior effectively "
                    f"when provided with proper expectations and consistency. While occasional concerns may arise, the overall trend "
                    f"indicates that the student responds positively to correction and maintains appropriate conduct in most situations. "
                ),
                "Moderate": (
                    f"The student is assessed at a **MODERATE RISK** of future violations. Among the **{total_incidents} incidents**, "
                    f"a total of **{total_violations} resulted in documented violations**, suggesting noticeable behavior patterns beginning "
                    f"to emerge. These trends imply that although the student is capable of appropriate conduct, there are recurring "
                    f"behavioral indicators that should be addressed early to prevent escalation. This classification highlights the need "
                    f"for consistent monitoring and supportive guidance to help the student stabilize their behavioral responses. "
                ),
                "High": (
                    f"The student is classified as **HIGH RISK** for repeated violations. Out of **{total_incidents} incidents**, "
                    f"the student recorded **{total_violations} violations**, signaling a growing pattern of behavioral inconsistency. "
                    f"This suggests that current corrective strategies may not be fully sufficient, and there may be persistent factors "
                    f"contributing to misconduct. At this level, increased structure, heightened supervision, and strategic intervention "
                    f"are recommended to mitigate continued behavioral recurrence. "
                ),
                "Critical": (
                    f"The student is identified under a **CRITICAL RISK** category. With **{total_violations} violations** out of "
                    f"**{total_incidents} total incidents**, the overall pattern strongly reflects ongoing or escalating misconduct. "
                    f"This risk level typically indicates significant difficulty in maintaining behavioral expectations, possibly signaling "
                    f"the need for immediate and coordinated action involving the Prefect, Guidance Office, and guardians. "
                    f"Proactive intervention is crucial to prevent further behavioral deterioration. "
                )
            }

            risk_statement = risk_map.get(risk_level)

            # --- General violation frequency ---
            if total_violations == 0:
                violation_behavior = (
                    "There are **no recorded violations**, reflecting strong consistency and adherence to expectations. "
                )
            elif total_violations <= 2:
                violation_behavior = (
                    f"The student committed **{total_violations} violation(s)**, indicating isolated concerns that do not currently suggest "
                    f"a developing pattern. "
                )
            elif total_violations <= 5:
                violation_behavior = (
                    f"The student committed **{total_violations} violations**, suggesting a noticeable trend that warrants monitoring and "
                    f"possible early guidance. "
                )
            else:
                violation_behavior = (
                    f"The student committed **{total_violations} violations**, demonstrating a recurring pattern that requires structured "
                    f"behavioral intervention. "
                )

            # --- Repeated violations ---
            if total_repeated == 0:
                repeated_behavior = (
                    "There are **no repeated violations**, indicating effective responsiveness to previous corrective actions. "
                )
            elif repeat_rate >= 0.40:
                repeated_behavior = (
                    f"A substantial portion of the violations (**{total_repeated} repeated cases**) suggests reinforced behavior patterns "
                    f"that may escalate without targeted behavioral guidance. "
                )
            else:
                repeated_behavior = (
                    f"There are **{total_repeated} repeated violations**, showing emerging behavioral tendencies that require monitoring. "
                )

            # --- No-violation cases ---
            if total_no_violation == 0:
                nv_behavior = (
                    "There are **no incidents** where the student avoided a violation, which may indicate that intervention strategies "
                    "should focus on improving self-regulation and decision-making during monitored situations. "
                )
            elif positive_rate >= 0.50:
                nv_behavior = (
                    f"There are **{total_no_violation} cases** where no violation occurred, demonstrating positive potential and "
                    f"appropriate behavior in several monitored situations. "
                )
            else:
                nv_behavior = (
                    f"The student recorded **{total_no_violation} no-violation case(s)**, indicating the presence of good judgment in selected "
                    f"situations, which can be reinforced through guided support. "
                )

            summary = (
                f"{risk_statement}{violation_behavior}{repeated_behavior}{nv_behavior}"
                "Overall, these observations highlight the importance of consistent supervision, intentional guidance, and structured "
                "behavioral follow-through to ensure that the student continues moving toward positive and stable conduct."
            )

            return summary

        # ---------------------------------------------------------------
        # 🧩 Insight for Student (Friendly, Encouraging, Count-Based)
        # ---------------------------------------------------------------
        def generate_insight_student(student_result):

            total_incidents = student_result.get("total_incidents", 0)
            total_violations = student_result.get("total_violations", 0)
            total_repeated = student_result.get("total_repeated_violations", 0)
            total_no_violation = student_result.get("total_no_violations", 0)
            risk_level = (student_result.get("risk_level") or "Low").capitalize()

            # --- Friendly risk message ---
            if risk_level == "Low":
                risk_text = (
                    f"You were involved in **{total_incidents} different situations**, and only **{total_violations}** turned into violations, "
                    "which shows that you usually make good decisions. You're doing well, so keep trying to stay consistent with your actions "
                    "and continue choosing what’s right even when situations become challenging."
                )
            elif risk_level == "Moderate":
                risk_text = (
                    f"You had **{total_incidents} incidents** in total, with **{total_violations}** leading to violations. "
                    "This means you’re trying, but there are still moments where you could make better choices. "
                    "Paying a little more attention and thinking ahead can help you stay away from trouble and build better habits."
                )
            elif risk_level == "High":
                risk_text = (
                    f"You were involved in **{total_incidents} incidents**, where **{total_violations}** resulted in violations. "
                    "This shows that you may be struggling with some decisions, but that doesn’t mean you can’t improve. "
                    "With guidance and more awareness, you can learn to handle situations better and avoid repeating mistakes."
                )
            else:
                risk_text = (
                    f"You had **{total_violations} violations** out of **{total_incidents} incidents**, "
                    "which means you need to work harder on making safer and better choices. But don’t worry—we’re here to support you, "
                    "and you can always make changes that help you improve and succeed."
                )

            # --- Repeated violations ---
            if total_repeated == 0:
                repeat_text = (
                    "You had **no repeated violations**, which is a great sign that you’re learning from your experiences and improving."
                )
            else:
                repeat_text = (
                    f"You repeated **{total_repeated} violation(s)**. This means some actions happened more than once, "
                    "so this is your chance to reflect and avoid doing them again. You can still make better choices next time."
                )

            # --- No-violation cases ---
            if total_no_violation == 0:
                nv_text = (
                    "None of your incidents resulted in a no-violation outcome, so try to be more careful and think before acting next time."
                )
            else:
                nv_text = (
                    f"You had **{total_no_violation} incident(s)** where you did **not** receive a violation. "
                    "This shows that you DO know how to make good choices when you try—keep building on these positive moments!"
                )

            summary = (
                f"{risk_text}\n{repeat_text}\n{nv_text}\n"
                "You have so much potential to improve, and every good choice you make helps you grow. Keep going—we believe in you."
            )

            return summary

        # ---------------------------------------------------------------
        # FINAL RETURN
        # ---------------------------------------------------------------
        risk_level = (student_result.get("risk_level") or "Low").capitalize()

        return make_json_safe({
            "risk_level": risk_level,
            "insight_summary": generate_insight_prefect(student_result),
            "insight_summary_student": generate_insight_student(student_result)
        })








    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    # 🎯 Generate Recommendation (Neutral Prefect Advisory Notes)
    # ---------------------------------------------------------------
    def generate_recommendation(self, student_result, risk_level):
        """
        Returns:
        - recommendation (prefect)
        - recommendation_student (student)
        Includes all counts:
            total_incidents
            total_violations
            total_repeated_violations
            total_no_violations
        """

        level = (risk_level or "Low").capitalize()

        # ---------------------------------------------------------------
        # 🎯 Recommendation for Prefect (Formal + Detailed)
        # ---------------------------------------------------------------
        def generate_recommendation_prefect(student_result):

            total_incidents = student_result.get("total_incidents", 0)
            total_violations = student_result.get("total_violations", 0)
            tr = student_result.get("total_repeated_violations", 0)
            tnv = student_result.get("total_no_violations", 0)

            base_map = {
                "Critical": (
                    f"Based on the student’s overall behavioral record, which includes **{total_incidents} total incidents** "
                    f"and **{total_violations} violations**, the student currently falls under a **CRITICAL RISK** category. "
                    "This level of risk strongly indicates persistent or escalating behavioral concerns that require "
                    "immediate, structured, and collaborative intervention. It is recommended to initiate a coordinated "
                    "Behavior Support or Intervention Plan involving the Prefect, the Guidance Office, and the student's "
                    "parents or guardians. Clear expectations, close monitoring, and consistent follow-through should be maintained "
                    "to help prevent further escalation. "
                ),
                "High": (
                    f"With **{total_incidents} documented incidents** and **{total_violations} violations**, the student "
                    "falls under a **HIGH RISK** classification. This indicates recurring behavior concerns that may intensify "
                    "without structured support. It is recommended to establish regular behavioral check-ins, guided reflection sessions, "
                    "and strengthened classroom and campus monitoring. Collaboration with the Guidance Office is encouraged to provide "
                    "additional emotional and behavioral support to the student. "
                ),
                "Moderate": (
                    f"The student recorded **{total_incidents} incidents** and **{total_violations} violation(s)**, "
                    "placing them in the **MODERATE RISK** category. Early intervention at this stage is important to prevent escalation. "
                    "Recommended actions include consistent follow-ups after incidents, constructive feedback conversations, and positive reinforcement "
                    "when improvements are observed. "
                ),
                "Low": (
                    f"With **{total_incidents} incidents** and only **{total_violations} violation(s)**, the student is "
                    "classified as **LOW RISK**. Continued light supervision, recognition of positive behavior, and reminders of expectations "
                    "are recommended to maintain their current level of conduct. "
                )
            }

            base = base_map.get(level, base_map["Low"])

            if tr > 2:
                extra = (
                    f"The presence of **{tr} repeated violations** suggests recurring behavioral patterns that must be addressed directly. "
                    "Restorative dialogues, mentoring sessions, or structured behavior coaching are recommended to help the student understand "
                    "the impact of their choices and develop alternative responses. "
                )
            elif tr > 0:
                extra = (
                    f"The student has **{tr} repeated violation(s)**, which indicates that certain behaviors are resurfacing. "
                    "Periodic reflection activities and follow-up conversations focused on these specific behaviors may help reduce recurrence. "
                )
            else:
                extra = (
                    "There are **no repeated violations**, which suggests that corrective actions have been effective so far. "
                    "It is advisable to continue reinforcing positive change through acknowledgment and encouragement. "
                )

            if tnv > 0:
                positive = (
                    f"In addition, there are **{tnv} incident(s)** where no violation was recorded, showing that the student is capable of appropriate behavior "
                    "in certain situations. These positive examples can be highlighted during guidance or mentoring sessions to reinforce strengths "
                    "and encourage similar choices in the future. "
                )
            else:
                positive = (
                    "There are no recorded no-violation incidents. Providing the student with skills in conflict management, emotional regulation, "
                    "and decision-making may help support better behavioral outcomes moving forward. "
                )

            return f"{base}{extra}{positive}"

        # ---------------------------------------------------------------
        # 🎯 Recommendation for Student (Friendly + Action-Oriented)
        # ---------------------------------------------------------------
        def generate_recommendation_student(student_result):

            total_incidents = student_result.get("total_incidents", 0)
            total_violations = student_result.get("total_violations", 0)
            tr = student_result.get("total_repeated_violations", 0)
            tnv = student_result.get("total_no_violations", 0)

            if level == "Low":
                base = (
                    f"You were involved in **{total_incidents} situations**, and only **{total_violations}** became violations. "
                    "You’re doing well so far. To keep it that way, try to continue thinking before you act, especially in situations that feel stressful or confusing. "
                )
            elif level == "Moderate":
                base = (
                    f"You had **{total_incidents} incidents**, and **{total_violations}** turned into violations. "
                    "This shows there are times when you make good choices and times when things get more difficult. "
                    "From now on, it would help to slow down, listen carefully to reminders, and ask for help if you feel a situation is getting out of control. "
                )
            elif level == "High":
                base = (
                    f"You were involved in **{total_incidents} incidents**, and **{total_violations}** led to violations. "
                    "This means your actions are causing more problems than they should. A good next step is to fully join guidance or mentoring sessions, "
                    "honestly share what you are going through, and try to practice new ways of reacting when you feel upset or pressured. "
                )
            else:
                base = (
                    f"You had **{total_violations} violations** out of **{total_incidents} incidents**, which shows that you really need to work on your decisions right now. "
                    "Even so, this is not the end—you can still change your direction. Start by being open to support and focusing on one small improvement at a time. "
                )

            if tr == 0:
                extra = (
                    "You have **no repeated violations**, which is a very good sign. Keep remembering what you learned from each situation so you don’t fall into the same problem again. "
                )
            else:
                extra = (
                    f"You repeated **{tr} violation(s)**. A helpful step would be to think about what usually happens before those violations occur—"
                    "who you are with, what you feel, or what you are thinking—and then plan a different reaction you can use next time. "
                )

            if tnv == 0:
                nv_text = (
                    "There were no situations where you avoided a violation, so this is your chance to start building those positive moments. "
                    "When you feel tempted to break a rule, try to pause, take a breath, and choose a response that will not get you into trouble. "
                )
            else:
                nv_text = (
                    f"You had **{tnv} moment(s)** where you did **not** get a violation. Those are proof that you CAN make good choices. "
                    "Think about what you did in those moments—how you acted, what you said, or how you stayed calm—and try to do more of that in the future. "
                )

            return f"{base}{extra}{nv_text}"

        # ---------------------------------------------------------------
        # FINAL RETURN
        # ---------------------------------------------------------------
        return make_json_safe({
            "risk_level": level,
            "recommendation": generate_recommendation_prefect(student_result),
            "recommendation_student": generate_recommendation_student(student_result),
        })







        
    def append_test_data(self, list):
        pass
        




"""
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
        
        print("🔹 Logistic Regression — Reoffense Risk")
        print("Logistic Regression Accuracy: ", accuracy_score(y_risk_test, self.logistic_model.predict(X_test)))
        print()
        print(classification_report(y_risk_test, self.logistic_model.predict(X_test)))
        print()
        print("🔹 Decision Tree — Fair Disciplinary Action Prediction")
        print("Decision Tree Accuracy: ", accuracy_score(y_violation_test, self.tree_model.predict(X_test)))
        print()
        print(classification_report(y_violation_test, self.tree_model.predict(X_test)))
        
        

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
            
            """