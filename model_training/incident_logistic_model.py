import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

class IncidentLogisticModel:
    def __init__(self, csv_file):
        self.df = pd.read_csv(f"./dataset/{csv_file}.csv")
        self.model = None
        self.train()

    def compute_avg_severity(self):
        if "ongoing_incident_count" in self.df.columns:
            self.df["avg_severity"] = (
                0.1 * self.df["minor_count"] +
                0.2 * self.df["repeated_incident_count"] +
                0.25 * self.df["ongoing_incident_count"] +
                0.3 * self.df["major_count"]
            )
        else:
            self.df["avg_severity"] = (
                0.1 * self.df["minor_count"] +
                0.2 * self.df["repeated_incident_count"] +
                0.3 * self.df["major_count"]
            )
        self.df["avg_severity"] = (self.df["avg_severity"] - self.df["avg_severity"].min()) / (
            self.df["avg_severity"].max() - self.df["avg_severity"].min()
        )

    def train(self):
        if "avg_severity" not in self.df.columns:
            self.compute_avg_severity()

        feature_cols = [col for col in ["repeated_incident_count", "major_count", "minor_count", "ongoing_incident_count", "avg_severity"]
                        if col in self.df.columns]
        X = self.df[feature_cols]
        y = self.df["will_have_another_incident"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        self.model = LogisticRegression(max_iter=2000, solver="liblinear")
        self.model.fit(X_train, y_train)

        self.x_test = X_test
        self.y_test = y_test


    def evaluate_model(self):
        if self.model is None:
            print("⚠️ Please train the model first using train_model().")
            return

        y_pred = self.model.predict(self.x_test)
        y_prob = self.model.predict_proba(self.x_test)[:, 1]

        print("\n📊 Classification Report:")
        print(classification_report(self.y_test, y_pred))
        print("Accuracy:", round(accuracy_score(self.y_test, y_prob), 3))


    def predict(self, repeated_incident_count, major_count, minor_count, ongoing_incident_count=0):
        remark = ''
        if self.model is None:
            print("⚠️ Please train the model first using train_model().")
            return

        avg_severity = (
            0.1 * float(minor_count) +
            0.2 * float(repeated_incident_count) +
            0.25 * float(ongoing_incident_count) +
            0.3 * float(major_count)
        )
        avg_severity = (avg_severity - self.df["avg_severity"].min()) / (
            self.df["avg_severity"].max() - self.df["avg_severity"].min()
        )

        new_student = pd.DataFrame([{
            "repeated_incident_count": repeated_incident_count,
            "major_count": major_count,
            "minor_count": minor_count,
            "ongoing_incident_count": ongoing_incident_count,
            "avg_severity": avg_severity
        }])

        pred_class = self.model.predict(new_student)[0]
        pred_prob = self.model.predict_proba(new_student)[:, 1][0]
        
        if pred_class == 1:
            if major_count > 3:
                remark = (
                    "High concern: The student has accumulated several major disciplinary offenses, indicating a recurring pattern "
                    "of serious misconduct. Due to the frequency and severity of these incidents, the student should be referred "
                    "to the Guidance Office for a comprehensive behavioral evaluation and intervention plan. "
                    "A structured counseling session focusing on accountability, values formation, and personal reflection is strongly advised."
                )
            elif repeated_incident_count > 2:
                remark = (
                    "Moderate concern: The student has shown repeated involvement in similar violations, which suggests a habit-forming "
                    "behavioral issue. The student should be scheduled for a guidance counseling session to address underlying factors "
                    "that may be contributing to this pattern. Continuous monitoring and positive reinforcement strategies are recommended."
                )
            elif pred_prob > 0.7:
                remark = (
                    "Elevated risk: Data indicates a high likelihood that the student may commit another offense soon. "
                    "Immediate referral to the Guidance Office is recommended for preventive intervention. "
                    "A behavioral follow-up session should be arranged to help the student reflect and redirect actions positively."
                )
            else:
                remark = (
                    "Noticeable concern: The student has shown minor but recurring behavioral issues. Although not yet severe, "
                    "it is recommended that the student attend a brief counseling session to discuss behavior management and "
                    "understanding the impact of their actions. Preventive guidance may help avoid future incidents."
                )
        else:
            if major_count == 0 and repeated_incident_count == 0:
                remark = (
                    "Good standing: The student maintains a clean behavioral record with no major or repeated offenses. "
                    "Encourage continued positive conduct through acknowledgment and reinforcement. "
                    "No immediate action is required, but ongoing awareness of behavioral expectations is advised."
                )
            elif avg_severity < 0.3:
                remark = (
                    "Low concern: The student has exhibited only minor offenses, suggesting good general behavior with occasional lapses. "
                    "A short counseling dialogue may be conducted to sustain positive habits and prevent future misconduct."
                )
            else:
                remark = (
                    "Stable but monitor behavior: The student generally complies with school rules, though recent actions suggest mild concern. "
                    "A light consultation with the Guidance Office is encouraged to reinforce awareness and prevent potential future violations."
                )




        #print(f"\n🎯 Predicted class (Will have another incident?): {pred_class}")
        #print(f"🔮 Predicted probability: {round(pred_prob, 3)}")

        return pred_class, pred_prob, remark