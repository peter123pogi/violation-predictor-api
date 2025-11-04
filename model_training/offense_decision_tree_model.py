import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, accuracy_score

    
class OffenseDecisionTree:
    def __init__(self, csv_file):
        self.df = pd.read_csv(f"./dataset/{csv_file}.csv")

        if "avg_severity" not in self.df.columns:
            self.df["avg_severity"] = (
                0.1 * self.df["minor_count"] +
                0.2 * self.df["repeated_incident_count"] +
                0.3 * self.df["major_count"]
            )
            self.df["avg_severity"] = (
                self.df["avg_severity"] - self.df["avg_severity"].min()
            ) / (self.df["avg_severity"].max() - self.df["avg_severity"].min())

        self.X = self.df.drop(columns=["will_reoffend"])
        self.y = self.df["will_reoffend"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        self.model = None        
        self.train()

    def train(self, max_depth=4, criterion="entropy"):
        self.model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        print("\n📈 Classification Report:")
        print(classification_report(self.y_test, y_pred))
        print("Accuracy:", round(accuracy_score(self.y_test, y_prob), 3))

    def predict(self, repeated_incident_count, major_count, minor_count):
        avg_severity = 0.1 * minor_count + 0.2 * repeated_incident_count + 0.3 * major_count
        avg_severity = (avg_severity - self.df["avg_severity"].min()) / (
            self.df["avg_severity"].max() - self.df["avg_severity"].min()
        )

        student = pd.DataFrame([{
            "repeated_incident_count": repeated_incident_count,
            "major_count": major_count,
            "minor_count": minor_count,
            "avg_severity": avg_severity
        }])

        pred_class = self.model.predict(student)[0]
        pred_prob = self.model.predict_proba(student)[:, 1][0]

        if pred_class == 1:
            if major_count > 3:
                remark = (
                    "High risk: The student has accumulated several major disciplinary violations, "
                    "indicating a consistent disregard for school policies. This behavior pattern strongly suggests "
                    "a likelihood of committing another serious offense if no corrective intervention is applied."
                )
            elif repeated_incident_count > 2:
                remark = (
                    "Moderate risk: The student has shown repeated involvement in similar types of incidents. "
                    "Such recurring patterns of misconduct indicate behavioral habits that may lead to future reoffenses "
                    "without close supervision or counseling."
                )
            elif pred_prob > 0.7:
                remark = (
                    "Elevated probability of reoffense: Statistical analysis of this student's past behavior "
                    "reveals a high likelihood of another incident occurring. Continuous monitoring and behavioral "
                    "guidance are strongly recommended."
                )
            else:
                remark = (
                    "Mild risk: The student displays occasional behavioral issues, but these are not yet severe. "
                    "It is advisable to maintain awareness and offer preventive guidance to ensure these do not escalate."
                )
        else:
            if major_count == 0 and repeated_incident_count == 0:
                remark = (
                    "Very low risk: The student maintains a clean disciplinary record with no major or repeated offenses. "
                    "Their behavior indicates strong compliance with institutional rules and a low probability of future incidents."
                )
            elif avg_severity < 0.3:
                remark = (
                    "Low risk: The student's offenses are minor and isolated in nature, suggesting temporary lapses "
                    "rather than a persistent behavioral concern. Regular encouragement and positive reinforcement "
                    "should help sustain improvement."
                )
            else:
                remark = (
                    "Stable but monitor periodically: Although the student’s recent conduct is generally acceptable, "
                    "some indicators suggest mild behavioral fluctuation. Continued observation and mentoring could help "
                    "ensure long-term stability."
                )


        #print("\n🎯 Prediction Result:")
        #print(f"Predicted class (Will reoffend?): {pred_class}")
        #print(f"Predicted probability: {round(pred_prob, 3)}")
        #print(f"📝 Remark: {remark}")

        return pred_class, pred_prob, remark