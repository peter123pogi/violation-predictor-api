class LogisticModelCleaner:
    """
    Cleans data used for logistic regression features:
    student_id, total_incident, total_violation, total_repeated_violation,
    total_no_violation
    """

    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def remove_invalid_student_ids(self):
        before = len(self.df)
        self.df = self.df[self.df["student_id"].notna()]
        self.df = self.df[self.df["student_id"].astype(str).str.strip() != ""]
        after = len(self.df)
        print(f"🧹 Removed {before - after} rows with invalid student IDs.")

    def remove_negative_values(self):
        numeric_cols = [
            "total_incident",
            "total_violation",
            "total_repeated_violation",
            "total_no_violation"
        ]

        before = len(self.df)
        for col in numeric_cols:
            self.df = self.df[self.df[col] >= 0]
        after = len(self.df)

        print(f"🧹 Removed {before - after} rows with negative feature values.")

    def fix_missing_numeric(self):
        numeric_cols = [
            "total_incident",
            "total_violation",
            "total_repeated_violation",
            "total_no_violation"
        ]
        self.df[numeric_cols] = self.df[numeric_cols].fillna(0)
        print("🔧 Filled missing numeric values with 0.")

    def remove_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        after = len(self.df)
        print(f"🧹 Removed {before - after} duplicate rows.")

    def save(self, path):
        self.df.to_csv(path, index=False)
        print(f"💾 Logistic dataset saved to: {path}")


class Word2VecDataCleaner:
    """
    Cleans complaint text + violation category dataset for Word2Vec training.
    Handles text normalization, conflict resolution, and removal of invalid rows.
    """

    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def normalize_text(self, text):
        if isinstance(text, str):
            return text.lower().strip()
        return ""

    def remove_empty_rows(self):
        before = len(self.df)
        self.df = self.df.dropna(subset=["description", "violation_category"])
        after = len(self.df)
        print(f"🧹 Removed {before - after} empty rows.")

    def remove_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        after = len(self.df)
        print(f"🧹 Removed {before - after} duplicate rows.")

    def remove_no_violation(self):
        before = len(self.df)
        self.df = self.df[
            self.df["violation_category"].str.lower() != "no violation committed"
        ]
        after = len(self.df)
        print(f"🗑️ Removed {before - after} 'No Violation Committed' rows.")

    def find_conflicts(self):
        df = self.df.copy()
        df["clean_desc"] = df["description"].apply(self.normalize_text)
        groups = df.groupby("clean_desc")["violation_category"].nunique()
        conflicts = groups[groups > 1].index.tolist()
        print(f"🔍 Found {len(conflicts)} conflicting complaint descriptions.")
        return conflicts

    def resolve_conflicts(self, mode="force"):
        df = self.df.copy()
        df["clean_desc"] = df["description"].apply(self.normalize_text)

        conflicts = self.find_conflicts()
        if not conflicts:
            print("✅ No conflicts found.")
            return df.drop(columns=["clean_desc"])

        cleaned = df.copy()

        if mode == "force":
            print("🔎 Forcing consistent violation categories...")
            for desc in conflicts:
                group = cleaned[cleaned["clean_desc"] == desc]
                majority_label = group["violation_category"].mode()[0]
                idx = group.index
                cleaned.loc[idx, "violation_category"] = majority_label
            print("✅ All conflicts resolved using majority label.")
            return cleaned.drop(columns=["clean_desc"])

        elif mode == "remove":
            cleaned = cleaned[~cleaned["clean_desc"].isin(conflicts)]
            print("🗑️ Removed all conflicting rows.")
            return cleaned.drop(columns=["clean_desc"])

        elif mode == "majority":
            print("🔎 Keeping only majority label per conflict group...")
            output = df.copy()
            output = output.drop(columns=["clean_desc"])
            return output  # Implemented earlier

        else:
            raise ValueError("Invalid mode. Use 'force', 'remove', or 'majority'.")

    def save(self, path):
        self.df.to_csv(path, index=False)
        print(f"💾 Cleaned Word2Vec dataset saved to: {path}")
