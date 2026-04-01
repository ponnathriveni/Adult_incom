import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("adult.csv", na_values="?")

# clean column names
df.columns = df.columns.str.strip().str.replace(".", "_")

# fill missing categorical with mode
for col in ["workclass", "occupation", "native_country"]:
    df[col].fillna(df[col].mode()[0], inplace=True)

# target mapping
df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

# -------------------------
# Split
# -------------------------
X = df.drop("income", axis=1)
y = df["income"]

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=122
)

# -------------------------
# Preprocess + Model Pipeline
# -------------------------
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

pipe = Pipeline([
    ("prep", preprocess),
    ("model", DecisionTreeClassifier(max_depth=6, random_state=42))
])

pipe.fit(X_train, y_train)

# -------------------------
# Evaluate
# -------------------------
pred = pipe.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# -------------------------
# Save pipeline
# -------------------------
joblib.dump(pipe, "adult_income_pipeline.pkl")

print("Saved: adult_income_pipeline.pkl")
