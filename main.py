import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("loan.csv")

# Drop irrelevant columns
df.drop(['Loan_ID'], axis=1, inplace=True)

# Fill missing values
for column in df.columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

# Encode categorical columns
label = LabelEncoder()
for col in df.select_dtypes(include='object'):
    df[col] = label.fit_transform(df[col])

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

test = [[1, 0, 1, 0, 1, 5000, 0, 200, 360, 1, 0, 2]]  # Example applicant
print("Loan Status:", "Approved" if model.predict(test)[0]==1 else "Rejected")
