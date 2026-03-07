import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('job_dataset.csv')

le_degree = LabelEncoder()
le_spec = LabelEncoder()
le_job = LabelEncoder()

df['Degree_Encoded'] = le_degree.fit_transform(df['Degree'])
df['Spec_Encoded'] = le_spec.fit_transform(df['Specialization'])
df['JobRole_Encoded'] = le_job.fit_transform(df['JobRole'])

X = df[['Degree_Encoded', 'Spec_Encoded', 'CGPA']]
y = df['JobRole_Encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('encoders.pkl', 'wb') as f:
    pickle.dump({'degree': le_degree, 'spec': le_spec, 'job': le_job}, f)

print("Model and Encoders saved successfully!")