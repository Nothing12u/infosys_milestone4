import pandas as pd
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("🚀 Training Career Prediction Model (3 Features ONLY)...")
print("=" * 60)

# Load dataset
df = pd.read_csv('job_dataset.csv')
df = df.copy()  # Avoid SettingWithCopyWarning
print(f"📊 Loaded {len(df)} records")

# ✅ SELECT ONLY 3 FEATURES - EXCLUDE 'Technologies'
feature_columns = ['Degree', 'Specialization', 'CGPA']
X = df[feature_columns].copy()
y = df['JobRole'].copy()

print(f"🔍 Features used: {feature_columns}")
print(f"❌ Excluded: Technologies (not used in prediction)")

# Encode categorical features
encoders = {}

# Encode Degree
le_degree = LabelEncoder()
X['Degree'] = le_degree.fit_transform(X['Degree'])
encoders['degree'] = le_degree
print(f"🔑 Encoded 'Degree': {len(le_degree.classes_)} classes")

# Encode Specialization  
le_spec = LabelEncoder()
X['Specialization'] = le_spec.fit_transform(X['Specialization'])
encoders['spec'] = le_spec
print(f"🔑 Encoded 'Specialization': {len(le_spec.classes_)} classes")

# Encode JobRole (target)
le_job = LabelEncoder()
y_encoded = le_job.fit_transform(y)
encoders['job'] = le_job
print(f"🔑 Encoded 'JobRole': {len(le_job.classes_)} classes")

# Ensure CGPA is numeric
X['CGPA'] = pd.to_numeric(X['CGPA'], errors='coerce').fillna(X['CGPA'].median())

# Split data - handle stratify error for rare classes
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print("✅ Stratified split successful")
except ValueError:
    print("⚠️ Stratify failed (rare classes). Using random split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

# Train model with robust parameters
model = DecisionTreeClassifier(
    max_depth=12,
    min_samples_split=8,
    min_samples_leaf=4,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# Evaluate
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Training Accuracy: {train_acc*100:.2f}%")
print(f"✅ Testing Accuracy: {test_acc*100:.2f}%")

# ✅ SAVE MODEL AS DICTIONARY WITH METADATA
model_package = {
    'model': model,
    'encoders': encoders,
    'feature_names': feature_columns,
    'n_features': len(feature_columns),
    'accuracy': test_acc
}

with open('model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

# Also save encoders separately for compatibility
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("=" * 60)
print(f"💾 Model saved successfully!")
print(f"📐 Model expects EXACTLY {model.n_features_in_} features")
print(f"🔑 Expected order: {feature_columns}")
print(f"📊 Test Accuracy: {test_acc*100:.2f}%")
print("✨ Run 'streamlit run app.py' to test!")
print("=" * 60)
