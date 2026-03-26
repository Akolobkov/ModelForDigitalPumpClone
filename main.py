from download import download
from EDA import perform_eda
from data_cleaning import prepare_data_for_model1
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
path = download()
path_to_df = os.path.join(path, "industrial_pump_failure_dataset.csv")
df = pd.read_csv(path_to_df)
print('EDA надо? Сосал?')
t = input()
if t == 'да':
    perform_eda(df, 'failure_event')
X, y = prepare_data_for_model1(df)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_scaled, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=42, stratify=y_resampled)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'balanced_subsample', None],
    'max_features': ['sqrt', 'log2', None]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_search = RandomizedSearchCV(
    rf, rf_params, n_iter=50, cv=cv,
    scoring='f1_macro',
    n_jobs=-1, random_state=42
)
print("Обучение Random Forest...")
rf_search.fit(X_train, y_train)
print("Лучшие параметры RF:", rf_search.best_params_)
print("Лучший score RF:", rf_search.best_score_)
rf_best = rf_search.best_estimator_
y_pred_rf = rf_best.predict(X_test)
print("\nRandom Forest Results:")
print(classification_report(y_test, y_pred_rf))
print(f"F1-macro: {f1_score(y_test, y_pred_rf, average='macro'):.4f}")