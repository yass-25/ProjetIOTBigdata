import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
import joblib
import warnings

warnings.filterwarnings('ignore')

# Charger les données
file_path = 'data/flights.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Le fichier spécifié n'existe pas: {file_path}")
df = pd.read_csv(file_path)
df = df.sample(n=50000, random_state=42)

# Comparaison des techniques de preprocessing
def preprocess_data(df, imputation_strategy='mean', scaling_strategy='standard'):
    df = df.drop(columns=['YEAR', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'CANCELLATION_REASON', 
                          'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 
                          'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'])
    
    numeric_features = ['DEPARTURE_DELAY', 'TAXI_OUT', 'TAXI_IN', 'ARRIVAL_DELAY']
    
    if imputation_strategy == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif imputation_strategy == 'median':
        imputer = SimpleImputer(strategy='median')
    elif imputation_strategy == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    
    df[numeric_features] = imputer.fit_transform(df[numeric_features])
    df = df.dropna()
    df = pd.get_dummies(df, columns=['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'])
    
    if scaling_strategy == 'standard':
        scaler = StandardScaler()
    elif scaling_strategy == 'minmax':
        scaler = MinMaxScaler()
    
    df[['DEPARTURE_DELAY', 'TAXI_OUT', 'TAXI_IN']] = scaler.fit_transform(df[['DEPARTURE_DELAY', 'TAXI_OUT', 'TAXI_IN']])
    
    df['NEW_FEATURE'] = df['MONTH'] * df['DAY_OF_WEEK']
    return df

# Comparaison avec et sans feature engineering
df_fe = preprocess_data(df, imputation_strategy='mean', scaling_strategy='standard')
df_no_fe = df_fe.drop(columns=['NEW_FEATURE'])

# Définition de la target
X_fe, y_fe = df_fe.drop('ARRIVAL_DELAY', axis=1), df_fe['ARRIVAL_DELAY'].apply(lambda x: 1 if x > 15 else 0)
X_no_fe, y_no_fe = df_no_fe.drop('ARRIVAL_DELAY', axis=1), y_fe

X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(X_fe, y_fe, test_size=0.3, random_state=42)
X_train_no_fe, X_test_no_fe, y_train_no_fe, y_test_no_fe = train_test_split(X_no_fe, y_no_fe, test_size=0.3, random_state=42)

# Comparaison des modèles et Feature Selection
def train_and_evaluate_model(X_train, X_test, y_train, y_test, feature_selection=False):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'XGBoost': XGBClassifier()
    }
    
    if feature_selection:
        selector = SelectKBest(score_func=f_classif, k=10)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        results[name] = {'accuracy': accuracy, 'f1': f1, 'roc_auc': roc_auc}
        print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
        print(classification_report(y_test, y_pred))
        joblib.dump(model, f'{name.replace(" ", "_").lower()}_model.pkl')
    
    return results

# Comparaison avec un modèle baseline
def baseline_model(y_test):
    y_pred_baseline = np.zeros_like(y_test)
    accuracy = accuracy_score(y_test, y_pred_baseline)
    f1 = f1_score(y_test, y_pred_baseline)
    roc_auc = roc_auc_score(y_test, y_pred_baseline)
    print(f"Baseline Model - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
    return {'accuracy': accuracy, 'f1': f1, 'roc_auc': roc_auc}

# Exécution des expériences
def main():
    print("\nModèles avec Feature Engineering:")
    results_fe = train_and_evaluate_model(X_train_fe, X_test_fe, y_train_fe, y_test_fe)
    print("\nModèles sans Feature Engineering:")
    results_no_fe = train_and_evaluate_model(X_train_no_fe, X_test_no_fe, y_train_no_fe, y_test_no_fe)
    print("\nModèles avec Sélection de Variables:")
    results_fs = train_and_evaluate_model(X_train_fe, X_test_fe, y_train_fe, y_test_fe, feature_selection=True)
    print("\nModèle Baseline:")
    baseline_results = baseline_model(y_test_fe)
    
    # Comparaison des performances
    comparison_df = pd.DataFrame({'With Feature Engineering': results_fe,
                                   'Without Feature Engineering': results_no_fe,
                                   'With Feature Selection': results_fs,
                                   'Baseline': baseline_results}).T
    print(comparison_df)
    comparison_df.to_csv('comparison_results.csv')

if __name__ == "__main__":
    main()
