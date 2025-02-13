import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import joblib
import warnings

warnings.filterwarnings('ignore')

# Définir le chemin du fichier
file_path = 'data/flights.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Le fichier spécifié n'existe pas: {file_path}")

# Charger les données
df = pd.read_csv(file_path)
df = df.sample(n=50000, random_state=42)

# Vérification des valeurs manquantes
print("Valeurs manquantes avant traitement :")
print(df.isnull().sum())

# Préparation des données
def preprocess_data(df):
    df = df.drop(columns=['YEAR', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'CANCELLATION_REASON', 
                          'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 
                          'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'])
    
    imputer = SimpleImputer(strategy='mean')
    numeric_features = ['DEPARTURE_DELAY', 'TAXI_OUT', 'TAXI_IN', 'ARRIVAL_DELAY']
    df[numeric_features] = imputer.fit_transform(df[numeric_features])
    df = df.dropna()

    df = pd.get_dummies(df, columns=['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'])
    
    scaler = StandardScaler()
    df[['DEPARTURE_DELAY', 'TAXI_OUT', 'TAXI_IN']] = scaler.fit_transform(df[['DEPARTURE_DELAY', 'TAXI_OUT', 'TAXI_IN']])
    
    df['NEW_FEATURE'] = df['MONTH'] + df['DAY_OF_WEEK']
    
    return df

df = preprocess_data(df)

# Vérification des valeurs manquantes après traitement
print("Valeurs manquantes après traitement :")
print(df.isnull().sum())

# Visualisation des retards
plt.figure(figsize=(10, 6))
sns.histplot(df['ARRIVAL_DELAY'], bins=50, kde=True)
plt.title('Distribution des retards')
plt.xlabel('Retard (minutes)')
plt.ylabel('Fréquence')
plt.savefig('retards_distribution.png')
plt.show()

# Heatmap des corrélations
numeric_cols = df.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap of Correlations')
plt.savefig('correlations_heatmap.png')
plt.show()

# Division des données
X = df.drop('ARRIVAL_DELAY', axis=1)
y = df['ARRIVAL_DELAY'].apply(lambda x: 1 if x > 15 else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Hyperparamétrage
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

gb_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 10]
}

# Entraînement et évaluation
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
        'Random Forest': RandomizedSearchCV(RandomForestClassifier(class_weight='balanced'), rf_params, n_iter=5, cv=3, verbose=0, n_jobs=-1),
        'Gradient Boosting': RandomizedSearchCV(GradientBoostingClassifier(), gb_params, n_iter=5, cv=3, verbose=0, n_jobs=-1),
        'Decision Tree': DecisionTreeClassifier(class_weight='balanced')
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'f1': f1,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, ROC AUC: {roc_auc:.4f}")
        print(classification_report(y_test, y_pred))

        joblib.dump(model, f'{name.replace(" ", "_").lower()}_model.pkl')
        
        if name == 'Random Forest':
            feature_importances = model.best_estimator_.feature_importances_
            features = X_train.columns
            sorted_idx = np.argsort(feature_importances)[::-1]
            plt.figure(figsize=(10, 5))
            plt.bar(range(len(features)), feature_importances[sorted_idx], align='center')
            plt.xticks(range(len(features)), features[sorted_idx], rotation=90)
            plt.title("Feature Importances (Random Forest)")
            plt.savefig("feature_importances.png")
            plt.show()
    
    return results

def main():
    results = train_and_evaluate_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
