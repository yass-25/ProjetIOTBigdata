import cudf
import cuml
import time

# Charger les données avec cuDF (DataFrame accéléré par GPU)
df_rapids = cudf.read_csv(file_path)

# Convertir les colonnes numériques
numeric_features = ['DEPARTURE_DELAY', 'TAXI_OUT', 'TAXI_IN', 'ARRIVAL_DELAY']
df_rapids[numeric_features] = df_rapids[numeric_features].astype('float32')

# Prétraitement avec cuDF (similaire à pandas, mais accéléré par GPU)
df_rapids = df_rapids.drop(columns=['YEAR', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'CANCELLATION_REASON', 
                                    'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 
                                    'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'])

# Remplir les valeurs manquantes
df_rapids = df_rapids.fillna(0)

# Assembler les features et normaliser avec RAPIDS (cuml pour les modèles ML)
from cuml.linear_model import LogisticRegression as cuLogReg
from cuml.metrics import roc_auc_score

# Diviser les données (par exemple, 70/30)
train_data_rapids = df_rapids.sample(frac=0.7, random_state=42)
test_data_rapids = df_rapids.drop(train_data_rapids.index)

# Entraîner un modèle Logistic Regression avec RAPIDS
X_train = train_data_rapids[numeric_features].values
y_train = train_data_rapids['ARRIVAL_DELAY'] > 15  # Binary classification
X_test = test_data_rapids[numeric_features].values
y_test = test_data_rapids['ARRIVAL_DELAY'] > 15

# Mesurer le temps d'entraînement RAPIDS
start_time = time.time()
log_reg_rapids = cuLogReg()
log_reg_rapids.fit(X_train, y_train)
end_time = time.time()
print(f"Training time with RAPIDS: {end_time - start_time} seconds")

# Prédictions et évaluation
y_pred_rapids = log_reg_rapids.predict(X_test)
roc_auc = roc_auc_score(y_test, y_pred_rapids)
print(f"RAPIDS ROC AUC: {roc_auc:.4f}")
