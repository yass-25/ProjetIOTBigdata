import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer, OneHotEncoder, StringIndexer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Initialiser une session Spark
spark = SparkSession.builder \
    .appName("FlightDelayPrediction") \
    .getOrCreate()

# Charger les données
file_path = 'data/flights.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Le fichier spécifié n'existe pas: {file_path}")

df = spark.read.option("header", "true").csv(file_path)

# Afficher les premières lignes du DataFrame Spark
df.printSchema()
df.show(5)

# Convertir les colonnes numériques
numeric_features = ['DEPARTURE_DELAY', 'TAXI_OUT', 'TAXI_IN', 'ARRIVAL_DELAY']
for feature in numeric_features:
    df = df.withColumn(feature, col(feature).cast("double"))

# Prétraitement des données
# Suppression de colonnes non pertinentes
df = df.drop('YEAR', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'CANCELLATION_REASON', 
             'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 
             'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY')

# Remplir les valeurs manquantes pour les colonnes numériques et catégorielles
imputer = Imputer(inputCols=numeric_features, outputCols=numeric_features)
df = imputer.fit(df).transform(df)

# Convertir les données catégorielles
categorical_features = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
indexers = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in categorical_features]
encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol=indexer.getOutputCol() + "_vec") for indexer in indexers]

# Ajouter des colonnes dérivées
df = df.withColumn('NEW_FEATURE', col('MONTH') * col('DAY_OF_WEEK'))

# Assembler les features
assembler = VectorAssembler(
    inputCols=[col + "_index_vec" for col in categorical_features] + numeric_features + ['NEW_FEATURE'],
    outputCol="features")

# Normaliser les données
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Définir la colonne étiquette (label)
df = df.withColumn("label", when(col("ARRIVAL_DELAY") > 15, 1).otherwise(0))

# Diviser les données en ensemble d'entraînement et de test
train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

# Définir et entraîner les modèles
models = {
    'Logistic Regression': LogisticRegression(featuresCol="scaled_features"),
    'Random Forest': RandomForestClassifier(featuresCol="scaled_features"),
    'Gradient Boosting': GBTClassifier(featuresCol="scaled_features"),
    'Decision Tree': DecisionTreeClassifier(featuresCol="scaled_features")
}

# Pipeline commun
pipeline_stages = indexers + encoders + [assembler, scaler]
results = {}

# Optimisation des hyperparamètres avec CrossValidator et ParamGridBuilder
param_grid = ParamGridBuilder() \
    .addGrid(models['Logistic Regression'].regParam, [0.1, 0.01]) \
    .addGrid(models['Random Forest'].numTrees, [10, 20]) \
    .build()

cross_validator = CrossValidator(estimator=models['Logistic Regression'],
                                 estimatorParamMaps=param_grid,
                                 evaluator=BinaryClassificationEvaluator(),
                                 numFolds=3)

for name, model in models.items():
    print(f"Training {name}...")
    pipeline = Pipeline(stages=pipeline_stages + [model])
    model_fit = pipeline.fit(train_data)
    predictions = model_fit.transform(test_data)
    
    evaluator = BinaryClassificationEvaluator()
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
    
    print(f"{name} - ROC AUC: {accuracy:.4f}")
    results[name] = accuracy

# Comparaison des résultats
print("Comparaison des modèles :")
df_results = pd.DataFrame(list(results.items()), columns=['Modèle', 'ROC AUC'])
print(df_results)

# Visualisation des résultats
plt.figure(figsize=(10, 6))
sns.barplot(x='Modèle', y='ROC AUC', data=df_results)
plt.title('Comparaison des modèles')
plt.ylabel('Score')
plt.xlabel('Modèles')
plt.xticks(rotation=45)
plt.show()

# Arrêter la session Spark
spark.stop()
