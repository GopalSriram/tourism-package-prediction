import pandas as pd
import os
import joblib
import mlflow
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-package-experiment")

api = HfApi(token=os.getenv("HF_TOKEN"))

# Load train/test data from Hugging Face
Xtrain = pd.read_csv("hf://datasets/GopalSriram/tourism-package-prediction/Xtrain.csv")
Xtest  = pd.read_csv("hf://datasets/GopalSriram/tourism-package-prediction/Xtest.csv")
ytrain = pd.read_csv("hf://datasets/GopalSriram/tourism-package-prediction/ytrain.csv").squeeze()
ytest  = pd.read_csv("hf://datasets/GopalSriram/tourism-package-prediction/ytest.csv").squeeze()

print("✅ Data loaded from Hugging Face")
print("Xtrain shape:", Xtrain.shape)
print("Xtest shape:", Xtest.shape)

# Define numeric and categorical features
numeric_features = [
    'Age', 'DurationOfPitch', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips',
    'PitchSatisfactionScore', 'NumberOfChildrenVisiting', 'MonthlyIncome'
]
categorical_features = [
    'TypeofContact', 'Occupation', 'Gender',
    'MaritalStatus', 'Designation', 'ProductPitched'
]

# Preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    remainder='passthrough'
)

# Model
rf_model = RandomForestClassifier(random_state=42)

# Full pipeline
model_pipeline = make_pipeline(preprocessor, rf_model)

# Hyperparameter grid
param_grid = {
    'randomforestclassifier__n_estimators': [50, 100],
    'randomforestclassifier__max_depth': [3, 5, 10],
    'randomforestclassifier__min_samples_split': [2, 5],
}

# Start MLflow run
with mlflow.start_run():

    # Grid search with CV
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results['params'][i])
            mlflow.log_metric("mean_test_score", results['mean_test_score'][i])
            mlflow.log_metric("std_test_score",  results['std_test_score'][i])

    # Best model
    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)

    # Evaluate
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test  = best_model.predict(Xtest)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report  = classification_report(ytest,  y_pred_test,  output_dict=True)

    mlflow.log_metrics({
        "train_accuracy":  train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall":    train_report['1']['recall'],
        "train_f1":        train_report['1']['f1-score'],
        "test_accuracy":   test_report['accuracy'],
        "test_precision":  test_report['1']['precision'],
        "test_recall":     test_report['1']['recall'],
        "test_f1":         test_report['1']['f1-score'],
    })

    # Save model locally
    model_path = "best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"✅ Best model saved: {model_path}")

    # Upload model to HF model hub
    model_repo_id = "GopalSriram/tourism-package-model"
    try:
        api.repo_info(repo_id=model_repo_id, repo_type="model")
        print(f"Model repo already exists.")
    except RepositoryNotFoundError:
        create_repo(repo_id=model_repo_id, repo_type="model", private=False)
        print(f"✅ Model repo created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=model_repo_id,
        repo_type="model",
    )
    print(f"✅ Model uploaded to HF: {model_repo_id}")
