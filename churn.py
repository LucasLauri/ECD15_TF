import pandas as pd
import mlflow
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from mlflow.models.signature import infer_signature

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("customerchurn_ECD15_LL")

def preprocess_data(df):
    """
    Faz o pré-processamento dos dados de um df.
    Neste caso, remove a coluna 'customerID', mapeia yes-no para 1-0, converte int64 para float64,
    converte object para string com um LabelEncoder da coluna e, por fim, preenche N.A. como 0
    """
    df.drop(columns=["customerID"], inplace=True, errors="ignore")
    df.replace({"Yes": 1, "No": 0}, inplace=True)
    df = df.infer_objects(copy=False)

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("float64")
    
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)
        df[col] = LabelEncoder().fit_transform(df[col])

    df.fillna(0, inplace=True)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    mlflow_dataset = mlflow.data.from_pandas(df, targets="Churn")

    return X, y, mlflow_dataset


def load_data(dataset_name):
    """
    Carrega um .csv como um dataframe do pandas
    """
    df = pd.read_csv(dataset_name)
    return df

def train_random_forest(X, y, dataset_name, mlflow_dataset):
    """
    Treina um modelo de random forest variando uma grid de parametros para tentar encontrar a melhor combinação deles
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
    }

    rf = RandomForestClassifier(random_state=42)

    for params in (dict(zip(param_grid.keys(), values)) for values in 
                   [(n, d, s) for n in param_grid["n_estimators"] 
                               for d in param_grid["max_depth"] 
                               for s in param_grid["min_samples_split"]]):
        
        rf.set_params(**params)
        rf.fit(X_train, y_train)
        y_test_pred = rf.predict(X_test)

        with mlflow.start_run(run_name=f"RF_{params['n_estimators']}_{params['max_depth']}_{params['min_samples_split']}"):
            mlflow.log_input(mlflow_dataset, context="training")
            mlflow.log_params(params)
            mlflow.set_tag("dataset_used", dataset_name)

            signature = infer_signature(X_train, y_test_pred)
            model_info = mlflow.sklearn.log_model(rf, "random_forest_model", 
                                     signature=signature, 
                                     input_example=X_train, 
                                     registered_model_name="RandomForestGridSearch")

            #carrega modelo do mlflow
            loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
            predictions = loaded_model.predict(X_test)
            result = pd.DataFrame(X_test, columns=X.columns.values)
            result["label"] = y_test.values
            result["predictions"] = predictions

            #usa o mflow para avaliar o modelo carregado
            mlflow.evaluate(
                data=result,
                targets="label",
                predictions="predictions",
                model_type="classifier",
            )

            print(result[:5])

    return rf

def main():
    # dataset
    dataset_name = "datasets\\CustomerChurn.csv"
    df = load_data(dataset_name)
    # pré-processamento dos dados
    X, y, mlflow_dataset = preprocess_data(df)
    # treinamento do modelo
    rf_model = train_random_forest(X, y, dataset_name, mlflow_dataset)

if __name__ == "__main__":
    main()