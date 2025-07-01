import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

class PredictiveModel:
    def __init__(self, model_path: str, model_type: str):
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.metrics = {}
        self.target_column = None

    def preprocess(self, df, target_column=None):
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        df = df.dropna()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df = df.drop(columns=['timestamp'])
        
        if target_column and target_column in df.columns:
            target_series = df[target_column].copy()
            categorical_columns = df.select_dtypes(include=['object']).columns.drop(target_column, errors='ignore')
            df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
            df[target_column] = pd.to_numeric(target_series, errors='coerce')  # Forçar conversão numérica
        else:
            categorical_columns = df.select_dtypes(include=['object']).columns
            df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        
        return df

    def train(self, df: pd.DataFrame, target_column: str, segment_column: str = None, segment_value: str = None):
        print(f"Training with initial DataFrame columns: {df.columns.tolist()}")
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        if segment_column and segment_value:
            df = df[df[segment_column] == segment_value]
            print(f"After segmentation by {segment_column} = {segment_value}, columns: {df.columns.tolist()}")

        df_processed = self.preprocess(df, target_column)
        print(f"After preprocess, columns: {df_processed.columns.tolist()}")

        if target_column not in df_processed.columns:
            raise ValueError(f"Target column '{target_column}' lost after preprocessing")
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]

        # Forçar conversão numérica e remover NaN
        y = pd.to_numeric(y, errors='coerce').dropna()
        if y.empty:
            raise ValueError(f"Target column '{target_column}' cannot be converted to numeric")

        print(f"y dtype: {y.dtype}")
        print(f"y head: {y.head().to_dict()}")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.model_type == "linear":
            self.model = LinearRegression()
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier()
        elif self.model_type == "prophet":
            df_prophet = df[['timestamp', target_column]].rename(columns={'timestamp': 'ds', target_column: 'y'})
            if df_prophet.empty or 'ds' not in df_prophet.columns or 'y' not in df_prophet.columns:
                raise ValueError("Prophet requires 'timestamp' and target column")
            self.model = Prophet()
            self.model.fit(df_prophet)
            future = self.model.make_future_dataframe(periods=30)
            forecast = self.model.predict(future)
            self.metrics = {"forecast": forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]}
            joblib.dump(self.model, self.model_path)
            return

        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

        if self.model_type == "linear":
            self.metrics = {
                "MSE": mean_squared_error(self.y_test, self.y_pred),
                "R2": r2_score(self.y_test, self.y_pred)
            }
        else:
            self.metrics = {
                "Accuracy": accuracy_score(self.y_test, self.y_pred),
                "Precision": precision_score(self.y_test, self.y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(self.y_test, self.y_pred, average="weighted", zero_division=0),
                "F1": f1_score(self.y_test, self.y_pred, average="weighted", zero_division=0)
            }
        joblib.dump(self.model, self.model_path)

    def load(self):
        self.model = joblib.load(self.model_path)
    
    def predict(self, df, segment_column=None, segment_value=None):
        if segment_column and segment_value:
            df = df[df[segment_column] == segment_value]
        df = self.preprocess(df, self.target_column)
        X = df.select_dtypes(include=['number']).drop(columns=[self.target_column], errors='ignore')
        if X.empty:
            raise ValueError("No numeric features available for prediction")
        return self.model.predict(X)
    
    def detect_anomalies(self, df, column, segment_column=None, segment_value=None):
        if segment_column and segment_value:
            df = df[df[segment_column] == segment_value]
        df = self.preprocess(df, column)
        data = df[[column]].values
        if data.size == 0:
            raise ValueError(f"No data available for column '{column}'")
        model = IsolationForest(contamination=0.1, random_state=42)
        predictions = model.fit_predict(data)
        df['anomaly'] = predictions
        return df[df['anomaly'] == -1]
    
    def check_alerts(self, df, metric, threshold=90, segment_column=None, segment_value=None):
        if segment_column and segment_value:
            df = df[df[segment_column] == segment_value]
        df = self.preprocess(df)
        alerts = df[df[metric] > threshold][['ocnr_tx_namespace', metric]]
        return alerts if not alerts.empty else pd.DataFrame(columns=['ocnr_tx_namespace', metric])