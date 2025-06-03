import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class PredictiveModel:
    def __init__(self, model_path: str, model_type: str):
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.metrics = {}

    def preprocess(self, df):
        df = df.dropna()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            # Drop the original timestamp column
            df = df.drop(columns=['timestamp'])
        
        # Codify categorical variables (e. g., NODE_ROLES, NAMESPACE_STATUS)
        categorical_columns = df.select_dtypes(include=['object']).columns
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

        return df
    
    def train(self, df, target_column, segment_column=None, segment_value=None):
        if segment_column and segment_value:
            df = df[df[segment_column] == segment_value]

        if df.empty:
            raise ValueError(f"No data found for {segment_column} = {segment_value}")
        
        df = self.preprocess(df)
        x = df.drop(columns=[target_column])
        y = df[target_column]

        x = x.select_dtypes(include=['number'])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        if self.model_type == "linear":
            self.model = LinearRegression()
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier()
        elif self.model_type == "prophet":
            df_prophet = df[['timestamp', target_column]].rename(columns={
                'timestamp': 'ds', target_column: 'y'})
            self.model = Prophet()
            self.model.fit(df_prophet)
            future = self.model.make_future_dataframe(periods=30)
            forecast = self.model.predict(future)
            self.metrics = {"forecast": forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]}
            joblib.dump(self.model, self.model_path)
            return

        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)

        if self.model_type == "linear":
            self.metrics = {
                "MSE": mean_squared_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred)
            }
        else:
            self.metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0)
            }
        joblib.dump(self.model, self.model_path)

    def load(self):
        self.model = joblib.load(self.model_path)
    
    def predict(self, df, column, segment_column=None, segment_value=None):
        if segment_column and segment_value:
            df = df[df[segment_column] == segment_value]

        df = self.preprocess(df)
        X = df.select_dtypes(include=['number'])
        return self.model.predict(X)
    
    # To detect those important outliers
    def detect_anomalies(self, df, column, segment_column=None, segment_value=None):
        if segment_column and segment_value:
            df = df[df[segment_column] == segment_value]

        df = self.preprocess(df)
        data = df[[column]].values
        model = IsolationForest(contamination=0.1, random_state=42)
        predictions = model.fit_predict(data)
        df['anomaly'] = predictions # -1 for anomaly, 1 for normal
        return df[df['anomaly'] == -1]
    
    def check_alerts(self, df, metric, threshold=90, segment_column=None, segment_value=None):
        if segment_column and segment_value:
            df = df[df[segment_column] == segment_value]

        alerts = df[df[metric] > threshold][['NAMESPACE', metric]]
        return alerts if not alerts.empty else pd.DataFrame(columns=['NAMESPACE', metric])
    
