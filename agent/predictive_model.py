import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

class PredictiveModel:
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.is_linear_model = model_type in ['linear', 'ridge', 'lasso', 'elasticnet']
        self.model_zoo = {
            'linear': LinearRegression(), 'ridge': Ridge(), 'lasso': Lasso(),
            'elasticnet': ElasticNet(),
            'random_forest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'xgboost': xgb.XGBRegressor(random_state=42, n_jobs=-1),
            'lightgbm': lgb.LGBMRegressor(random_state=42, n_jobs=-1)
        }

    def preprocess(self, df: pd.DataFrame, target_column: str):
        if target_column not in df.columns:
            raise KeyError(f"A coluna alvo '{target_column}' não foi encontrada.")

        date_index = pd.to_datetime(df['ocnr_dt_date'])

        df['hour'] = date_index.dt.hour
        df['day_of_week'] = date_index.dt.dayofweek
        df['day_of_month'] = date_index.dt.day
        df['month'] = date_index.dt.month
        
        numeric_features = df.select_dtypes(include=['number', 'bool'])
        imputed_features = numeric_features.ffill().bfill().fillna(0)
        df[imputed_features.columns] = imputed_features
        
        df = df.dropna(subset=[target_column])
        if df.empty:
            raise ValueError("DataFrame vazio após remover NaNs da coluna alvo.")

        y = df[target_column]
        features_df = df.drop(columns=[target_column], errors='ignore')
        X = features_df.select_dtypes(include=['number', 'bool'])

        y = y.loc[X.index]
        X.index = date_index.loc[X.index]
        y.index = date_index.loc[y.index]

        return X, y

    def train(self, X_train, y_train):
        if self.model_type not in self.model_zoo:
            raise ValueError(f"Tipo de modelo '{self.model_type}' não suportado.")
        self.model = self.model_zoo[self.model_type]

        if self.is_linear_model:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.model.fit(X_train_scaled, y_train)
        else:
            self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        if self.is_linear_model and self.scaler:
            X_test_scaled = self.scaler.transform(X_test)
            y_pred = self.model.predict(X_test_scaled)
        else:
            y_pred = self.model.predict(X_test)
        metrics = { "MSE": mean_squared_error(y_test, y_pred), "R2": r2_score(y_test, y_pred) }
        return metrics, y_pred
