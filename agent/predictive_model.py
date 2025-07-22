import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Modelos de Regressão Padrão
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor

# Modelos de Gradient Boosting (Avançados)
import xgboost as xgb
import lightgbm as lgb

# Modelo de Série Temporal
from prophet import Prophet


class PredictiveModel:
    """
    Uma classe versátil para treinar e avaliar diferentes modelos de regressão e série temporal.
    """
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.y_pred = None
        self.metrics = {}
        self.is_time_series = (self.model_type == 'prophet')

    def train(self, df: pd.DataFrame, target_column: str):
        """
        Treina o modelo selecionado com o DataFrame fornecido.
        A lógica de pré-processamento agora está integrada aqui.
        """
        if df.empty:
            raise ValueError("O DataFrame de treinamento está vazio.")

        # --- Caminho de Execução para o Modelo Prophet (Série Temporal) ---
        if self.is_time_series:
            if 'ocnr_dt_date' not in df.columns:
                raise ValueError("Para o Prophet, a coluna 'ocnr_dt_date' é obrigatória.")
            
            ts_df = df[['ocnr_dt_date', target_column]].copy()
            ts_df.rename(columns={'ocnr_dt_date': 'ds', target_column: 'y'}, inplace=True)
            ts_df = ts_df.dropna()

            self.model = Prophet()
            self.model.fit(ts_df)
            
            self.metrics = {"status": "Modelo Prophet treinado. Use-o para fazer previsões."}
            return

        # --- Caminho de Execução para Modelos de Regressão ---
        
        # 1. Engenharia de Features de Tempo
        if 'ocnr_dt_date' in df.columns:
            df['ocnr_dt_date'] = pd.to_datetime(df['ocnr_dt_date'])
            df['hour'] = df['ocnr_dt_date'].dt.hour
            df['day_of_week'] = df['ocnr_dt_date'].dt.dayofweek
            df['day_of_month'] = df['ocnr_dt_date'].dt.day
            df['month'] = df['ocnr_dt_date'].dt.month
        
        # 2. Limpeza de NaNs no Alvo
        df = df.dropna(subset=[target_column])
        if df.empty:
            raise ValueError("DataFrame ficou vazio após remover NaNs da coluna alvo.")

        # 3. Separar Alvo (y) e Features (X)
        y = df[target_column]
        
        # --- CORREÇÃO PRINCIPAL APLICADA AQUI ---
        # Primeiro, removemos o alvo para definir as features.
        features_df = df.drop(columns=[target_column])
        
        # Em seguida, selecionamos APENAS as colunas numéricas para o X.
        # Isso garante que colunas como 'ocnr_tx_namespace' (object) sejam descartadas.
        X = features_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32', 'bool'])

        # Garante que X e y continuem alinhados
        y = y.loc[X.index]

        # 4. Divisão em Treino e Teste
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 5. Seleção e Treinamento do Modelo
        model_zoo = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'elasticnet': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'xgboost': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }
        
        if self.model_type not in model_zoo:
            raise ValueError(f"Tipo de modelo '{self.model_type}' não é suportado.")

        self.model = model_zoo[self.model_type]
        self.model.fit(self.X_train, self.y_train)

        # 6. Avaliação
        self.y_pred = self.model.predict(self.X_test)
        self.metrics = {
            "MSE": mean_squared_error(self.y_test, self.y_pred),
            "R2": r2_score(self.y_test, self.y_pred)
        }

    def predict_forecast(self, periods=30):
        if not self.is_time_series or not self.model:
            raise ValueError("A função de forecast só está disponível para um modelo Prophet treinado.")
        
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        
        return forecast