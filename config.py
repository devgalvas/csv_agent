# Caminho padrão para os dados processados
PARQUET_DEFAULT_PATH = "/home/lucas/UNIFEI/Vertis/vertis_research_agent/archive/partitioned_parquet"

# Opções para a interface do usuário (UI)
MODEL_OPTIONS = [
    'lightgbm',
    'xgboost',
    'random_forest',
    'linear',
    'ridge',
    'lasso',
    'elasticnet'
]

TARGET_OPTIONS = [
    'NAMESPACE_MEMORY_USAGE',
    'NAMESPACE_CPU_USAGE',
    'NAMESPACE_POD_COUNT'
]

# Métricas que serão pivotadas para se tornarem features
METRICS_TO_PIVOT = [
    'NAMESPACE_CPU_USAGE', 'NAMESPACE_MEMORY_USAGE',
    'NAMESPACE_POD_COUNT', 'NAMESPACE_POD_RESTARTS',
    'NAMESPACE_NETWORK_TX', 'NAMESPACE_NETWORK_RX'
]
