import os
import pandas as pd
import streamlit as st
from streamlit import cache_resource, cache_data

@cache_resource
def load_data_and_get_schema(file_path_glob):
    """Carrega os dados e retorna o data_loader."""
    from agent.data_loader import DataLoader
    data_loader = DataLoader(file_path_glob, samples=50, query="SELECT 1", file_type="parquet")
    try:
        data_loader.connect_to_db()
        return data_loader
    except Exception as e:
        st.error(f"Falha ao carregar dados: {e}")
        return None

@cache_data
def get_namespaces_from_filesystem(parquet_dir):
    """Obtém a lista de namespaces a partir dos subdiretórios."""
    if not os.path.exists(parquet_dir): return []
    try:
        return [d.split("=")[1] for d in os.listdir(parquet_dir) if os.path.isdir(os.path.join(parquet_dir, d)) and "=" in d]
    except Exception: return []

def _get_base_pivot_query(namespace, metrics_to_pivot):
    """Função auxiliar para construir a base da query PIVOT."""
    namespace_filter_clause = f"AND ocnr_tx_namespace = '{namespace}'" if namespace != 'All' else ""
    return f"""
        PIVOT (
            SELECT ocnr_dt_date, ocnr_tx_namespace, ocnr_tx_query, ocnr_nm_result
            FROM logs
            WHERE ocnr_tx_query IN {tuple(metrics_to_pivot)}
            {namespace_filter_clause}
        )
        ON ocnr_tx_query
        USING FIRST(ocnr_nm_result)
        GROUP BY ocnr_dt_date, ocnr_tx_namespace
    """

@cache_data
def get_pivoted_dataframe_for_training(_data_loader, namespace, metrics_to_pivot, sample_size):
    """
    Busca os dados mais ANTIGOS para TREINAMENTO, simulando um cenário real.
    CRITICAL FIX: Changed from random sampling to time-based split.
    """
    if not _data_loader: return pd.DataFrame()
    
    base_query = _get_base_pivot_query(namespace, metrics_to_pivot)
    # CORREÇÃO: Usar ORDER BY ASC e LIMIT para pegar os dados mais antigos para treino.
    training_query = f"SELECT * FROM ({base_query}) ORDER BY ocnr_dt_date ASC LIMIT {sample_size};"
    
    try:
        with st.spinner(f"Buscando os {sample_size} registros mais antigos para treino..."):
            df = _data_loader.connect.execute(training_query).fetchdf()
        df['ocnr_dt_date'] = pd.to_datetime(df['ocnr_dt_date'])
        # A ordenação já está correta pela query
        return df
    except Exception as e:
        st.warning(f"Falha ao buscar dados de treino: {e}")
        return pd.DataFrame()

@cache_data
def get_pivoted_dataframe_for_testing(_data_loader, namespace, metrics_to_pivot, limit):
    """Busca os dados MAIS RECENTES para TESTE e VISUALIZAÇÃO."""
    if not _data_loader: return pd.DataFrame()

    base_query = _get_base_pivot_query(namespace, metrics_to_pivot)
    testing_query = f"SELECT * FROM ({base_query}) ORDER BY ocnr_dt_date DESC LIMIT {limit};"
    
    try:
        with st.spinner(f"Buscando os {limit} registros mais recentes para teste..."):
            df = _data_loader.connect.execute(testing_query).fetchdf()
        df['ocnr_dt_date'] = pd.to_datetime(df['ocnr_dt_date'])
        # Reverte para ordem cronológica (mais antigo -> mais novo)
        return df.iloc[::-1].reset_index(drop=True)
    except Exception as e:
        st.warning(f"Falha ao buscar dados de teste: {e}")
        return pd.DataFrame()

@cache_data
def get_single_metric_data(_data_loader, namespace, metric, limit):
    """Busca dados de uma única métrica para análises de série temporal."""
    if not _data_loader: return pd.DataFrame()
    
    namespace_filter_clause = f"AND ocnr_tx_namespace = '{namespace}'" if namespace != 'All' else ""
    query = f"""
        SELECT ocnr_dt_date, ocnr_nm_result as "{metric}"
        FROM logs
        WHERE ocnr_tx_query = '{metric}'
        {namespace_filter_clause}
        ORDER BY ocnr_dt_date DESC
        LIMIT {limit}
    """
    try:
        with st.spinner(f"Buscando os {limit} registros mais recentes da métrica '{metric}'..."):
            df = _data_loader.connect.execute(query).fetchdf()
        df['ocnr_dt_date'] = pd.to_datetime(df['ocnr_dt_date'])
        return df.iloc[::-1].sort_values('ocnr_dt_date').set_index('ocnr_dt_date')
    except Exception as e:
        st.warning(f"Falha ao buscar dados da métrica {metric}: {e}")
        return pd.DataFrame()