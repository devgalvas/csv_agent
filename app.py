import os
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

# Importações dos seus módulos
from agent.predictive_model import PredictiveModel

# --- Funções de Cache ---

@st.cache_resource
def load_data_and_get_schema(file_path_glob):
    """Carrega os dados usando o DataLoader e retorna o objeto data_loader."""
    from agent.data_loader import DataLoader
    data_loader = DataLoader(file_path_glob, samples=50, query="SELECT 1", file_type="parquet")
    try:
        data_loader.connect_to_db()
        return data_loader
    except Exception as e:
        st.error(f"Falha ao carregar dados: {e}")
        return None

@st.cache_data
def get_namespaces_from_filesystem(parquet_dir):
    """Obtém a lista de namespaces a partir dos subdiretórios."""
    if not os.path.exists(parquet_dir): return []
    try:
        return [d.split("=")[1] for d in os.listdir(parquet_dir) if os.path.isdir(os.path.join(parquet_dir, d)) and "=" in d]
    except Exception: return []

@st.cache_data
def get_pivoted_dataframe(_data_loader, namespace, limit):
    """Cria um DataFrame "largo" (wide) usando PIVOT, buscando os dados mais recentes."""
    metrics_to_pivot = [
        'NAMESPACE_CPU_USAGE', 'NAMESPACE_MEMORY_USAGE',
        'NAMESPACE_POD_COUNT', 'NAMESPACE_POD_RESTARTS',
        'NAMESPACE_NETWORK_TX', 'NAMESPACE_NETWORK_RX'
    ]
    namespace_filter_clause = f"AND ocnr_tx_namespace = '{namespace}'" if namespace != 'All' else ""

    # --- LÓGICA CORRIGIDA: Adicionado ORDER BY ocnr_dt_date DESC ---
    # Isso garante que o LIMIT pegue os dados mais recentes, não uma amostra aleatória.
    pivot_query = f"""
        WITH PivotedData AS (
            PIVOT (
                SELECT ocnr_dt_date, ocnr_tx_namespace, ocnr_tx_query, ocnr_nm_result
                FROM logs
                WHERE ocnr_tx_query IN {tuple(metrics_to_pivot)}
                {namespace_filter_clause}
            )
            ON ocnr_tx_query
            USING FIRST(ocnr_nm_result)
            GROUP BY ocnr_dt_date, ocnr_tx_namespace
        )
        SELECT * FROM PivotedData
        ORDER BY ocnr_dt_date DESC
        LIMIT {limit};
    """
    try:
        with st.spinner(f"Buscando os {limit} registros mais recentes e pivotando..."):
            feature_df = _data_loader.connect.execute(pivot_query).fetchdf()
        # Reverte a ordem para que fique cronológica (do mais antigo para o mais novo)
        return feature_df.iloc[::-1].reset_index(drop=True)
    except Exception as e:
        st.warning(f"Falha ao executar PIVOT: {e}")
        return pd.DataFrame()

class App:
    def __init__(self, title, **page_config):
        st.set_page_config(page_title=title, **page_config)
        self._apply_custom_styling()
        self._create_header(title)
        load_dotenv()
        
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
            st.session_state.data_loader = None
        
        self._create_sidebar()

    def _apply_custom_styling(self):
        st.markdown("""<style>...</style>""", unsafe_allow_html=True)

    def _create_header(self, title):
        col1, col2 = st.columns([1, 5])
        with col1:
            # Usando um placeholder para evitar o erro de arquivo não encontrado
            st.markdown('<div style="width:90px; height:90px; background-color:#112240; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size: 2rem; color: #64ffda; font-weight: bold;">V</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<h1 style="color:#64ffda;">📊 {title}</h1>', unsafe_allow_html=True)
            st.caption('<span style="color:#e3f0fc;">🚀 Agente de IA para análise de dados do OpenShift</span>', unsafe_allow_html=True)

    def _create_sidebar(self):
        with st.sidebar:
            st.header("Configurações")
            self.namespace_dir = st.text_input("Diretório Parquet", "/home/lucas/UNIFEI/Vertis/vertis_research_agent/archive/partitioned_parquet")
            namespaces = get_namespaces_from_filesystem(self.namespace_dir)
            self.namespace_selection = st.selectbox("Selecione o Namespace", ["All"] + namespaces)
            if st.sidebar.button("🚀 Carregar Dados e Analisar"):
                self._handle_parquet_loading()

    def _handle_parquet_loading(self):
        if not self.namespace_dir or not os.path.exists(self.namespace_dir):
            st.error("O diretório Parquet não existe.")
            return
        file_path_glob = os.path.join(self.namespace_dir, "**", "*.parquet")
        with st.spinner("Conectando ao banco de dados..."):
            data_loader = load_data_and_get_schema(file_path_glob)
        if data_loader:
            st.session_state.data_loader = data_loader
            st.session_state.data_loaded = True
            st.rerun()
        else:
            st.error("Falha ao carregar os dados.")

    def plot_forecast_comparison(self, model_types, target_column, sample_size):
        # 1. Obter os dados mais recentes
        data_df = get_pivoted_dataframe(st.session_state.data_loader, self.namespace_selection, sample_size)
        
        if data_df.empty or target_column not in data_df.columns:
            st.error(f"Não há dados para a métrica '{target_column}' neste namespace.")
            return

        # 2. Pré-processar os dados UMA VEZ
        temp_model = PredictiveModel('linear')
        X, y = temp_model.preprocess(data_df, target_column)

        # 3. Dividir em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
        results_df = pd.DataFrame({'Valores Reais': y_test})

        # 4. Treinar e prever
        with st.spinner("Treinando e avaliando modelos..."):
            for model_type in model_types:
                try:
                    model = PredictiveModel(model_type)
                    model.train(X_train, y_train)
                    _, predictions = model.evaluate(X_test, y_test)
                    results_df[f'Previsão_{model_type}'] = predictions
                except Exception as e:
                    st.warning(f"Falha ao treinar o modelo '{model_type}': {e}")
        
        # 5. Plotar
        y_axis_title = "Valor da Métrica"
        if target_column == 'NAMESPACE_MEMORY_USAGE':
            results_df = results_df / 1e9 # Converte para GB
            y_axis_title = "Uso de Memória (GB)"

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Valores Reais'],
                                 mode='lines+markers', name='Valores Reais', line=dict(color='royalblue', width=3)))
        for col in results_df.columns[1:]:
            fig.add_trace(go.Scatter(x=results_df.index, y=results_df[col],
                                     mode='lines+markers', name=col, line=dict(dash='dot', width=2)))
        
        fig.update_layout(
            title=f'Comparativo de Modelos para "{target_column}"',
            xaxis_title='Tempo (Índice de Amostra)',
            yaxis_title=y_axis_title,
            legend_title='Legenda',
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)

    def run(self):
        if not st.session_state.data_loaded:
            st.info("⬅️ Configure o diretório e clique em 'Carregar Dados' para começar.")
            return

        st.header("🤖 Análise Preditiva")
        model_options = ['random_forest', 'lightgbm', 'xgboost', 'linear', 'ridge', 'lasso', 'elasticnet']
        selected_models = st.multiselect("Selecione os modelos para comparar:", model_options, default=['lightgbm', 'xgboost'])
        target_column = st.selectbox("Selecione a Métrica Alvo:", ['NAMESPACE_CPU_USAGE', 'NAMESPACE_MEMORY_USAGE'])
        sample_size = st.slider("Nº de registros recentes para análise:", 500, 10000, 2000, 250)


        if st.button("📊 Gerar Gráfico Comparativo"):
            self.plot_forecast_comparison(selected_models, target_column, sample_size)

if __name__ == "__main__":
    app = App("Vertis Data Consultant", layout="wide")
    app.run()
