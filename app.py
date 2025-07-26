import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from agent.predictive_model import PredictiveModel
import streamlit as st
import shap
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split


@st.cache_resource
def load_data_and_get_schema(file_path_glob):
    from agent.data_loader import DataLoader
    data_loader = DataLoader(file_path_glob, samples=50, query="SELECT 1", file_type="parquet")
    try:
        data_loader.connect_to_db()
        columns = data_loader.connect.execute("DESCRIBE logs").fetchdf()['column_name'].tolist()
        return data_loader, columns
    except Exception as e:
        st.error(f"Falha ao carregar dados: {e}")
        return None, []

@st.cache_data
def get_namespaces_from_filesystem(parquet_dir):
    if not os.path.exists(parquet_dir): return []
    try:
        return [d.split("=")[1] for d in os.listdir(parquet_dir) if os.path.isdir(os.path.join(parquet_dir, d)) and "=" in d]
    except Exception: return []

@st.cache_data
def get_pivoted_dataframe_for_training(_data_loader, namespace, sample_size):
    if not _data_loader:
        return pd.DataFrame()
    
    metrics_to_pivot = [
        'NAMESPACE_CPU_USAGE', 'NAMESPACE_MEMORY_USAGE',
        'NAMESPACE_POD_COUNT', 'NAMESPACE_POD_RESTARTS',
        'NAMESPACE_NETWORK_TX', 'NAMESPACE_NETWORK_RX'
    ]
    
    namespace_filter_clause = ""
    if namespace != 'All':
        namespace_filter_clause = f"AND ocnr_tx_namespace = '{namespace}'"

    pivot_query = f"""
        PIVOT (
            SELECT ocnr_dt_date, ocnr_tx_namespace, ocnr_tx_query, ocnr_nm_result
            FROM logs
            WHERE ocnr_tx_query IN {tuple(metrics_to_pivot)}
            {namespace_filter_clause}
        )
        ON ocnr_tx_query
        USING FIRST(ocnr_nm_result)
        GROUP BY ocnr_dt_date, ocnr_tx_namespace
        LIMIT {sample_size};
    """
    
    try:
        with st.spinner("Executando PIVOT para criar features..."):
            feature_df = _data_loader.connect.execute(pivot_query).fetchdf()
        return feature_df
    except Exception as e:
        st.warning(f"Falha ao executar PIVOT: {e}")
        return pd.DataFrame()

class App:
    def __init__(self, title: str, **page_config):
        st.set_page_config(page_title=title, **page_config)
        self._apply_custom_styling()
        self._create_header(title)
        load_dotenv()
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
            st.session_state.data_loader = None
            st.session_state.columns = []
            st.session_state.trained_model = None
        self._create_sidebar()

    def _apply_custom_styling(self):
        st.markdown("""<style>body, .main { background-color: #0a192f !important; color: #e3f0fc !important; } .stApp { background: linear-gradient(135deg, #0a192f 0%, #112240 100%) !important; color: #e3f0fc !important; } .stButton>button { background-color: #1565c0; color: #e3f0fc; border-radius: 8px; font-weight: bold; border: 1px solid #64ffda; } .stTextInput>div>div>input { background-color: #112240; color: #e3f0fc; border: 1px solid #1565c0; } .st-expanderHeader { color: #64ffda !important; font-weight: bold; } footer {visibility: hidden;}</style>""", unsafe_allow_html=True)

    def _create_header(self, title):
        col1, col2 = st.columns([1, 5])
        with col1:
            st.image("https://placehold.co/90x90/0a192f/64ffda?text=V", width=90)
        with col2:
            st.markdown(f'<h1 style="color:#64ffda;">📊 {title}</h1>', unsafe_allow_html=True)
            st.caption('<span style="color:#e3f0fc;">🚀 Agente de IA para análise de dados do OpenShift</span>', unsafe_allow_html=True)

    def _create_sidebar(self):
        with st.sidebar:
            st.header("Configurações")
            self.namespace_dir = st.text_input("Diretório Parquet", "/home/lucas/UNIFEI/Vertis/vertis_research_agent/archive/partitioned_parquet")
            namespaces = get_namespaces_from_filesystem(self.namespace_dir)
            self.namespace_selection = st.selectbox("Selecione o Namespace", ["All"] + namespaces)
            if st.button("🚀 Carregar Dados e Analisar"):
                self._handle_parquet_loading()
            st.markdown("---")
            self.show_data = st.checkbox("🔍 Mostrar amostra dos dados")
            st.info("Feito por [Lucas Galvão Freitas](https://github.com/devgalvas)")

    def _handle_parquet_loading(self):
        if not self.namespace_dir or not os.path.exists(self.namespace_dir):
            st.error("O diretório Parquet não existe.")
            return
        file_path_glob = os.path.join(self.namespace_dir, "**", "*.parquet")
        with st.spinner("Conectando ao banco de dados..."):
            data_loader, columns = load_data_and_get_schema(file_path_glob)
        if data_loader and columns:
            st.session_state.data_loader = data_loader
            st.session_state.columns = columns
            st.session_state.data_loaded = True
            st.rerun()
        else:
            st.error("Falha ao carregar os dados.")

    def _plot_forecast_comparison(self, model_types, target_column, sample_size):
        data_df = get_pivoted_dataframe_for_training(st.session_state.data_loader, self.namespace_selection, sample_size * 5)
        
        if data_df.empty or target_column not in data_df.columns:
            st.error(f"Não há dados suficientes para a métrica '{target_column}' neste namespace.")
            return

        # 1. Pré-processar os dados UMA VEZ antes de tudo
        temp_model_for_preprocessing = PredictiveModel('linear')
        X, y = temp_model_for_preprocessing.preprocess(data_df.reset_index(), target_column)

        # 2. Dividir os dados JÁ PROCESSADOS em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

        results_df = pd.DataFrame(y_test).rename(columns={target_column: 'Valores Reais'})

        with st.spinner("Treinando e avaliando modelos..."):
            for model_type in model_types:
                try:
                    # 3. Treinar cada modelo com o MESMO conjunto de treino limpo
                    model = PredictiveModel(model_type)
                    model.train(X_train, y_train)
                    
                    # 4. Fazer previsões no MESMO conjunto de teste limpo
                    model.evaluate(X_test, y_test)
                    results_df[f'Previsão_{model_type}'] = model.y_pred
                except Exception as e:
                    st.warning(f"Falha ao treinar o modelo '{model_type}': {e}")
        
        if len(results_df.columns) < 2:
            st.error("Nenhum modelo foi treinado com sucesso. Não é possível gerar o gráfico.")
            return

        # 5. Criar o gráfico interativo com Plotly
        results_df = results_df.sort_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Valores Reais'],
                                 mode='lines', name='Valores Reais', line=dict(color='royalblue', width=4)))
        for model_name in results_df.columns[1:]:
            fig.add_trace(go.Scatter(x=results_df.index, y=results_df[model_name],
                                     mode='lines', name=model_name, line=dict(dash='dot')))
        fig.update_layout(title=f'Comparativo de Modelos para "{target_column}"',
                          xaxis_title='Tempo', yaxis_title='Valor da Métrica',
                          legend_title='Legenda', template='plotly_dark')
        st.success("Gráfico comparativo gerado com sucesso!")
        st.plotly_chart(fig, use_container_width=True)

    # --- SEÇÃO DE MODELAGEM UNIFICADA COM ABAS (TABS) ---
    def _predictive_modeling_section(self):
        st.header("🤖 Análise Preditiva")
        
        tab1, tab2 = st.tabs(["📊 Comparação de Modelos", "🧠 Treino e Diagnóstico Individual"])

        with tab1:
            st.markdown("#### Compare a performance de múltiplos modelos ao longo do tempo")
            model_options_comp = ['random_forest', 'lightgbm', 'xgboost', 'linear', 'ridge', 'lasso']
            selected_models = st.multiselect("Selecione os modelos para comparar:", model_options_comp, default=['lightgbm', 'xgboost'])
            target_column_comp = st.selectbox("Selecione a Métrica Alvo para Comparação", ['NAMESPACE_CPU_USAGE', 'NAMESPACE_MEMORY_USAGE'])
            sample_size_comp = st.slider("Tamanho da amostra para busca de dados", 1000, 10000, 2000, 500)
            if st.button("Gerar Gráfico Comparativo"):
                self._plot_forecast_comparison(selected_models, target_column_comp, sample_size_comp)

        with tab2:
            st.markdown("#### Treine um único modelo e analise seus resultados em detalhe")
            potential_targets = ['NAMESPACE_CPU_USAGE', 'NAMESPACE_MEMORY_USAGE', 'NAMESPACE_POD_COUNT']
            target_column = st.selectbox("Selecione a Métrica Alvo", potential_targets, key="individual_target")
            
            model_options_ind = ['random_forest', 'lightgbm', 'xgboost', 'linear']
            model_type = st.selectbox("Tipo de Modelo:", model_options_ind, key="individual_model")
            sample_size = st.slider("Tamanho da amostra para treino", 2000, 50000, 10000, 1000, key="individual_sample")

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Treinar Modelo Individual"):
                    self._train_model(target_column, model_type, sample_size)
            with col2:
                if st.button("Plotar Performance Básica"):
                    self._plot_model_performance()
            with col3:
                if st.button("Plotar Diagnósticos Avançados"):
                    self._plot_advanced_diagnostics()


    def _train_model(self, target_column, model_type, sample_size):
        data_df = get_pivoted_dataframe_for_training(st.session_state.data_loader, self.namespace_selection, sample_size)
        
        if data_df.empty:
            st.error("Não foi possível gerar um DataFrame de features.")
            return

        st.write("Amostra do DataFrame de Features (Após PIVOT):")
        st.dataframe(data_df.head())

        if target_column not in data_df.columns:
            st.error(f"Erro: A métrica alvo '{target_column}' não foi encontrada nos dados para o namespace '{self.namespace_selection}'.")
            return

        try:
            model = PredictiveModel(model_type)
            X, y = model.preprocess(data_df.reset_index(), target_column)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            with st.spinner("Treinando modelo..."):
                model.train(X_train, y_train)
                model.evaluate(X_test, y_test)
                
            st.session_state.trained_model = model
            
            st.success(f"Modelo para '{target_column}' treinado com sucesso!")
            st.write("Métricas de Performance:")
            st.json(model.metrics)
            
        except Exception as e:
            st.error(f"Erro ao treinar o modelo: {e}")
            st.exception(e)

    def _plot_model_performance(self):
        model = st.session_state.get('trained_model')
        if not model or not hasattr(model, 'y_test') or model.y_test is None:
            st.error("Nenhum modelo treinado disponível. Treine um modelo primeiro.")
            return
        try:
            st.subheader("Análise Gráfica da Performance Básica")
            fig, ax = plt.subplots(1, 2, figsize=(16, 6))
            sns.scatterplot(x=model.y_test, y=model.y_pred, alpha=0.5, ax=ax[0])
            ax[0].plot([model.y_test.min(), model.y_test.max()], [model.y_test.min(), model.y_test.max()], 'r--', lw=2)
            ax[0].set_xlabel("Valores Reais")
            ax[0].set_ylabel("Valores Preditos")
            ax[0].set_title("Reais vs. Preditos")
            residuals = model.y_test - model.y_pred
            sns.histplot(residuals, kde=True, ax=ax[1])
            ax[1].set_xlabel("Resíduos (Erro = Real - Predito)")
            ax[1].set_ylabel("Frequência")
            ax[1].set_title("Distribuição dos Erros do Modelo")
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Falha ao gerar gráficos: {e}")

    def _plot_model_performance(self):
        model = st.session_state.get('trained_model')
        if not model or not hasattr(model, 'y_test') or model.y_test is None:
            st.error("Nenhum modelo treinado disponível. Treine um modelo primeiro.")
            return
        try:
            st.subheader("Análise Gráfica da Performance Básica")
            fig, ax = plt.subplots(1, 2, figsize=(16, 6))
            sns.scatterplot(x=model.y_test, y=model.y_pred, alpha=0.5, ax=ax[0])
            ax[0].plot([model.y_test.min(), model.y_test.max()], [model.y_test.min(), model.y_test.max()], 'r--', lw=2)
            ax[0].set_xlabel("Valores Reais")
            ax[0].set_ylabel("Valores Preditos")
            ax[0].set_title("Reais vs. Preditos")
            residuals = model.y_test - model.y_pred
            sns.histplot(residuals, kde=True, ax=ax[1])
            ax[1].set_xlabel("Resíduos (Erro = Real - Predito)")
            ax[1].set_ylabel("Frequência")
            ax[1].set_title("Distribuição dos Erros do Modelo")
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Falha ao gerar gráficos: {e}")
            
    # --- NOVA FUNÇÃO PARA PLOTS AVANÇADOS ---
    def _plot_advanced_diagnostics(self):
        """Plota gráficos de interpretabilidade do modelo, como Feature Importance e SHAP."""
        model_wrapper = st.session_state.get('trained_model')
        
        if not model_wrapper or not hasattr(model_wrapper, 'model'):
            st.error("Nenhum modelo treinado disponível. Treine um modelo primeiro.")
            return
            
        model = model_wrapper.model
        model_type = model_wrapper.model_type
        
        if model_type in ['linear', 'ridge', 'lasso', 'elasticnet']:
            st.warning(f"Diagnósticos avançados (SHAP) não são ideais para o modelo '{model_type}'.")
            st.info("O gráfico de importância de features abaixo mostra os coeficientes do modelo.")
        
        st.subheader(f"Diagnósticos Avançados para o Modelo: '{model_type.replace('_', ' ').title()}'")
        
        try:
            # 1. Gráfico de Importância das Features
            if hasattr(model, 'feature_importances_'):
                st.markdown("#### Importância das Features")
                st.write("Este gráfico mostra quais 'pistas' o modelo considerou mais importantes.")
                feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, model_wrapper.X_train.columns)), columns=['Valor','Feature'])
                fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                sns.barplot(x="Valor", y="Feature", data=feature_imp.sort_values(by="Valor", ascending=False), ax=ax_imp)
                plt.tight_layout()
                st.pyplot(fig_imp)
            elif hasattr(model, 'coef_'):
                 st.markdown("#### Coeficientes do Modelo Linear")
                 st.write("Para modelos lineares, a magnitude dos coeficientes indica a importância da feature.")
                 coef_df = pd.DataFrame(model.coef_, index=model_wrapper.X_train.columns, columns=['Coeficiente'])
                 st.dataframe(coef_df.sort_values(by='Coeficiente', ascending=False))
            
            # 2. Gráfico SHAP (para modelos baseados em árvore)
            if model_type in ['random_forest', 'xgboost', 'lightgbm']:
                st.markdown("---")
                st.markdown("#### Análise SHAP (SHapley Additive exPlanations)")
                st.write("""
                Este gráfico é mais poderoso. Ele mostra não apenas a importância de cada feature, 
                mas também *como* o valor da feature impacta a previsão.
                - **Eixo Y:** Features, da mais importante (topo) para a menos importante.
                - **Eixo X:** Valor SHAP (o impacto na previsão). Valores > 0 aumentam a previsão, < 0 diminuem.
                - **Cor:** O valor da feature. Vermelho = alto, Azul = baixo.
                """)
                with st.spinner("Calculando valores SHAP... Isso pode levar um momento."):
                    # Usamos o TreeExplainer para modelos baseados em árvore
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(model_wrapper.X_train)
                    
                    fig_shap, ax_shap = plt.subplots()
                    shap.summary_plot(shap_values, model_wrapper.X_train, plot_type="dot", show=False)
                    st.pyplot(fig_shap)

        except Exception as e:
            st.error(f"Falha ao gerar diagnósticos avançados: {e}")
            st.exception(e)

    def run(self):
        if not st.session_state.data_loaded:
            st.info("⬅️ Configure o diretório e clique em 'Carregar Dados' para começar.")
            return
        
        if self.show_data:
            st.subheader("Amostra dos Dados Brutos (50 linhas)")
            with st.spinner("Buscando amostra..."):
                sample_df = st.session_state.data_loader.get_sample()
                if sample_df.empty:
                    st.warning("A amostra de dados está vazia.")
                else:
                    st.dataframe(sample_df)

        self._predictive_modeling_section()

if __name__ == "__main__":
    app = App("Vertis Data Consultant", layout="wide")
    app.run()