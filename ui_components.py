import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# --- Importações Modificadas ---
from agent.predictive_model import PredictiveModel
from config import MODEL_OPTIONS, TARGET_OPTIONS, METRICS_TO_PIVOT
from utils import get_namespaces_from_filesystem, get_pivoted_dataframe_for_training, get_pivoted_dataframe_for_testing, get_single_metric_data
from analysis import calculate_anomaly_bands, plot_anomaly_detection_graph # NOVA IMPORTAÇÃO
# --------------------------------

def apply_custom_styling():
    """Aplica o CSS customizado para a aplicação."""
    st.markdown("""<style>body, .main { background-color: #0a192f !important; color: #e3f0fc !important; } .stApp { background: linear-gradient(135deg, #0a192f 0%, #112240 100%) !important; color: #e3f0fc !important; } .stButton>button { background-color: #1565c0; color: #e3f0fc; border-radius: 8px; font-weight: bold; border: 1px solid #64ffda; } .stTextInput>div>div>input { background-color: #112240; color: #e3f0fc; border: 1px solid #1565c0; } .st-expanderHeader { color: #64ffda !important; font-weight: bold; } footer {visibility: hidden;}</style>""", unsafe_allow_html=True)

def create_header(title):
    """Cria o cabeçalho da aplicação."""
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown('<div style="width:90px; height:90px; background-color:#112240; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size: 2rem; color: #64ffda; font-weight: bold;">V</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<h1 style="color:#64ffda;">📊 {title}</h1>', unsafe_allow_html=True)
        st.caption('<span style="color:#e3f0fc;">🚀 Agente de IA para análise de dados do OpenShift</span>', unsafe_allow_html=True)

def create_sidebar(default_path):
    """Cria a barra lateral e retorna as seleções do usuário."""
    with st.sidebar:
        st.header("Configurações")
        namespace_dir = st.text_input("Diretório Parquet", default_path)
        namespaces = get_namespaces_from_filesystem(namespace_dir)
        namespace_selection = st.selectbox("Selecione o Namespace", ["All"] + namespaces)
        show_data = st.checkbox("🔍 Mostrar amostra dos dados")
        st.markdown("---")
        st.info("Feito por [Lucas Galvão Freitas](https://github.com/devgalvas)")
        return namespace_dir, namespace_selection, show_data

# --- NOVA SEÇÃO DE UI ---
def create_anomaly_detection_section(data_loader, namespace_selection):
    """Cria a seção de análise de estabilidade na UI."""
    st.header("🛡️ Análise de Estabilidade (Detecção de Anomalias)")
    st.markdown("Identifique comportamentos atípicos em suas métricas usando média móvel e desvio padrão.")
    
    col1, col2 = st.columns(2)
    with col1:
        anomaly_metric = st.selectbox(
            "Selecione a Métrica para Análise:",
            options=METRICS_TO_PIVOT,
            key='anomaly_metric'
        )
        history_limit = st.slider(
            "Nº de registros recentes para analisar:",
            min_value=500, max_value=10000, value=2000, step=250,
            key='anomaly_limit'
        )
    with col2:
        window = st.slider("Janela da Média Móvel (dias):", 1, 90, 30, key='anomaly_window')
        std_dev = st.slider("Nº de Desvios Padrão para Anomalia:", 1.0, 3.0, 2.0, 0.5, key='anomaly_std')

    if st.button("🔍 Analisar Estabilidade", use_container_width=True):
        metric_df = get_single_metric_data(data_loader, namespace_selection, anomaly_metric, history_limit)
        if not metric_df.empty:
            analysis_df = calculate_anomaly_bands(metric_df, anomaly_metric, window, std_dev)
            fig = plot_anomaly_detection_graph(analysis_df, anomaly_metric)
            st.plotly_chart(fig, use_container_width=True)
            
            num_anomalies = analysis_df['anomaly'].sum()
            st.metric("Anomalias Detectadas", f"{num_anomalies} pontos")
        else:
            st.warning("Não foram encontrados dados para a métrica selecionada.")

def create_predictive_modeling_section(data_loader, namespace_selection):
    """Cria a seção principal de modelagem na UI."""
    st.header("🤖 Bancada de Testes de Modelos (Model Workbench)")
    
    st.markdown("#### 1. Configure o Experimento")
    
    col1, col2 = st.columns(2)
    with col1:
        target_column = st.selectbox("Selecione a Métrica Alvo:", TARGET_OPTIONS)
        training_sample_size = st.slider(
            "Tamanho da amostra de treino (aleatória do histórico):", 
            min_value=5000, max_value=50000, value=20000, step=1000,
            help="Número de registros aleatórios de todo o histórico para ensinar o modelo. Mais dados podem gerar modelos melhores, mas demoram mais para treinar."
        )
    with col2:
        resample_freq_map = {'Dados Detalhados': None, 'Média Diária': 'D', 'Média Horária': 'H'}
        resample_choice = st.selectbox("Visualizar gráfico por:", resample_freq_map.keys())
        selected_freq = resample_freq_map[resample_choice]
        test_limit = st.slider(
            "Nº de registros recentes para teste/visualização:", 
            min_value=500, max_value=5000, value=1000, step=100,
            help="Número de pontos de dados mais recentes a serem usados para testar o modelo e plotar no gráfico."
        )

    st.markdown("#### 2. Selecione os Modelos para Treinar e Comparar")
    selected_models = st.multiselect("Modelos:", MODEL_OPTIONS, default=['xgboost', 'lightgbm'])

    if st.button("🚀 Executar Experimento", use_container_width=True):
        run_experiment(data_loader, namespace_selection, selected_models, target_column, selected_freq, training_sample_size, test_limit)

def run_experiment(data_loader, namespace, model_types, target_column, resample_freq, training_size, test_limit):
    """Orquestra o fluxo de treinamento, avaliação e visualização."""
    # 1. Obter dados de TREINO (amostra aleatória e rica)
    training_df = get_pivoted_dataframe_for_training(data_loader, namespace, METRICS_TO_PIVOT, sample_size=training_size)
    if training_df.empty or target_column not in training_df.columns:
        st.error(f"Não foi possível obter dados de treino suficientes para a métrica '{target_column}'.")
        return

    # 2. Obter dados de TESTE (os mais recentes)
    testing_df = get_pivoted_dataframe_for_testing(data_loader, namespace, METRICS_TO_PIVOT, limit=test_limit)
    if testing_df.empty or target_column not in testing_df.columns:
        st.error(f"Não foi possível obter dados de teste recentes para a métrica '{target_column}'.")
        return
        
    # 3. Pré-processar ambos os datasets
    temp_model = PredictiveModel('linear')
    X_train, y_train = temp_model.preprocess(training_df, target_column)
    X_test, y_test = temp_model.preprocess(testing_df, target_column)

    # 4. Treinar modelos e coletar resultados
    trained_models = {}
    
    with st.spinner("Treinando modelos com dados históricos e avaliando no período recente..."):
        for model_type in model_types:
            try:
                model = PredictiveModel(model_type)
                model.train(X_train, y_train)
                model.evaluate(X_test, y_test)
                trained_models[model_type] = model
            except Exception as e:
                st.warning(f"Falha ao treinar o modelo '{model_type}': {e}")
    
    if not trained_models:
        st.error("Nenhum modelo foi treinado com sucesso.")
        return

    # 5. Exibir os resultados
    st.markdown("---")
    st.header("📈 Resultados do Experimento")
    
    display_metrics_table(trained_models)
    plot_forecast_comparison(y_test, trained_models, target_column, resample_freq)
    display_individual_diagnostics(trained_models)

def display_metrics_table(trained_models):
    """Exibe a tabela comparativa de métricas."""
    st.markdown("#### Tabela Comparativa de Métricas")
    metrics_data = {
        "Modelo": list(trained_models.keys()),
        "R² Score": [model.metrics['R2'] for model in trained_models.values()],
        "MSE": [model.metrics['MSE'] for model in trained_models.values()]
    }
    metrics_df = pd.DataFrame(metrics_data).sort_values("R² Score", ascending=False).reset_index(drop=True)
    st.dataframe(metrics_df, use_container_width=True)

def plot_forecast_comparison(y_test, trained_models, target_column, resample_freq):
    """Plota o gráfico comparativo ao longo do tempo."""
    st.markdown("#### Gráfico Comparativo ao Longo do Tempo")

    results_df = pd.DataFrame({'Valores Reais': y_test})
    for name, model in trained_models.items():
        results_df[f'Previsão_{name}'] = model.y_pred

    plot_df = results_df
    view_mode = "Dados Detalhados"
    if resample_freq:
        view_mode = f"Média {resample_freq}"
        with st.spinner(f"Reamostrando resultados para visualização..."):
            plot_df = results_df.resample(resample_freq).mean().ffill()
    
    y_axis_title = "Valor da Métrica"
    if target_column == 'NAMESPACE_MEMORY_USAGE':
        plot_df = plot_df / 1e9
        y_axis_title = "Uso de Memória (GB)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Valores Reais'],
                             mode='lines+markers', name='Valores Reais', line=dict(color='royalblue', width=2), marker=dict(size=4)))
    for col in plot_df.columns[1:]:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[col],
                                 mode='lines+markers', name=col, line=dict(dash='dot', width=2), marker=dict(size=4)))
    fig.update_layout(title=f'Comparativo de Modelos para "{target_column}" ({view_mode})',
                      xaxis_title='Data e Hora', yaxis_title=y_axis_title,
                      legend_title='Legenda', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

def display_individual_diagnostics(trained_models):
    """Exibe os expanders com diagnósticos individuais para cada modelo."""
    st.markdown("#### Análise Individual por Modelo")
    for name, model in trained_models.items():
        with st.expander(f"🔍 Diagnósticos para o modelo: {name.replace('_', ' ').title()}"):
            # Gráfico de Performance Básica
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            sns.scatterplot(x=model.y_test, y=model.y_pred, alpha=0.6, ax=ax[0])
            ax[0].plot([model.y_test.min(), model.y_test.max()], [model.y_test.min(), model.y_test.max()], 'r--', lw=2)
            ax[0].set_title("Reais vs. Preditos")
            ax[0].set_xlabel("Valores Reais")
            ax[0].set_ylabel("Valores Preditos")
            
            sns.histplot(model.y_test - model.y_pred, kde=True, ax=ax[1])
            ax[1].set_title("Distribuição dos Erros")
            ax[1].set_xlabel("Erro (Real - Predito)")
            st.pyplot(fig)

            # Gráficos de Interpretabilidade
            if model.model_type in ['random_forest', 'xgboost', 'lightgbm']:
                with st.spinner(f"Calculando SHAP para {name}..."):
                    try:
                        explainer = shap.TreeExplainer(model.model)
                        shap_values = explainer.shap_values(model.X_train)
                        
                        st.markdown("##### Análise de Impacto das Features (SHAP)")
                        fig_shap, _ = plt.subplots()
                        shap.summary_plot(shap_values, model.X_train, plot_type="dot", show=False, plot_size=(10, 5))
                        st.pyplot(fig_shap, clear_figure=True)
                    except Exception as e:
                        st.warning(f"Não foi possível gerar o gráfico SHAP: {e}")
