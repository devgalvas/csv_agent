import pandas as pd
import plotly.graph_objects as go

def calculate_anomaly_bands(df, metric_column, window=30, std_dev=2):
    """
    Calcula a média móvel, o desvio padrão móvel e as bandas de anomalia.
    
    Args:
        df (pd.DataFrame): DataFrame com uma coluna de métrica.
        metric_column (str): O nome da coluna da métrica.
        window (int): A janela para o cálculo móvel.
        std_dev (int): O número de desvios padrão para definir as bandas.
        
    Returns:
        pd.DataFrame: DataFrame original com as novas colunas de análise.
    """
    df['rolling_mean'] = df[metric_column].rolling(window=window, min_periods=1).mean()
    df['rolling_std'] = df[metric_column].rolling(window=window, min_periods=1).std()
    
    df['upper_band'] = df['rolling_mean'] + (df['rolling_std'] * std_dev)
    df['lower_band'] = df['rolling_mean'] - (df['rolling_std'] * std_dev)
    
    # Preenche NaNs no início/fim das bandas para um gráfico limpo
    df['upper_band'].bfill(inplace=True)
    df['upper_band'].ffill(inplace=True)
    df['lower_band'].bfill(inplace=True)
    df['lower_band'].ffill(inplace=True)
    
    df['anomaly'] = (df[metric_column] > df['upper_band']) | (df[metric_column] < df['lower_band'])
    return df

def plot_anomaly_detection_graph(df_analysis, metric_column):
    """
    Cria um gráfico interativo do Plotly para visualização de anomalias.
    """
    y_axis_title = metric_column.replace('_', ' ').title()
    if 'MEMORY' in metric_column:
        # Converte para GB para melhor visualização
        df_analysis[metric_column] = df_analysis[metric_column] / 1e9
        df_analysis['upper_band'] = df_analysis['upper_band'] / 1e9
        df_analysis['lower_band'] = df_analysis['lower_band'] / 1e9
        y_axis_title = "Uso de Memória (GB)"

    fig = go.Figure()

    # Adiciona as bandas de normalidade
    fig.add_trace(go.Scatter(
        x=df_analysis.index, y=df_analysis['upper_band'], mode='lines',
        line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=df_analysis.index, y=df_analysis['lower_band'], mode='lines',
        line=dict(width=0), name='Banda de Normalidade',
        fill='tonexty', fillcolor='rgba(68, 68, 68, 0.2)'
    ))

    # Linha principal da métrica
    fig.add_trace(go.Scatter(
        x=df_analysis.index, y=df_analysis[metric_column],
        mode='lines', name='Valor Real da Métrica', line=dict(color='royalblue', width=2)
    ))

    # Destaca os pontos de anomalia
    anomalies = df_analysis[df_analysis['anomaly']]
    fig.add_trace(go.Scatter(
        x=anomalies.index, y=anomalies[metric_column],
        mode='markers', name='Anomalia Detectada',
        marker=dict(color='red', size=8, symbol='x')
    ))

    fig.update_layout(
        title=f'Detecção de Anomalias para "{metric_column}"',
        xaxis_title='Data e Hora',
        yaxis_title=y_axis_title,
        legend_title='Legenda',
        template='plotly_dark'
    )
    return fig