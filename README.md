# Vertis Data Consultant 🤖📊

Plataforma em Streamlit para análise de dados de monitoramento do OpenShift. Foca em:
- Ingestão eficiente de dados com DuckDB (Parquet/CSV)
- Pré-processamento massivo com Dask (CSV → Parquet particionado)
- Modelagem preditiva com scikit-learn, XGBoost e LightGBM
- Diagnósticos e interpretabilidade com SHAP
- UI interativa para exploração e treino de modelos

Este README detalha a estrutura do repositório, como configurar o ambiente, preparar os dados e executar a aplicação.

---

## Sumário

- Visão geral da arquitetura
- Estrutura do repositório
- Pré-requisitos
- Configuração do ambiente
- Pipeline de dados (CSV → Parquet)
- Executando a aplicação (Streamlit)
- Uso: análise e modelagem
- Diagnósticos avançados (Feature Importance e SHAP)
- Ajustes de desempenho
- Solução de problemas
- Referências de arquivos
- Licença

---

## Visão geral da arquitetura

Fluxo principal:
1) CSV bruto → [agent/preprocess.py](agent/preprocess.py) → Parquet particionado por namespace
2) Parquets → [`agent.data_loader.DataLoader`](agent/data_loader.py) → Tabela logs (DuckDB)
3) Tabela logs → PIVOT e features → [`agent.predictive_model.PredictiveModel`](agent/predictive_model.py)
4) Interface Streamlit → [app.py](app.py)

Componentes-chave:
- Ingestão/consulta: [`agent.data_loader.DataLoader`](agent/data_loader.py)
- Modelagem: [`agent.predictive_model.PredictiveModel`](agent/predictive_model.py)
- UI Streamlit principal: [app.py](app.py)
- Configurações (paths, modelos, alvos): [config.py](config.py)
- Utilitários/diagnósticos: [analysis.py](analysis.py), [utils.py](utils.py)

Obs.: Integração LLM/Ollama removida do fluxo (não utilizada).

---

## Estrutura do repositório

```
vertis_research_agent/
├── app.py                      # Aplicação Streamlit principal
├── analysis.py                 # Funções auxiliares de análise (opcional)
├── ui_components.py            # Componentes/modos de UI (opcional)
├── utils.py                    # Utilitários
├── config.py                   # Opções de UI, métricas e caminhos padrão
├── check_duckdb.py             # (Opcional) verificação/diagnóstico do DuckDB
├── requirements.txt            # Dependências Python
├── .env                        # Variáveis de ambiente (opcional; não requer API)
├── agent/
│   ├── __init__.py
│   ├── data_loader.py          # DuckDB (CSV/Parquet → tabela logs)
│   ├── predictive_model.py     # Pré-processamento + treino/avaliação
│   └── preprocess.py           # Dask (CSV grande → Parquet particionado)
└── archive/
    ├── 00.csv                  # CSV bruto (exemplo)
    └── partitioned_parquet/    # Saída Parquet particionada por ocnr_tx_namespace
```

---

## Pré-requisitos

- Python 3.10+
- Linux: build-essential (para XGBoost/LightGBM), libomp (para LightGBM em algumas distros)
  - Debian/Ubuntu: sudo apt-get update && sudo apt-get install -y build-essential libgomp1
- Parquet: pyarrow (já em requirements)
- Recomendado: virtualenv/venv

Para diagnósticos avançados:
- SHAP: pip install shap

---

## Configuração do ambiente

1) Clonar o repositório
```bash
git clone https://github.com/devgalvas/csv_agent.git
cd vertis-data-consultant/vertis_research_agent
```

2) (Recomendado) Criar ambiente virtual
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows (PowerShell)
```

3) Instalar dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
# Para os diagnósticos avançados (se ainda não incluso em requirements):
pip install shap
```

4) Variáveis de ambiente
- Não é necessário configurar API keys. O arquivo `.env` é opcional.

---

## Pipeline de dados (CSV → Parquet)

Use o Dask para converter grandes CSVs para Parquet particionado por namespace:
```bash
python agent/preprocess.py
```

O script:
- Lê o CSV de entrada (padrão: archive/00.csv)
- Normaliza tipos, remove nulos críticos
- Salva particionado por `ocnr_tx_namespace` em `archive/partitioned_parquet`

Personalize parâmetros (encoding, blocksize, colunas) diretamente em [agent/preprocess.py](agent/preprocess.py).

---

## Executando a aplicação (Streamlit)

1) Garanta que existem Parquets em `archive/partitioned_parquet/`
2) Rode a aplicação:
```bash
streamlit run app.py
```
3) No sidebar:
- Informe o diretório Parquet (padrão em [config.py](config.py): `PARQUET_DEFAULT_PATH`)
- Selecione um Namespace (ou All)
- Clique em “🚀 Carregar Dados e Analisar”

A aplicação criará a tabela `logs` no DuckDB usando [`DataLoader`](agent/data_loader.py).

---

## Uso: análise e modelagem

A UI de [app.py](app.py) disponibiliza:
- Carregamento e cache:
  - `load_data_and_get_schema`: conecta ao DuckDB e obtém schema
  - `get_namespaces_from_filesystem`: lista namespaces a partir do FS
  - `get_pivoted_dataframe_for_training`: executa PIVOT e retorna DataFrame “wide” com métricas por timestamp
- “🤖 Modelagem Preditiva com Engenharia de Features”:
  - Métrica-alvo (ex.: `NAMESPACE_CPU_USAGE`, `NAMESPACE_MEMORY_USAGE`, `NAMESPACE_POD_COUNT`)
  - Tipo de modelo (ver [config.py](config.py): `MODEL_OPTIONS`)
  - Tamanho de amostra para busca de dados
  - Botões:
    - “Treinar Modelo”
    - “Plotar Performance Básica”
    - “Plotar Diagnósticos Avançados”

Pipeline interno de modelagem ([predictive_model.py](agent/predictive_model.py)):
- Ordenação temporal por `ocnr_dt_date`
- Conversão de alvo para numérico
- Lags: 1, 3, 5, 7
- Janelas rolantes (mean/std, janela=5)
- Features de tempo: hora, dia da semana, dia do mês, mês
- Alinhamento X/y, limpeza de NaNs, índices temporais
- Treino com modelos lineares/árvores; métricas: MSE, R² (e RMSE na UI estendida)

---

## Diagnósticos avançados (Feature Importance e SHAP)

- Importância de features:
  - Modelos de árvore: `feature_importances_` (barplot ordenado)
  - Modelos lineares: coeficientes (tabela ordenada)
- SHAP (para `random_forest`, `xgboost`, `lightgbm`):
  - `TreeExplainer` + `summary_plot (dot)` para impacto local/global
  - Útil para entender como cada métrica influencia as previsões
- Observação: SHAP pode ser custoso em CPU/memória; reduza o tamanho de amostra se necessário.

---

## Ajustes de desempenho

- Pré-processamento:
  - Ajuste `blocksize` no Dask (p.ex. 64MB) para equilibrar memória/velocidade
  - Parquet + particionamento por `ocnr_tx_namespace` acelera filtros
- Consultas:
  - Use LIMIT e um conjunto enxuto de métricas no PIVOT
  - Prefira scans de Parquet a CSV em produção
- Modelos:
  - Ajuste `sample_size` no carregamento para controlar custo de treino
  - XGBoost/LightGBM: `n_jobs=-1` para paralelismo

---

## Solução de problemas

- “Diretório Parquet não existe”
  - Corrija o caminho no sidebar (ou ajuste `PARQUET_DEFAULT_PATH` em [config.py](config.py))
  - Gere os Parquets: `python agent/preprocess.py`
- “Falha ao executar PIVOT”
  - Verifique colunas: `ocnr_dt_date`, `ocnr_tx_namespace`, `ocnr_tx_query`, `ocnr_nm_result`
  - Cheque se há dados no namespace selecionado
- “R² negativo / erros altos”
  - Aumente a janela (mais linhas)
  - Revise a métrica-alvo e a sazonalidade
- “Erro com SHAP”
  - Instale `shap` e confirme compatibilidade da versão do XGBoost/LightGBM
  - Reduza a amostra para o cálculo SHAP

---

## Referências de arquivos

- App Streamlit: [app.py](app.py)
  - Cache/ingestão/PIVOT: `load_data_and_get_schema`, `get_namespaces_from_filesystem`, `get_pivoted_dataframe_for_training`
- Data loader (DuckDB): [`agent.data_loader.DataLoader`](agent/data_loader.py)
- Modelagem: [`agent.predictive_model.PredictiveModel`](agent/predictive_model.py)
- Configurações (paths/modelos/alvos): [config.py](config.py)
- Pipeline CSV → Parquet: [agent/preprocess.py](agent/preprocess.py)

---

## Licença