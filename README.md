# Vertis Data Consultant 🤖📊

A modern, interactive data analysis platform built with **Streamlit** that provides advanced analytics for OpenShift monitoring data. Features predictive modeling, natural language querying, and automated data preprocessing using Dask for large-scale CSV processing.

---

## Features

### 🔍 Data Analysis & Querying
- **Upload CSV files** or load **Parquet datasets** (supports large files)
- **Ask questions in natural language** about your data using LLM integration
- **Automatic SQL generation** via Groq API for complex queries
- **DuckDB backend** for fast, efficient data processing

### 🤖 Machine Learning & Predictive Analytics
- **Predictive modeling** with Random Forest and Linear Regression
- **Feature engineering** with automatic PIVOT operations
- **Model performance visualization** with residuals and prediction plots
- **OpenShift metrics analysis** (CPU, Memory, Pod counts, Network usage)

### ⚡ Big Data Processing
- **Dask integration** for processing large CSV files (>GB scale)
- **Automatic partitioning** by namespace for optimized queries
- **Parquet conversion** for faster subsequent data access
- **Memory-efficient processing** with configurable block sizes

### 🎨 User Interface
- **Beautiful dark blue UI** with modern styling
- **Interactive sidebar** for configuration and data loading
- **Expandable sections** for organized content display
- **Real-time progress indicators** and error handling

---

## Repository Structure

```
vertis_research_agent/
├── app.py                 # Main Streamlit application
├── agent/                 # Core analysis modules
│   ├── __init__.py
│   ├── csv_agent.py       # Natural language to SQL conversion
│   ├── data_loader.py     # DuckDB data loading and querying
│   ├── predictive_model.py # ML models and training
│   └── preprocess.py      # Large-scale data preprocessing with Dask
├── archive/               # Data storage
│   ├── 00.csv            # Raw input data
│   └── partitioned_parquet/ # Processed parquet files by namespace
├── image/                 # UI assets
├── requirements.txt       # Python dependencies
├── .env                  # Environment configuration
└── README.md             # This file
```

---

## Screenshots

![screenshot](image/chatbot.webp)

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/vertis-data-consultant.git
cd vertis-data-consultant/vertis_research_agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API key (optional)

Create a `.env` file in the project root:

```env
API_KEY=your_groq_api_key_here
```

Or export it in your shell:

```bash
export API_KEY=your_groq_api_key_here
```

### 4. Data Preprocessing (Must for large CSV files, like the dumps)

If you have a large CSV file to process, first run the preprocessing script:

```bash
python agent/preprocess.py
```

This will:
- Start a Dask client for distributed processing
- Convert CSV to partitioned Parquet format
- Clean and transform the data
- Partition by `ocnr_tx_namespace` for optimized queries

### 5. Run the application

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

---

## Usage

### Basic Data Analysis
1. **Configure the data source** in the sidebar:
   - Set the Parquet directory path (default: `archive/partitioned_parquet`)
   - Select a specific namespace or "All" for complete dataset
2. **Load the data** by clicking "🚀 Carregar Dados e Analisar"
3. **Explore the data** using the "🔍 Mostrar amostra dos dados" option

### Predictive Modeling
1. **Expand the "🤖 Modelagem Preditiva" section**
2. **Select target metric**: Choose from CPU usage, Memory usage, or Pod count
3. **Choose model type**: Random Forest or Linear Regression
4. **Set sample size**: Adjust based on your computational resources
5. **Train the model** and view performance metrics
6. **Visualize results** with scatter plots and residual analysis

### Natural Language Querying (via CSV Agent)
The [`csv_agent.py`](agent/csv_agent.py) module provides LLM-powered natural language to SQL conversion:

```python
from agent.csv_agent import CSVAgent

agent = CSVAgent(api_key="your_groq_key")
agent.load_csv(your_file)
result = agent.ask("What are the top 5 namespaces by CPU usage?")
```

---

## Configuration

### Environment Variables
- `API_KEY`: Groq API key for LLM functionality

### Data Processing Parameters
The [`preprocess.py`](agent/preprocess.py) script can be configured for:
- **Encoding**: UTF-16 (default for OpenShift data)
- **Block size**: 64MB (adjustable for memory constraints)
- **Partitioning**: By namespace for query optimization

### Model Training Parameters
- **Sample size**: 2,000 - 50,000 records (configurable)
- **Target metrics**: CPU, Memory, Pod counts, Network usage
- **Model types**: Random Forest, Linear Regression

---

## Architecture

### Data Flow
1. **Raw CSV** → [`preprocess.py`](agent/preprocess.py) → **Partitioned Parquet**
2. **Parquet files** → [`data_loader.py`](agent/data_loader.py) → **DuckDB tables**
3. **DuckDB** → [`predictive_model.py`](agent/predictive_model.py) → **Trained ML models**
4. **User queries** → [`csv_agent.py`](agent/csv_agent.py) → **SQL results**

### Key Components
- **[`DataLoader`](agent/data_loader.py)**: Handles data loading and SQL execution with DuckDB
- **[`PredictiveModel`](agent/predictive_model.py)**: ML model training and evaluation
- **[`CSVAgent`](agent/csv_agent.py)**: Natural language processing for data queries
- **[`App`](app.py)**: Main Streamlit interface orchestrating all components

---

## Technologies Used

- **[Streamlit](https://streamlit.io/)** - Web application framework
- **[DuckDB](https://duckdb.org/)** - In-process analytical database
- **[Dask](https://dask.org/)** - Parallel computing and large data processing
- **[Groq API](https://console.groq.com/)** - Large Language Model integration
- **[scikit-learn](https://scikit-learn.org/)** - Machine learning algorithms
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation and analysis
- **[PyArrow](https://arrow.apache.org/docs/python/)** - Parquet file processing
- **[Seaborn/Matplotlib](https://seaborn.pydata.org/)** - Data visualization

---

## Performance Considerations

### Large Dataset Handling
- Use Dask preprocessing for files >1GB
- Parquet format provides 3-10x faster loading than CSV
- Namespace partitioning enables selective data loading
- Configurable sample sizes prevent memory overflow

### Query Optimization
- DuckDB provides columnar storage benefits
- Automatic query optimization and vectorization
- In-memory processing for maximum speed
- PIVOT operations for feature engineering

---

## Troubleshooting

### Common Issues
1. **Memory errors during processing**: Reduce block size in [`preprocess.py`](agent/preprocess.py)
2. **API connection failures**: Verify Groq API key in `.env` file
3. **Empty results**: Check namespace selection and data availability
4. **Model training errors**: Ensure sufficient data and valid target columns

### Debug Mode
Enable detailed logging by running:
```bash
python agent/preprocess.py
```
Check the Dask dashboard link in the console output for monitoring.

---

## Credits

Made with ❤️ by [Lucas Galvão Freitas](https://github.com/devgalvas)

---

## License

MIT License