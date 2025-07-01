import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from agent.csv_agent import CSVAgent
from agent.predictive_model import PredictiveModel
from agent.data_loader import DataLoader
from sklearn.metrics import confusion_matrix
import streamlit as st
import duckdb
import time

class App:
    def __init__(self, title: str, **page_config):
        """Initialize the Streamlit application with styling and configuration."""
        # Set up page configuration
        st.set_page_config(page_title=title, **page_config)
        
        # Apply custom styling
        self._apply_custom_styling()
        
        # Set up header layout
        self._create_header(title)
        
        # Initialize environment and API key
        load_dotenv()
        self.api_key = os.getenv("API_KEY")
        
        # Initialize sidebar
        self._create_sidebar()
        
        # Initialize class variables
        self.agent = CSVAgent(api_key=self.api_key)
        self.data_loader = None
        self.predictive_model = None
        self.df = None
        self.columns = []
        self.file_path = None
        
    def _apply_custom_styling(self):
        """Apply custom CSS styling to the application."""
        st.markdown("""
        <style>
        body, .main {
            background-color: #0a192f !important;
            color: #e3f0fc !important;
        }
        .stApp {
            background: linear-gradient(135deg, #0a192f 0%, #112240 100%) !important;
            color: #e3f0fc !important;
        }
        .stButton>button {
            background-color: #1565c0;
            color: #e3f0fc;
            border-radius: 8px;
            font-weight: bold;
            border: 1px solid #64ffda;
        }
        .stFileUploader {
            margin-bottom: 20px;
        }
        .stTextInput>div>div>input {
            background-color: #112240;
            color: #e3f0fc;
            border: 1px solid #1565c0;
        }
        .st-expanderHeader {
            color: #64ffda !important;
            font-weight: bold;
        }
        .stDataFrame {
            background-color: #112240;
            color: #e3f0fc;
        }
        .css-1v0mbdj, .css-1d391kg {
            background: #112240 !important;
            color: #e3f0fc !important;
        }
        .stMarkdown, .stCaption, .stInfo, .stSuccess, .stWarning, .stError {
            color: #e3f0fc !important;
        }
        footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)
        
    def _create_header(self, title):
        """Create header with logo and title."""
        col1, col2 = st.columns([1, 5])
        with col1:
            st.image("/home/lucas/UNIFEI/Vertis/vertis_research_agent/image/chatbot.webp", width=90)
        with col2:
            st.markdown(f'<h1 style="color:#64ffda;">📊 {title}</h1>', unsafe_allow_html=True)
            st.caption('<span style="color:#e3f0fc;">🚀 A Streamlit chatbot powered by Groq to consult .CSV files</span>', 
                      unsafe_allow_html=True)

    def _create_sidebar(self):
        """Create sidebar with upload and options."""
        with st.sidebar:
            st.header("Upload & Options")
            
            # File upload options
            self.uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
            
            # Data directory options
            self.namespace_dir = st.text_input("Parquet Directory", 
                                              "/home/lucas/UNIFEI/Vertis/vertis_research_agent/archive/partitioned_parquet")
            
            # Namespace selection if directory exists
            if self.namespace_dir and os.path.exists(self.namespace_dir):
                namespaces = ["All"] + [d for d in os.listdir(self.namespace_dir) 
                                       if os.path.isdir(os.path.join(self.namespace_dir, d))]
                self.namespace = st.selectbox("Select Namespace", namespaces)
            else:
                self.namespace = None
            
            self.show_data = st.checkbox("🔍 Show raw data")
            
            st.markdown("---")
            st.info("Made by [Lucas Galvão Freitas](https://github.com/devgalvas)")
    
    def _handle_file_upload(self):
        """Handle CSV file upload."""
        if self.uploaded_file:
            self.agent.load_csv(self.uploaded_file)
            st.success("✅ CSV file loaded successfully!")
            
            # Load DataFrame for DataLoader and PredictiveModel
            self.df = pd.read_csv(self.agent.csv_path)
            self.columns = self.df.columns.tolist()
            
            return True
        return False
            
    def _handle_parquet_loading(self):
        """Handle loading Parquet files from directory."""
        if not self.namespace_dir or not os.path.exists(self.namespace_dir):
            return False
            
        self.file_path = os.path.join(self.namespace_dir, "**", "*.parquet")
        
        files_found_by_glob = glob.glob(self.file_path, recursive=True)
        if not files_found_by_glob:
            st.warning("No Parquet files found in the specified directory.")
            st.info("Please check the directory path or upload a CSV file instead.")
            return False
        
        try:
            # O resto da função permanece igual
            file_size_mb = sum(os.path.getsize(f) for f in files_found_by_glob) / (1024 * 1024)
            st.info(f"Carregando {len(files_found_by_glob)} arquivos Parquet de '{self.namespace_dir}'... Tamanho total: {file_size_mb:.2f} MB")
            start_time = time.time()
            
            # O self.file_path agora contém o padrão glob, que será passado para o DataLoader
            self.data_loader = DataLoader(self.file_path, samples=50, query="SELECT 1", file_type="parquet")
            self.data_loader.connect_to_db()
            
            with st.spinner("Lendo o schema do Parquet..."):
                self.columns = self.data_loader.connect.execute("DESCRIBE logs").fetchdf()['column_name'].tolist()
                
            st.success(f"✅ Parquet carregado com sucesso em {time.time() - start_time:.2f} segundos! "
                     f"Colunas: {len(self.columns)}")
            return True
            
        except Exception as e:
            st.error(f"Falha ao processar Parquet: {str(e)}")
            return False


    def _show_data_sample(self):
        """Show a sample of the data."""
        if not self.show_data:
            return
            
        st.subheader("Raw Data (Sample)")
        with st.spinner("Fetching sample data..."):
            if self.uploaded_file:
                sample_df = self.df.head(100)
            else:
                sample_df = self.data_loader.get_sample()
                
            if sample_df.empty:
                st.warning("Sample data is empty.")
            else:
                st.dataframe(sample_df)
    
    def _qa_section(self):
        """Display Q&A section."""
        st.markdown('<p style="font-size:18px;color:#64ffda;font-weight:bold;">Ask a question about your data:</p>', 
                   unsafe_allow_html=True)
        question = st.text_input("Query", placeholder="e.g., Which rows have missing values?")
        
        if st.button("Ask"):
            if question.strip() == "":
                st.warning("Please type a question.")
            else:
                with st.spinner("Thinking..."):
                    answer = self.agent.ask(question)
                    st.markdown("**🧠 Answer:**")
                    if isinstance(answer, pd.DataFrame):
                        st.dataframe(answer)
                    else:
                        st.write(answer)
    
    def _predictive_modeling_section(self):
        """Display predictive modeling section."""
        with st.expander("🤖 Predictive Modeling"):
            st.markdown("Train a model on your data and make predictions")
            
            # Model parameters
            target_column = st.selectbox("Select a target column", self.columns)
            model_type = st.selectbox("Model Type:", ["linear", "random_forest", "prophet"])
            
            # Segmentation options (for parquet data)
            if self.data_loader:
                segment_column = st.selectbox("Segment by:", ["None", "ocnr_tx_namespace"])
                segment_values = ["None"] if segment_column == "None" else self.data_loader.connect.execute(
                    f"SELECT DISTINCT {segment_column} FROM logs LIMIT 100").fetchdf()[segment_column].tolist()
                segment_value = st.selectbox("Segment value:", segment_values)
                
                model_path = f"model_{target_column}_{self.namespace}_{segment_value}.joblib" \
                    if self.namespace != "All" and segment_column != "None" \
                    else f"model_{target_column}_{self.namespace}.joblib" \
                    if self.namespace != "All" else f"model_{target_column}.joblib"
            else:
                segment_column = "None"
                segment_value = "None"
                model_path = f"model_{target_column}.joblib"
            
            # Train model button
            if st.button("Train Model"):
                self._train_model(target_column, model_type, model_path, segment_column, segment_value)
            
            # Plot model performance button
            if st.button("Plot Model Performance"):
                self._plot_model_performance(model_type)
    
    def _train_model(self, target_column, model_type, model_path, segment_column, segment_value):
        """Train a predictive model."""
        try:
            # Initialize model
            self.predictive_model = PredictiveModel(model_path, model_type)
            
            # Get data for training
            if self.uploaded_file:
                train_df = self.df
            else:
                # For parquet data
                query = f"SELECT * FROM logs"
                if segment_column != "None" and segment_value != "None":
                    query += f" WHERE {segment_column} = '{segment_value}'"
                query += " LIMIT 10000"  # Limit rows for performance
                
                with st.spinner("Fetching data for training..."):
                    train_df = self.data_loader.connect.execute(query).fetchdf()
                
            # Train model
            with st.spinner("Training model... this may take a while"):
                if segment_column != "None" and segment_value != "None":
                    self.predictive_model.train(train_df, target_column, segment_column, segment_value)
                else:
                    self.predictive_model.train(train_df, target_column)
                
            # Show metrics
            st.success("Model trained successfully!")
            st.write("Metrics:", self.predictive_model.metrics)
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
    
    def _plot_model_performance(self, model_type):
        """Plot model performance metrics."""
        if not hasattr(self, 'predictive_model') or self.predictive_model is None:
            st.error("No model available. Please train a model first.")
            return
        
        try:
            with st.spinner("Generating model performance plots..."):
                if model_type == "random_forest":
                    # Classification metrics
                    if hasattr(self.predictive_model, 'y_test') and hasattr(self.predictive_model, 'y_pred'):
                        # Confusion Matrix
                        st.subheader("Confusion Matrix")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        cm = confusion_matrix(self.predictive_model.y_test, self.predictive_model.y_pred)
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("True")
                        st.pyplot(fig)
                        
                        # Performance metrics bar chart
                        st.subheader("Performance Metrics")
                        metrics_df = pd.DataFrame({
                            'Metric': list(self.predictive_model.metrics.keys()),
                            'Value': list(self.predictive_model.metrics.values())
                        })
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='Metric', y='Value', data=metrics_df, ax=ax)
                        ax.set_ylim(0, 1)  # Metrics are typically between 0 and 1
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                elif model_type == "linear":
                    # Regression metrics
                    if hasattr(self.predictive_model, 'y_test') and hasattr(self.predictive_model, 'y_pred'):
                        # True vs Predicted plot
                        st.subheader("True vs Predicted Values")
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.scatter(self.predictive_model.y_test, self.predictive_model.y_pred, alpha=0.5)
                        ax.plot([min(self.predictive_model.y_test), max(self.predictive_model.y_test)], 
                                [min(self.predictive_model.y_test), max(self.predictive_model.y_test)], 
                                'r--', lw=2)
                        ax.set_xlabel("True Values")
                        ax.set_ylabel("Predictions")
                        ax.set_title("True vs Predicted Values")
                        st.pyplot(fig)
                        
                        # Residuals plot
                        residuals = self.predictive_model.y_test - self.predictive_model.y_pred
                        st.subheader("Residuals Plot")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(self.predictive_model.y_pred, residuals, alpha=0.5)
                        ax.axhline(y=0, color='r', linestyle='--')
                        ax.set_xlabel("Predicted Values")
                        ax.set_ylabel("Residuals")
                        ax.set_title("Residuals vs Predicted Values")
                        st.pyplot(fig)
                        
                        # Performance metrics
                        st.subheader("Performance Metrics")
                        metrics_df = pd.DataFrame({
                            'Metric': list(self.predictive_model.metrics.keys()),
                            'Value': list(self.predictive_model.metrics.values())
                        })
                        st.table(metrics_df)
                else:
                    st.info(f"Plotting not yet implemented for model type: {model_type}")
            
        except Exception as e:
            st.error(f"Failed to generate plots: {str(e)}")

    def run(self):
        """Run the Streamlit application."""
        # Handle data sources
        data_loaded = False
        
        if self.uploaded_file:
            data_loaded = self._handle_file_upload()
        elif self.namespace_dir and os.path.exists(self.namespace_dir):
            data_loaded = self._handle_parquet_loading()
        else:
            st.info("⬅️ Please upload a CSV file or specify a Parquet directory to get started.")
            
        # If data is loaded, show sections
        if data_loaded:
            self._show_data_sample()
            self._qa_section()
            self._predictive_modeling_section()

if __name__ == "__main__":
    app = App("Vertis Data Consultant", layout="wide")
    app.run()