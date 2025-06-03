import os
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
        st.set_page_config(page_title=title, **page_config)

        # Custom CSS for a modern look
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
        .css-1v0mbdj, .css-1d391kg { /* Sidebar */
            background: #112240 !important;
            color: #e3f0fc !important;
        }
        .stMarkdown, .stCaption, .stInfo, .stSuccess, .stWarning, .stError {
            color: #e3f0fc !important;
        }
        footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

        # Layout: logo and title in columns
        col1, col2 = st.columns([1, 5])
        with col1:
            st.image("/home/lucas/UNIFEI/Vertis/vertis_research_agent/image/chatbot.webp", width=90)
        with col2:
            st.title(f"📊 {title}")
            st.caption("🚀 A Streamlit chatbot powered by Groq to consult .CSV files")

        load_dotenv()
        api_key = os.getenv("API_KEY")

        # Sidebar for file upload and options
        with st.sidebar:
            st.header("Upload & Options")
            self.uploaded_file = st.file_uploader("Upload a CSV file (supports up to 50GB)", type="csv")
            self.show_data = st.checkbox("🔍 Show raw data (sample of 50 rows)")
            st.markdown("---")
            st.info("Made by Lucas Galvão Freitas](https://github.com/devgalvas)")

        self.agent = CSVAgent(api_key=api_key)
        self.data_loader = None
        self.predictive_model = None
        self.csv_path = None
        self.columns = []

    def plot_top_namespaces(self, df, metric, top_n=5):
        df_sorted = df.sort_values(by=metric, ascending=False)
        top_over = df_sorted.head(top_n)
        top_under = df_sorted.tail(top_n)
        
        st.subheader(f"TOP {top_n} Namespaces por {metric}")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=metric, y='NAMESPACE', hue='NAMESPACE', data=pd.concat([top_over, top_under]), palette="coolwarm")
        ax.set_xlabel(metric)
        ax.set_ylabel("Namespace")
        st.pyplot(fig)

    def run(self):
        if self.uploaded_file:
            # Verificar tamanho do arquivo
            file_size_mb = self.uploaded_file.size / (1024 * 1024)  # Converter bytes para MB
            if file_size_mb > 50000:
                st.error("File size exceeds 50 GB limit. Please use a smaller file or process locally.")
                return

            st.info(f"Uploading file... Size: {file_size_mb:.2f} MB")
            start_time = time.time()

            try:
                # Salvar o arquivo temporariamente
                self.csv_path = f"temp_{self.uploaded_file.name}"
                with open(self.csv_path, "wb") as f:
                    f.write(self.uploaded_file.getbuffer())

                # Inicializar DataLoader sem consulta inicial pesada
                self.data_loader = DataLoader(self.csv_path, samples=50, query="SELECT 1")  # Consulta leve
                self.data_loader.connect_to_db()

                # Obter colunas do CSV usando DuckDB
                with st.spinner("Reading CSV schema..."):
                    self.columns = self.data_loader.connect.execute("DESCRIBE logs").fetchdf()['column_name'].tolist()
                st.success(f"✅ CSV loaded successfully in {time.time() - start_time:.2f} seconds! Columns: {len(self.columns)}")

                if self.show_data:
                    st.subheader("Raw Data (Sample)")
                    with st.spinner("Fetching sample data..."):
                        sample_df = self.data_loader.connect.execute("SELECT * FROM logs LIMIT 50").fetchdf()
                        if sample_df.empty:
                            st.warning("Sample data is empty.")
                        else:
                            st.dataframe(sample_df)

                st.markdown('<p style="font-size:18px;color:#2E86C1;">Ask a question about your data:</p>', unsafe_allow_html=True)
                question = st.text_input("", placeholder="e.g., Which rows have missing values?")

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

                with st.expander("🤖 Predictive Modeling"):
                    st.markdown("Train a model on your data and make predictions")
                    target_column = st.selectbox("Select a target column", self.columns)
                    model_type = st.selectbox("Model Type:", ["linear", "random_forest", "prophet"])
                    segment_column = st.selectbox("Segment by:", ["None", "NAMESPACE", "CLUSTER"])
                    segment_values = ["None"] if segment_column == "None" else self.data_loader.connect.execute(f"SELECT DISTINCT {segment_column} FROM logs LIMIT 100").fetchdf()[segment_column].tolist()
                    segment_value = st.selectbox("Segment value:", segment_values)
                    model_path = f"model_{target_column}_{segment_column}_{segment_value}.joblib" if segment_column != "None" else "model.joblib"

                    if st.button("Train Model"):
                        try:
                            self.predictive_model = PredictiveModel(model_path, model_type)
                            with st.spinner("Training ... this can take a while"):
                                if segment_column != "None":
                                    if segment_value == "None":
                                        st.error("Please select a valid segment value.")
                                        return
                                    query = f"SELECT * FROM logs WHERE {segment_column} = '{segment_value}'"
                                else:
                                    query = "SELECT * FROM logs LIMIT 100000"  # Limitar para evitar sobrecarga
                                df = self.data_loader.connect.execute(query).fetchdf()
                                if df.empty:
                                    st.error(f"No data found for {segment_column} = {segment_value}")
                                    return
                                self.predictive_model.train(df, target_column, segment_column, segment_value)
                                st.success("Model trained!")
                                st.write("Metrics: ", self.predictive_model.metrics)
                                
                                if model_type == "random_forest":
                                    st.subheader("Confusion Matrix")
                                    fig, ax = plt.subplots()
                                    cm = confusion_matrix(self.predictive_model.y_test, self.predictive_model.y_pred)
                                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                                    ax.set_xlabel("Predicted")
                                    ax.set_ylabel("True")
                                    st.pyplot(fig)
                                    st.subheader("Precision, Recall, F1, Accuracy")
                                    st.bar_chart(pd.DataFrame(self.predictive_model.metrics, index=["Score"]).T)
                                elif model_type == "linear":
                                    st.subheader("Predicted vs True Values")
                                    fig, ax = plt.subplots()
                                    ax.scatter(self.predictive_model.y_test, self.predictive_model.y_pred, color="skyblue")
                                    ax.plot(self.predictive_model.y_test, self.predictive_model.y_test, color="red", linestyle="--")
                                    ax.set_xlabel("True Values")
                                    ax.set_ylabel("Predicted Values")
                                    st.pyplot(fig)
                                elif model_type == "prophet":
                                    st.subheader("Forecast")
                                    st.dataframe(self.predictive_model.metrics["forecast"])
                        except ValueError as e:
                            st.error(f"Training failed: {str(e)}")
                        except Exception as e:
                            st.error(f"Unexpected error during training: {str(e)}")

                    if st.button("Load Model"):
                        try:
                            self.predictive_model = PredictiveModel(model_path, model_type)
                            self.predictive_model.load()
                            st.success("Model Loaded!")
                        except FileNotFoundError:
                            st.error(f"Model file {model_path} not found. Please train the model first.")
                        except Exception as e:
                            st.error(f"Failed to load model: {str(e)}")

                    if st.button("Predict on current data"):
                        if self.predictive_model is None or self.predictive_model.model is None:
                            st.error("No model loaded. Please train or load a model first.")
                        else:
                            with st.spinner("Predicting..."):
                                if segment_column != "None" and segment_value != "None":
                                    query = f"SELECT * FROM logs WHERE {segment_column} = '{segment_value}'"
                                else:
                                    query = "SELECT * FROM logs LIMIT 100000"
                                df = self.data_loader.connect.execute(query).fetchdf()
                                if df.empty:
                                    st.error(f"No data found for {segment_column} = {segment_value}")
                                    return
                                preds = self.predictive_model.predict(df, segment_column, segment_value)
                                st.write("Predictions: ", preds)

                with st.expander("🔍 Anomaly Detection"):
                    st.markdown("Detect anomalies in your data (no prior model training required)")
                    anomaly_column = st.selectbox("Select column for anomaly detection", self.columns)
                    segment_column_anomaly = st.selectbox("Segment by (anomaly detection):", ["None", "NAMESPACE", "CLUSTER"])
                    segment_values_anomaly = ["None"] if segment_column_anomaly == "None" else self.data_loader.connect.execute(f"SELECT DISTINCT {segment_column_anomaly} FROM logs LIMIT 100").fetchdf()[segment_column_anomaly].tolist()
                    segment_value_anomaly = st.selectbox("Segment value (anomaly detection):", segment_values_anomaly)
                    
                    if st.button("Detect Anomalies"):
                        try:
                            anomaly_model = PredictiveModel("temp.joblib", "isolation_forest")
                            with st.spinner("Detecting anomalies..."):
                                if segment_column_anomaly != "None" and segment_value_anomaly != "None":
                                    query = f"SELECT * FROM logs WHERE {segment_column_anomaly} = '{segment_value_anomaly}' LIMIT 100000"
                                else:
                                    query = "SELECT * FROM logs LIMIT 100000"
                                df = self.data_loader.connect.execute(query).fetchdf()
                                if df.empty:
                                    st.error(f"No data found for {segment_column_anomaly} = {segment_value_anomaly}")
                                    return
                                if not pd.api.types.is_numeric_dtype(df[anomaly_column]):
                                    st.error(f"Column {anomaly_column} must be numeric for anomaly detection.")
                                    return
                                anomalies = anomaly_model.detect_anomalies(df, anomaly_column, segment_column_anomaly, segment_value_anomaly)
                                if not anomalies.empty:
                                    st.subheader("Detected Anomalies")
                                    st.dataframe(anomalies)
                                else:
                                    st.info("No anomalies detected.")
                        except Exception as e:
                            st.error(f"Anomaly detection failed: {str(e)}")

                with st.expander("🚨 Alerts"):
                    st.markdown("Check for critical resource usage (no prior model training required)")
                    alert_metric = st.selectbox("Select metric for alerts", [col for col in self.columns if "PERCENT" in col])
                    segment_column_alert = st.selectbox("Segment by (alerts):", ["None", "NAMESPACE", "CLUSTER"])
                    segment_values_alert = ["None"] if segment_column_alert == "None" else self.data_loader.connect.execute(f"SELECT DISTINCT {segment_column_alert} FROM logs LIMIT 100").fetchdf()[segment_column_alert].tolist()
                    segment_value_alert = st.selectbox("Segment value (alerts):", segment_values_alert)
                    
                    if st.button("Check Alerts"):
                        try:
                            alert_model = PredictiveModel("temp.joblib", "isolation_forest")
                            with st.spinner("Checking alerts..."):
                                if segment_column_alert != "None" and segment_value_alert != "None":
                                    query = f"SELECT * FROM logs WHERE {segment_column_alert} = '{segment_value_alert}' LIMIT 100000"
                                else:
                                    query = "SELECT * FROM logs LIMIT 100000"
                                df = self.data_loader.connect.execute(query).fetchdf()
                                if df.empty:
                                    st.error(f"No data found for {segment_column_alert} = {segment_value_alert}")
                                    return
                                alerts = alert_model.check_alerts(df, alert_metric, threshold=90, segment_column=segment_column_alert, segment_value=segment_value_alert)
                                if not alerts.empty:
                                    st.subheader("Critical Alerts")
                                    st.dataframe(alerts)
                                else:
                                    st.info("No critical alerts found.")
                        except Exception as e:
                            st.error(f"Alert checking failed: {str(e)}")

                with st.expander("📊 Namespace Analysis"):
                    st.markdown("Visualize TOP namespaces by resource usage")
                    metric = st.selectbox("Select metric for visualization", [col for col in self.columns if "PERCENT" in col])
                    if st.button("Show TOP Namespaces"):
                        try:
                            query = f"SELECT NAMESPACE, {metric} FROM logs LIMIT 100000"
                            df = self.data_loader.connect.execute(query).fetchdf()
                            if df.empty:
                                st.error("No data available for visualization.")
                                return
                            self.plot_top_namespaces(df, metric)
                        except Exception as e:
                            st.error(f"Visualization failed: {str(e)}")

            except Exception as e:
                st.error(f"Failed to process CSV: {str(e)}")
                return

        else:
            st.info("⬅️ Please upload a CSV file to get started.")

if __name__ == "__main__":
    app = App("Vertis Data Consultant", layout="wide")
    app.run()