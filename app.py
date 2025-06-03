import os
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from agent.csv_agent import CSVAgent
from agent.predictive_model import PredictiveModel
from agent.data_loader import DataLoader
from sklearn.metrics import confusion_matrix

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
            self.uploaded_file = st.file_uploader("Upload a CSV file (NOW SUPPORTED BIGGER FILES!)", type="csv")
            self.show_data = st.checkbox("🔍 Show raw data")
            st.markdown("---")
            st.info("Made by Lucas Galvão Freitas](https://github.com/devgalvas)")

        
        self.agent = CSVAgent(api_key=api_key)
        self.data_loader = None
        self.predictive_model = None
        self.df = None

    def plot_top_namespaces(self, df, metric, top_n=5):
        df_sorted = df.sort_values(by=metric, ascending=False)
        top_over = df_sorted.head(top_n)
        top_under = df_sorted.tail(top_n)

        st.subheader(f"TOP {top_n} Namespaces per {metric}")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=metric, y='NAMESPACE', hue='NAMESPACE', data=pd.concat([top_over, top_under]), palette="coolwarm")
        ax.set_xlabel(metric)
        ax.set_ylabel("Namespace")
        st.pyplot(fig)

    def run(self):
        # CSV Q&A and data preview
        if self.uploaded_file:
            self.agent.load_csv(self.uploaded_file)
            st.success("✅ CSV file loaded successfully!")

            # LOad dataframe for dataloader and predictive model
            self.df = pd.read_csv(self.agent.csv_path)
            self.data_loader = DataLoader(self.agent.csv_path, samples=100, query="SELECT * FROM logs LIMIT 100")
            self.data_loader.connect_to_db()

            if self.show_data:
                st.subheader("Raw Data")
                st.dataframe(self.df)

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

            # Precitive modeling section
            with st.expander("🤖 Predictive Modeling"):
            
                st.markdown("Train a model on your data and make predictions")
            
                target_column = st.selectbox("Select a target column", self.df.columns)
                model_type = st.selectbox("Model Type:", ["linear", "random_forest", "prophet"])
                segment_column = st.selectbox("Segment by:", ["None", "NAMESPACE", "CLUSTER"])
                segment_value = st.selectbox("Segment value:", ["None"] + self.df[segment_column].unique().tolist() if segment_column != "None" else ["None"])
                model_path = f"model_{target_column}_{segment_column}_{segment_value}.joblib" if segment_column != "None" else "model.joblib"

                if st.button("Train Model"):
                    namespaces = self.data_loader.connect.execute("SELECT DISTINCT NAMESPACE FROM logs").fetchdf()['NAMESPACE'].tolist()
                    for namespace in namespaces:
                        df_segment = self.data_loader.get_data_by_segment("NAMESPACE", namespace)
                        self.predictive_model = PredictiveModel(model_path, model_type)

                    with st.spinner("Training ...  this can take a while"):
                        
                        if segment_column != "None":
                            self.predictive_model.train(self.df, target_column, segment_column, segment_value)
                        else:
                            self.predictive_model.train(self.df, target_column)
                        
                        st.success("Model trained!")
                        st.write("Metrics: ", self.predictive_model.metrics)
                        
                        if model_type == "random_forest":
                            st.subheader("Confusion Matrix")
                            fig, ax = plt.subplots()
                            cm = confusion_matrix(self.predictive_model.y_test, 
                                                  self.predictive_model.y_pred)
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
                
                if st.button("Load Model"):
                    self.predictive_model = PredictiveModel(model_path, model_type)
                    self.predictive_model.load()
                    st.success("Model Loaded!")
                
                if self.predictive_model and st.button("Predict on current data"):                    
                    with st.spinner("Predicting..."):
                        if segment_column != "None":
                            preds = self.predictive_model.predict(self.df, segment_column, self.df_columns)
                        else:
                            preds = self.predictive_model.predict(self.df)
                        st.write("Predictions: ", preds)
            with st.expander("🔍 Anomaly Detection"):
                st.markdown("Detect anomalies in your data")
                anomaly_column = st.selectbox("Select column for anomaly detection", self.df.columns)
                if st.button("Detect Anomalies"):
                    with st.spinner("Detecting anomalies..."):
                        if segment_column != "None":
                            anomalies = self.predictive_model.detect_anomalies(self.df, anomaly_column, segment_column, segment_value)
                        else:
                            anomalies = self.predictive_model.detect_anomalies(self.df, anomaly_column)
                        if not anomalies.empty:
                            st.subheader("Detected Anomalies")
                            st.dataframe(anomalies)
                        else:
                            st.info("No anomalies detected.")
            
            with st.expander("🚨 Alerts"):
                st.markdown("Check for critical resource usage")
                alert_metric = st.selectbox("Select metric for alerts", [col for col in self.df.columns if "PERCENT" in col])
                if st.button("Check Alerts"):
                    with st.spinner("Checking alerts..."):
                        if segment_column != "None":
                            alerts = self.predictive_model.check_alerts(self.df, alert_metric, threshold=90, segment_column=segment_column, segment_value=segment_value)
                        else:
                            alerts = self.predictive_model.check_alerts(self.df, alert_metric, threshold=90)
                        if not alerts.empty:
                            st.subheader("Critical Alerts")
                            st.dataframe(alerts)
                        else:
                            st.info("No critical alerts found.")

            with st.expander("📊 Namespace Analysis"):
                st.markdown("Visualize TOP namespaces by resource usage")
                metric = st.selectbox("Select metric for visualization", [col for col in self.df.columns if "PERCENT" in col])
                if st.button("Show TOP Namespaces"):
                    self.plot_top_namespaces(self.df, metric)
        else:
            st.info("⬅️ Please upload a CSV file to get started.")

if __name__ == "__main__":
    app = App("Vertis Data Consultant", layout="wide")
    app.run()