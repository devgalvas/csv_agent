import os
import pandas as pd
from dotenv import load_dotenv
from agent.csv_agent import CSVAgent
import streamlit as st

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
            st.info("Made with Lucas Galvão Freitas](https://github.com/devgalvas)")

        self.agent = CSVAgent(api_key=api_key)

    def run(self):
        if self.uploaded_file:
            self.agent.load_csv(self.uploaded_file)
            st.success("✅ CSV file loaded successfully!")

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

            if self.show_data:
                with st.expander("🔍 Show raw data (first 100 rows)"):
                    with st.spinner("Loading sample..."):
                        try:
                            sample_df = self.agent.conn.execute(
                                f"SELECT * FROM read_csv_auto('{self.agent.csv_path}') LIMIT 100"
                            ).fetchdf()
                            st.dataframe(sample_df)
                        except Exception as e:
                            st.error(f"Could not show raw data: {e}")
        else:
            st.info("⬅️ Please upload a CSV file to get started.")

if __name__ == "__main__":
    app = App("Vertis Data Consultant", layout="wide")
    app.run()