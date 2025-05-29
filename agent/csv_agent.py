import duckdb
import os
import pandas as pd
import requests
import tempfile

class CSVAgent:
    def __init__(self, api_key=None, model: str = 'llama3'):
        self.api_key = api_key
        self.model = model
        self.csv_path = None
        self.conn = duckdb.connect()
        self.system_prompt = (
            "You are a helpful assistant that converts natural language questions into SQL queries. "
            "The user is analyzing a CSV file using DuckDB. You must only return the SQL query, no explanations."
        )

    def load_csv(self, file):
        # Save uploaded file to a temporary file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file.name)

        with open(temp_path, "wb") as f:
            f.write(file.read())

        self.csv_path = temp_path

    def _get_schema(self):
        try:
            df = self.conn.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{self.csv_path}')").fetchdf()
            return "\n".join(f"{row['column_name']} ({row['column_type']})" for _, row in df.iterrows())
        except Exception as e:
            return f"⚠️ Error getting schema: {e}"

    def _question_to_sql(self, question: str):
        schema = self._get_schema()
        prompt = (
            f"{self.system_prompt}\n\n"
            f"CSV File Schema:\n{schema}\n\n"
            f"Question: {question}\n\n"
            f"SQL:"
        )
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else None,
                json={"model": self.model, "prompt": prompt, "stream": False}
            )
            sql = response.json().get("response", "").strip()
            return sql
        except Exception as e:
            return f"⚠️ LLM error: {e}"

    def ask(self, question: str):
        if not self.csv_path:
            return "❌ No CSV loaded."

        sql = self._question_to_sql(question)
        if not sql.lower().startswith("select"):
            return f"⚠️ Invalid SQL generated: `{sql}`"

        try:
            query = sql.replace("FROM", f"FROM read_csv_auto('{self.csv_path}')", 1)
            result_df = self.conn.execute(query).fetchdf()
            return result_df if not result_df.empty else "🤷 No results found."
        except Exception as e:
            return f"⚠️ Query error: {e}"
