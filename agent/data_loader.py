import duckdb
import pandas as pd

class DataLoader:
    """
    DataLoader is a utility class for loading and querying data (CSV or Parquet) using DuckDB.

    Attributes:
        file_path (str): File path to the data file (CSV or Parquet).
        samples (int): Number of samples to retrieve when sampling data.
        query (str): SQL query string to filter or select data from the loaded table.
        file_type (str): Type of file ('csv' or 'parquet').

    Methods:
        connect_to_db():
            Establishes a DuckDB connection and loads the data into a table named 'logs'.
        get_sample():
            Retrieves a sample of rows from the 'logs' table, limited by the 'samples' attribute.
        get_filtered():
            Executes the provided SQL query on the 'logs' table and returns the result as a DataFrame.
        get_data_by_segment(segment_type, segment_value):
            Retrieves data filtered by a specific segment (e.g., NAMESPACE or CLUSTER).
    """

    def __init__(self, file_path: str, samples: int, query: str, file_type: str = "parquet"):
        self.file_path = file_path
        self.samples = samples
        self.query = query
        self.file_type = file_type.lower()  
        self.connect = None

    def connect_to_db(self):
        self.connect = duckdb.connect()
        if self.file_type == "csv":
            self.connect.execute(f"""
                CREATE OR REPLACE TABLE logs AS
                SELECT * FROM read_csv('{self.file_path}',
                delimiter=',',
                quote='"',
                header=True,
                max_line_size=10000000,
                ignore_errors=True)
            """)
        elif self.file_type == "parquet":
            self.connect.execute(f"CREATE OR REPLACE TABLE logs AS SELECT * FROM parquet_scan('{self.file_path}')")
        else:
            raise ValueError("Unsuported file type. Use 'csv' or 'parquet'.")
        return self.connect
    
    def get_data_by_segment(self, segment_type: str, segment_value: str):
        """
        segment_type (str): "ocnr_tx_namespace" or other segment column
        segment_value (str): The value to filter by (e.g., 'openshift-kube-scheduler-operator')
        """
        query = f"SELECT * FROM logs WHERE {segment_type} = '{segment_value}'"
        return self.connect.execute(query).fetchdf()

    def get_sample(self):
        return self.connect.execute(f"SELECT * FROM logs LIMIT {self.samples}").fetchdf()

    def get_filtered(self):
        return self.connect.execute(self.query).fetchdf()