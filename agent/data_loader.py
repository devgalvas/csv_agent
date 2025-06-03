import duckdb
import pandas as pd

class DataLoader:
    """
    DataLoader is a utility class for loading and querying CSV data using DuckDB.
    
    Attributes:
        csv_pattern (str): File path or pattern to the CSV files to be loaded.
        samples (int): Number of samples to retrieve when sampling data.
        query (str): SQL query string to filter or select data from the loaded table.
    
    Methods:
        connect_to_db():
            Establishes a DuckDB connection and loads the CSV data into a table named 'logs'.
        get_sample():
            Retrieves a sample of rows from the 'logs' table, limited by the 'samples' attribute.
        get_filtered():
            Executes the provided SQL query on the 'logs' table and returns the result as a DataFrame.
    """
    
    def __init__(self, csv_pattern: str, samples: int, query: str):
        self.csv_pattern = csv_pattern
        self.samples = samples
        self.query = query

    def connect_to_db(self):
        self.connect = duckdb.connect()
        self.connect.execute(f"CREATE OR REPLACE TABLE logs AS SELECT * FROM '{self.csv_pattern}'")
        return self.connect
    
    def get_data_by_segment(self, segment_type: str, segment_value: str):
        """
            segment_type (str): "NAMESPACE" or "CLUSTER"
            segment_value (str): the name of the specific namespace or cluster ex: panda-redis
        """
        query = f"SELECT * FROM logs WHERE {segment_type} = '{segment_value}'"

    def get_sample(self):
        return self.connect.execute("SELECT * FROM logs LIMIT {self.samples}").fetchdf()
    
    def get_filtered(self):
        return self.connect.execute(self.query).fetchdf()
    