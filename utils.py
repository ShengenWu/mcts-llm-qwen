"""

Utilities to load various datasets.

Dataset dfs are expected to have the following columns:
- question
- answer

"""
import pandas as pd

def load_dataframe(file_path):
    try:
        df = pd.read_pickle(file_path)
        return df
    except Exception as e:
        print(f"Error occurred while reading the pickle file: {e}")
        return None