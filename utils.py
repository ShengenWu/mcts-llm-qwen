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
    
def load_aime(year: int | None = None) -> pd.DataFrame:
    df = pd.read_csv("/Users/shanewu/Code_Project/MathBlackBox/mcts-llm/datasets/AIME_Dataset_1983_2024.csv")
    df.rename(columns={"Question": "question", "Answer": "answer"}, inplace=True)
    if year is not None:
        df = df[df["Year"] == year]

    return df