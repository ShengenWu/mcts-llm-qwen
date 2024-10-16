import mctsr
# from utils import load_dataframe
from utils import load_aime
from ollama import Client
from tqdm import tqdm

if __name__ == "__main__":
    aime = load_aime(2024)
    aime['model_answer'] = ""

    client = Client(host='http://localhost:11434')

    for i, row in tqdm(aime.iterrows(), total=len(aime)):
        question = row['question']
    
        # 创建 MCTSr 实例
        mcts_instance = mctsr.MCTSrQen2515B(
            problem=question,
            max_rollouts=8,
            exploration_constant=1.0,
            max_children=2
        )
    
        best_answer = mcts_instance.run()
        aime.at[i, 'model_answer'] = best_answer

    output_path = 'result/result_file.csv'
    aime.to_csv(output_path, index=False)

    print(f"结果已保存到 {output_path}")

    