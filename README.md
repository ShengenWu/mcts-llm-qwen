# mcts-llm-Qwen
This project is a branch of [repo](https://github.com/BrendanGraham14/mcts-llm). We use [Ollama](https://github.com/ollama/ollama-python) to deploy Qwen2.5 locally instead of using the online API.

## Useful Links

Paper Link: [Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B](https://arxiv.org/abs/2406.07394) by Zhang, et al.

Previous Project: [mcts-llm](https://github.com/BrendanGraham14/mcts-llm) by BrendanGraham14. This project using Fireworks.ai and OpendAI API to deploy LLaMa-3 8B and GPT-4o.

## Usage
We need ollama-python to deploy Qwen2.5 locally.
```
pip install ollama
```
Then, you should design the main.py file personally. In this file, it must contain the process of activating the client and load your datasets. Here is an example.
```
import mctsr
# from utils import load_dataframe
from utils import load_aime
from ollama import Client
from tqdm import tqdm

if __name__ == "__main__":
    # Load dataset here
    aime = load_aime(2024)
    aime['model_answer'] = ""
    # Activate client here
    client = Client(host='http://localhost:11434')
    # Run MCTS instance here
    for i, row in tqdm(aime.iterrows(), total=len(aime)):
        question = row['question']
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

    print(f"Results save to {output_path}") 
```

