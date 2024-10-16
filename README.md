# mcts-llm-Qwen
This project is a branch of [repo](https://github.com/BrendanGraham14/mcts-llm). We use [Ollama](https://github.com/ollama/ollama-python) to deploy Qwen2.5 locally instead of using the online API.

## MCTSr

Based on [Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B](https://arxiv.org/abs/2406.07394) by Zhang, et al.

At a high level, MCTSr iteratively generates solutions to a specified (math) problem.

In a MCTSr tree, nodes correspond to attempted answers, and edges correspond to attempts to improve the answer.


## Usage


