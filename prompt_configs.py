from pydantic import BaseModel, Field


class PromptConfig(BaseModel):
    model: str
    critic_system_prompt: str
    refine_system_prompt: str
    evaluate_system_prompt: str
    base_url: str | None = None


# llama_3_8b_prompt_config = PromptConfig(
#     base_url="https://api.fireworks.ai/inference/v1",
#     model="accounts/fireworks/models/llama-v3-8b-instruct",
#     critic_system_prompt="Provide a detailed and constructive critique to improve the answer. "
#     "Highlight specific areas that need refinement or correction.",
#     refine_system_prompt="""# Instruction
# Refine the answer based on the critique. Your refined answer should be a direct and concise solution to the problem.

# ## Additional guidelines
# - Your response should not refer to or discuss the criticisms.
# - Do not repeat the problem statement.
# - Respond with only the answer.
# """,
#     evaluate_system_prompt=(
#         "Provide a reward score between -100 and 100 for the answer quality, using very strict standards. "
#         "Do not give a full score above 95. Make sure the reward score is an integer. "
#         "Return *ONLY* the score."
#     ),
# )

qwen_25_15b_prompt_config = PromptConfig(
    base_url="http://localhost:11434/v1/",
    model="qwen2.5:1.5b",
    critic_system_prompt="Provide a detailed and constructive critique to improve the answer. "
    "Highlight specific areas that need refinement or correction.",
    refine_system_prompt="""# Instruction
Refine the answer based on the critique. Your refined answer should be a direct and concise solution to the problem.

## Additional guidelines
- Your response should not refer to or discuss the criticisms.
- Do not repeat the problem statement.
- Respond with only the answer.
""",
    evaluate_system_prompt=(
        "Provide a reward score between -100 and 100 for the answer quality, using very strict standards. "
        "Do not give a full score above 95. Make sure the reward score is an integer. "
        "Do not use any markdown formatting or other symbol in your reply."
        "Return ONLY the score."
    ),
)


class RefineResponse(BaseModel):
    thought: str = Field(..., description="The thought process behind the answer.")
    answer: float = Field(..., description="The answer to the problem.")