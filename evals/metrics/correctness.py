# evals/metrics/correctness.py
from langchain_openai import ChatOpenAI

JUDGE_PROMPT = """
You are an expert evaluator grading an AI multi-agent system output.

User Input: {user_input}
Expected Key Facts: {expected_facts}
Agent Response: {response}

Grade the response on a scale of 0.0 to 1.0 based on factual accuracy and coverage of key facts:
- 1.0 = Completely accurate and covers key facts cleanly.
- 0.5 = Partially accurate or missing minor key facts.
- 0.0 = Factually incorrect, hallucinatory, or completely missed key facts.

Return ONLY a float number representing the score.
"""


def evaluate_correctness(user_input: str, expected_facts: list[str], response: str) -> float:
    # Lazy-load LLM judge here so it doesn't fail during test collection
    llm_judge = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = JUDGE_PROMPT.format(
        user_input=user_input,
        expected_facts=", ".join(expected_facts),
        response=response,
    )
    res = llm_judge.invoke(prompt)
    try:
        return float(res.content.strip())
    except ValueError:
        return 0.0
