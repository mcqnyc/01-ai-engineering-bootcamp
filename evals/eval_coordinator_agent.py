from langsmith import Client

from src.api.rag.agents import coordinator_agent_node
from src.api.rag.graph import State
from src.api.core.config import config


ls_client = Client(api_key=config.LANGSMITH_API_KEY)


def next_agent_evaluator_gpt_4_1(run, example):

    next_agent_match = run.outputs["next_agent"] == example.outputs["next_agent"]
    final_answer_match = run.outputs["coordinator_final_answer"] == example.outputs["coordinator_final_answer"]

    return next_agent_match and final_answer_match


def next_agent_evaluator_gpt_4_1_mini(run, example):

    next_agent_match = run.outputs["next_agent"] == example.outputs["next_agent"]
    final_answer_match = run.outputs["coordinator_final_answer"] == example.outputs["coordinator_final_answer"]

    return next_agent_match and final_answer_match


def next_agent_evaluator_groq_llama_3_3_70b_versatile(run, example):

    next_agent_match = run.outputs["next_agent"] == example.outputs["next_agent"]
    final_answer_match = run.outputs["coordinator_final_answer"] == example.outputs["coordinator_final_answer"]

    return next_agent_match and final_answer_match


results = ls_client.evaluate(
    lambda x: coordinator_agent_node(State(messages=x["messages"], models=["gpt-4.1"])),
    data="coordinator-evaluation-dataset",
    evaluators=[next_agent_evaluator_gpt_4_1],
    experiment_prefix="gpt-4.1",
)


results = ls_client.evaluate(
    lambda x: coordinator_agent_node(State(messages=x["messages"], models=["gpt-4.1-mini"])),
    data="coordinator-evaluation-dataset",
    evaluators=[next_agent_evaluator_gpt_4_1_mini],
    experiment_prefix="gpt-4.1-mini",
)


results = ls_client.evaluate(
    lambda x: coordinator_agent_node(State(messages=x["messages"], models=["groq/llama-3.3-70b-versatile"])),
    data="coordinator-evaluation-dataset",
    evaluators=[next_agent_evaluator_groq_llama_3_3_70b_versatile],
    experiment_prefix="groq/llama-3.3-70b-versatile",
)