from pydantic import BaseModel, Field
from operator import add
from typing import Dict, Any, Annotated, List, Optional

from api.rag.agent import ToolCall, RAGUsedContext, agent_node
from api.rag.utils.utils import mcp_tool_node, get_tool_descriptions_from_mcp_servers
from api.core.config import config

import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    answer: str = ""
    iteration: int = Field(default=0)
    final_answer: bool = Field(default=False)
    available_tools: List[Dict[str, Any]] = []
    tool_calls: Optional[List[ToolCall]] = Field(default_factory=list)
    retrieved_context_ids: List[RAGUsedContext] = []
    trace_id: str = ""


def tool_router(state: State) -> str:
    """Decide whether to continue or end"""
    
    if state.final_answer:
        return "end"
    elif state.iteration > 5:
        return "end"
    elif len(state.tool_calls) > 0:
        return "tools"
    else:
        return "end"


### WORKFLOW ###

workflow = StateGraph(State)

workflow.add_node("agent_node", agent_node)
workflow.add_node("mcp_tool_node", mcp_tool_node)

workflow.add_edge(START, "agent_node")

workflow.add_conditional_edges(
    "agent_node",
    tool_router,
    {
        "tools": "mcp_tool_node",
        "end": END
    }
)

workflow.add_edge("mcp_tool_node", "agent_node")


async def run_agent(question: str, thread_id: str):
    mcp_servers = ["http://items_mcp_server:8000/mcp", "http://reviews_mcp_server:8000/mcp"]

    tool_descriptions = await get_tool_descriptions_from_mcp_servers(mcp_servers)

    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "iteration": 0,
        "available_tools": tool_descriptions
    }

    config = { "configurable": { "thread_id": thread_id}}

    async with AsyncPostgresSaver.from_conn_string("postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db") as checkpointer:

        graph = workflow.compile(checkpointer=checkpointer)

        # graph.invoke(state, config=config)
        result = await graph.ainvoke(initial_state, config=config)

    return result


async def run_agent_wrapper(question: str, thread_id: str):

    qdrant_client = QdrantClient(url=config.QDRANT_URL)

    result = await run_agent(question, thread_id)

    image_url_list = []
    dummy_vector = np.zeros(1536).tolist()
    for id in result.get("retrieved_context_ids"):
        payload = qdrant_client.query_points(
            collection_name=config.QDRANT_COLLECTION_NAME_ITEMS,
            query=dummy_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="parent_asin",
                        match=MatchValue(value=id.id)
                    )
                ]
            ),
            with_payload=True,
            limit=1
        ).points[0].payload
        image_url = payload.get("first_large_image")
        price = payload.get("price")
        if image_url:
            image_url_list.append({"image_url": image_url, "price": price, "description": id.description})

    return {
        "answer": result.get("answer"),
        "retrieved_images": image_url_list,
        "trace_id": result.get("trace_id")
    }
