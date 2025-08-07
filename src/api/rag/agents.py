from pydantic import BaseModel, Field
from typing import List
import instructor
from openai import OpenAI
from langsmith import traceable, get_current_run_tree
from langchain_core.messages import AIMessage

from api.rag.utils.utils import lc_messages_to_regular_messages, prompt_template_config
from api.core.config import config


client = instructor.from_openai(OpenAI(api_key=config.OPENAI_API_KEY))


class ToolCall(BaseModel):
    name: str
    arguments: dict
    server: str


class RAGUsedContext(BaseModel):
    id: str
    description: str


class ProductQAAgentResponse(BaseModel):
    answer: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    final_answer: bool = Field(default=False)
    retrieved_context_ids: List[RAGUsedContext]


class IntentRouterAgentResponse(BaseModel):
    user_intent: str
    answer: str


### Product QA Agent ###

@traceable(
    name="product_qa_agent",
    run_type="llm",
    metadata={"ls_provider": config.GENERATION_MODEL_PROVIDER, "ls_model_name": config.GENERATION_MODEL}
)
def product_qa_agent_node(state) -> dict:

    prompt_template = prompt_template_config(config.RAG_PROMPT_TEMPLATE_PATH, "product_qa_agent")

    prompt = prompt_template.render(available_tools=state.available_tools)

    messages = state.messages

    conversation = []

    for msg in messages:
        conversation.append(lc_messages_to_regular_messages(msg))


    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1",
        response_model=ProductQAAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }

    if response.tool_calls and not response.final_answer:
        tool_calls = []
        for i, tc in enumerate(response.tool_calls):
            tool_calls.append({
                "id": f"call_{i}",
                "name": tc.name,
                "args": tc.arguments
            })

        ai_message = AIMessage(
            content=response.answer,
            tool_calls=tool_calls
            )
    else:
        ai_message = AIMessage(
            content=response.answer,
        )

    return {
        "messages": [ai_message],
        "tool_calls": response.tool_calls,
        "iteration": state.iteration + 1,
        "answer": response.answer,
        "final_answer": response.final_answer,
        "retrieved_context_ids": response.retrieved_context_ids,
    }


### Intent Router Agent ###

@traceable(
    name="intent_router_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def intent_router_agent_node(state) -> dict:

   prompt_template = prompt_template_config(config.RAG_PROMPT_TEMPLATE_PATH, "product_qa_agent")
   prompt = prompt_template.render(available_tools=state.available_tools)

   messages = state.messages

   conversation = []

   for msg in messages:
      conversation.append(lc_messages_to_regular_messages(msg))

   client = instructor.from_openai(OpenAI())

   response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1",
        response_model=IntentRouterAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0,
   )

   current_run = get_current_run_tree()
   if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }
        trace_id = str(getattr(current_run, "trace_id", current_run.id))


   if response.user_intent == "product_qa":

      ai_message = []
   else:
      ai_message = [AIMessage(
         content=response.answer,
      )]

   return {
      "messages": ai_message,
      "answer": response.answer,
      "user_intent": response.user_intent,
      "trace_id": trace_id,
   }