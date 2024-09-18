import json
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage

from agents.agents import (
    SelectorAgent,
    GeneratorAgent,
    ReviewerAgent,
    FinalReportAgent,
    EndNodeAgent, ListTablesAgent, ExecutorAgent, ListDbSchemaAgent
)
from prompts.prompts import (
    reviewer_prompt_template,
    selector_prompt_template, generator_prompt_template,
)
from states.state import AgentGraphState, get_agent_graph_state

def create_graph(server=None, model=None, base_url=None, stop=None, model_endpoint=None, temperature=0):

    graph = StateGraph(AgentGraphState)

    graph.add_node(
        "list_tables",
        lambda state: ListTablesAgent(
            state=state,
            model=model,
            server=server,
            base_url=base_url,
            stop=stop,
            model_endpoint=model_endpoint,
            temperature=temperature
        ).invoke()
    )

    graph.add_node(
        "selector",
        lambda state: SelectorAgent(
            state=state,
            model=model,
            server=server,
            base_url=base_url,
            stop=stop,
            model_endpoint=model_endpoint,
            temperature=temperature
        ).invoke(
            list_tables_tool=lambda: get_agent_graph_state(state=state, state_key="list_tables_tool"),
            prompt=selector_prompt_template,
        )
    )

    graph.add_node(
        "list_db_schema",
        lambda state: ListDbSchemaAgent(
            state=state,
            model=model,
            server=server,
            base_url=base_url,
            stop=stop,
            model_endpoint=model_endpoint,
            temperature=temperature
        ).invoke(
            selector=lambda : get_agent_graph_state(state=state, state_key="selector_latest")
        )
    )

    graph.add_node(
        "generator",
        lambda state: GeneratorAgent(
            state=state,
            model=model,
            server=server,
            base_url=base_url,
            stop=stop,
            model_endpoint=model_endpoint,
            temperature=temperature
        ).invoke(
            selector=lambda: get_agent_graph_state(state=state, state_key="selector_latest"),
            list_db_schema=lambda: get_agent_graph_state(state=state, state_key="list_db_schema_latest"),
            reviewer=lambda: get_agent_graph_state(state=state, state_key="reviewer_latest"),
            prompt=generator_prompt_template
        )
    )

    graph.add_node(
        "executor",
        lambda state: ExecutorAgent(
            state=state,
            model=model,
            server=server,
            base_url=base_url,
            stop=stop,
            model_endpoint=model_endpoint,
            temperature=temperature
        ).invoke(
            generator=lambda: get_agent_graph_state(state=state, state_key="generator_latest"),
        )
    )

    graph.add_node(
        "reviewer",
        lambda state: ReviewerAgent(
            state=state,
            model=model,
            server=server,
            base_url=base_url,
            stop=stop,
            model_endpoint=model_endpoint,
            temperature=temperature
        ).invoke(
            generator=lambda: get_agent_graph_state(state=state, state_key="generator_latest"),
            executor=lambda: get_agent_graph_state(state=state, state_key="executor_latest"),
            feedback=lambda: get_agent_graph_state(state=state, state_key="reviewer_all"),
            prompt=reviewer_prompt_template
        )
    )

    graph.add_node(
        "final_report",
        lambda state: FinalReportAgent(
            state=state
        ).invoke(
            reviewer_response=lambda: get_agent_graph_state(state=state, state_key="reviewer_latest"),
            generator_response=lambda: get_agent_graph_state(state=state, state_key="generator_latest")
        )
    )

    graph.add_node("end", lambda state: EndNodeAgent(state).invoke())

    # Define the edges in the agent graph
    def pass_review(resp: list):
        if resp:
            review = resp[-1]
        else:
            review = "No review"

        if review != "No review":
            if isinstance(review, HumanMessage):
                review_content = review.content
            else:
                review_content = review

            review_data = json.loads(review_content)
            next_agent = review_data["next_agent"]
        else:
            next_agent = "end"

        return next_agent

    # Add edges to the graph
    graph.set_entry_point("list_tables")
    graph.set_finish_point("end")
    graph.add_edge("list_tables", "selector")
    graph.add_edge("selector", "list_db_schema")
    graph.add_edge("list_db_schema", "generator")
    graph.add_conditional_edges(
        "generator",
        lambda state: pass_review(resp=state["generator_response"]),
    )
    graph.add_edge("executor", "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        lambda state: pass_review(resp=state["reviewer_response"]),
    )
    graph.add_edge("final_report", "end")

    return graph


def compile_workflow(graph: StateGraph):
    workflow = graph.compile()
    return workflow