from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

# Define the state object for the agent graph
class AgentGraphState(TypedDict):
    user_question: str
    list_tables_tool_response: Annotated[list, add_messages]
    selector_response: Annotated[list, add_messages]
    list_db_schema_response: Annotated[list, add_messages]
    generator_response: Annotated[list, add_messages]
    executor_response: Annotated[list, add_messages]
    reviewer_response: Annotated[list, add_messages]
    final_reports: Annotated[list, add_messages]
    end_chain: Annotated[list, add_messages]

# Define the nodes in the agent graph
def get_agent_graph_state(state:AgentGraphState, state_key:str):
    if state_key == "list_tables_tool":
        return state["list_tables_tool_response"]

    elif state_key == "selector_all":
        return state["selector_response"]
    elif state_key == "selector_latest":
        if state["selector_response"]:
            return state["selector_response"][-1]
        else:
            return state["selector_response"]

    elif state_key == "list_db_schema_all":
        return state["list_db_schema_response"]
    elif state_key == "list_db_schema_latest":
        if state["list_db_schema_response"]:
            return state["list_db_schema_response"][-1]
        else:
            return state["list_db_schema_response"]

    elif state_key == "generator_all":
        return state["generator_response"]
    elif state_key == "generator_latest":
        if state["generator_response"]:
            return state["generator_response"][-1]
        else:
            return state["generator_response"]

    elif state_key == "executor_all":
        return state["executor_response"]
    elif state_key == "executor_latest":
        if state["executor_response"]:
            return state["executor_response"][-1]
        else:
            return state["executor_response"]

    elif state_key == "reviewer_all":
        return state["reviewer_response"]
    elif state_key == "reviewer_latest":
        if state["reviewer_response"]:
            return state["reviewer_response"][-1]
        else:
            return state["reviewer_response"]

    else:
        return None
    
state = {
    "user_question":"",
    "list_tables_tool_response": [],
    "selector_response": [],
    "list_db_schema_response": [],
    "generator_response": [],
    "executor_response":[],
    "reviewer_response": [],
    "final_reports": [],
    "end_chain": []
}