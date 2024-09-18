import json
import os
from typing import List, Dict, Any

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableWithFallbacks, RunnableLambda
from langgraph.prebuilt import ToolNode
from termcolor import colored

from models.openai_models import get_open_ai, get_open_ai_json
from prompts.prompts import (
    selector_prompt_template,
    reviewer_prompt_template, generator_prompt_template,
)
from utils.helper_functions import check_for_content
from states.state import AgentGraphState


class Agent:
    def __init__(self, state: AgentGraphState, model=None, base_url=None, server=None, temperature=0,
                 model_endpoint=None, stop=None, guided_json=None):
        self.state = state
        self.model = model
        self.server = server
        self.temperature = temperature
        self.model_endpoint = model_endpoint
        self.stop = stop
        self.guided_json = guided_json
        self.base_url = base_url

    def get_llm(self, json_model=True):
        if self.server == 'openai':
            return get_open_ai_json(model=self.model, base_url=self.base_url,
                                    temperature=self.temperature) if json_model else get_open_ai(model=self.model,
                                                                                                 base_url=self.base_url,
                                                                                                 temperature=self.temperature)

    def update_state(self, key, value):
        self.state = {**self.state, key: value}


def init_messages(state: AgentGraphState) -> List[Dict]:
    return [
        {"role": "user", "content": f"user question: {state.get('user_question')}"}
    ]


db_uri = os.getenv("DB_URI")
db = SQLDatabase.from_uri(db_uri)


def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


class ListTablesAgent(Agent):
    def invoke(self):
        toolkit = SQLDatabaseToolkit(db=db, llm=get_open_ai())
        tools = toolkit.get_tools()

        list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")

        # todo
        #  1、封装工具调用逻辑，将调用结果抽象为绑定state对应的response中。
        #  2、抽象error处理(https://python.langchain.com/v0.2/docs/how_to/custom_tools/#handling-tool-errors)
        tabl_names = list_tables_tool.invoke("")

        print(colored(f"list_tables 🧑🏼‍💻: {tabl_names}", 'blue'))
        self.update_state("list_tables_tool_response", tabl_names)
        return self.state


class ListDbSchemaAgent(Agent):
    def invoke(self, selector=None):
        toolkit = SQLDatabaseToolkit(db=db, llm=get_open_ai())
        tools = toolkit.get_tools()
        get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

        selector_response = selector() if callable(selector) else selector
        selector_response = check_for_content(selector_response)

        if selector_response:
            selector_json = json.loads(selector_response)
            table_names = selector_json["tables"]
            if table_names and len(table_names) > 0:
                table_names_str = ",".join(table_names)
                # todo
                #  1、封装工具调用逻辑，将调用结果抽象为绑定state对应的response中。
                #  2、抽象error处理(https://python.langchain.com/v0.2/docs/how_to/custom_tools/#handling-tool-errors)
                table_schemas = get_schema_tool.invoke(table_names_str)
            else:
                return
        else:
            return

        print(colored(f"list_tables 🧑🏼‍💻: {table_schemas}", 'green'))
        self.update_state("list_db_schema_response", table_schemas)
        return self.state


class SelectorAgent(Agent):
    def invoke(self, list_tables_tool=None, prompt=selector_prompt_template):
        list_tables_tool_response = list_tables_tool() if callable(list_tables_tool) else list_tables_tool
        list_tables_tool_response = check_for_content(list_tables_tool_response)

        selector_prompt = prompt.format(
            table_names=list_tables_tool_response
        )

        messages = init_messages(self.state)
        messages.append({"role": "system", "content": selector_prompt})

        llm = self.get_llm()
        ai_msg = llm.invoke(messages)
        response = ai_msg.content

        print(colored(f"selector 🧑🏼‍💻: {response}", 'red'))
        self.update_state("selector_response", response)
        return self.state


class GeneratorAgent(Agent):
    def invoke(self, prompt=generator_prompt_template, reviewer=None, selector=None, list_db_schema=None):
        reviewer_value = reviewer() if callable(reviewer) else reviewer
        selector_value = selector() if callable(selector) else selector
        list_db_schema_value = list_db_schema() if callable(list_db_schema) else list_db_schema

        reviewer_value = check_for_content(reviewer_value)
        selector_value = check_for_content(selector_value)
        list_db_schema_value = check_for_content(list_db_schema_value)

        reviewer_feedback = "None"
        if reviewer_value:
            reviewer_json = json.loads(reviewer_value)
            reviewer_feedback = reviewer_json["feedback"]

        reporter_prompt = prompt.format(
            selector_response=selector_value,
            reviewer_feedback=reviewer_feedback,
            tables_schema=list_db_schema_value
        )

        messages = init_messages(self.state)
        messages.append({"role": "system", "content": reporter_prompt})

        llm = self.get_llm()
        ai_msg = llm.invoke(messages)
        response = ai_msg.content

        print(colored(f"Generator 👨‍💻: {response}", 'black'))
        self.update_state("generator_response", response)
        return self.state


class ExecutorAgent(Agent):
    def invoke(self, generator=None):
        toolkit = SQLDatabaseToolkit(db=db, llm=get_open_ai())
        tools = toolkit.get_tools()
        sql_db_query = next(tool for tool in tools if tool.name == "sql_db_query")

        generator_value = generator() if callable(generator) else generator
        generator_value = check_for_content(generator_value)

        sql = ""
        if generator_value:
            generator_json = json.loads(generator_value)
            sql = generator_json["sql"]
        if sql == "":
            return
        response = sql_db_query.invoke(sql)

        print(colored(f"executor 👨‍💻: {response}", 'yellow'))
        self.update_state("executor_response", response)
        return self.state


class ReviewerAgent(Agent):
    def invoke(self, generator=None, prompt=reviewer_prompt_template, executor=None, feedback=None):
        generator_value = generator() if callable(generator) else generator
        executor_value = executor() if callable(executor) else executor
        feedback_value = feedback() if callable(feedback) else feedback

        generator_value = check_for_content(generator_value)
        executor_value = check_for_content(executor_value)
        feedback_value = check_for_content(feedback_value)

        reviewer_prompt = prompt.format(
            generator_response=generator_value,
            executor_response=executor_value,
            feedback=feedback_value,
            state=self.state,
        )

        messages = init_messages(self.state)
        messages.append({"role": "system", "content": reviewer_prompt})

        llm = self.get_llm()
        ai_msg = llm.invoke(messages)
        response = ai_msg.content

        print(colored(f"Reviewer 👩🏽‍⚖️: {response}", 'magenta'))
        self.update_state("reviewer_response", response)
        return self.state


class FinalReportAgent(Agent):
    def invoke(self, reviewer_response=None, generator_response=None):
        reviewer_response_value = reviewer_response() if callable(reviewer_response) else reviewer_response
        generator_response_value = generator_response() if callable(generator_response) else generator_response

        reviewer_response_value = check_for_content(reviewer_response_value)
        generator_response_value = check_for_content(generator_response_value)

        sql_query = generator_feedback = reporter = reviewer_feedback = ""
        if generator_response_value:
            generator_response_json = json.loads(generator_response_value)
            sql_query = generator_response_json["sql"]
            generator_feedback = generator_response_json["generator_feedback"]
        if reviewer_response_value:
            reviewer_resp_json = json.loads(reviewer_response_value)
            reporter = reviewer_resp_json["reporter"]
            reviewer_feedback = reviewer_resp_json["feedback"]

        if sql_query == "":
            final_report = generator_feedback
        elif reporter == "":
            final_report = reviewer_feedback
        else:
            final_report = f"""
        {reporter}
        the sql query :
        {sql_query}
        """

        print(colored(f"Final Report 📝: {final_report}", 'blue'))
        self.update_state("final_reports", final_report)
        return self.state


class EndNodeAgent(Agent):
    def invoke(self):
        self.update_state("end_chain", "end_chain")
        return self.state
