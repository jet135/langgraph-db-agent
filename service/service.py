import os
from contextlib import asynccontextmanager
from typing import Dict, Any, Tuple
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from langchain_core.runnables import RunnableConfig
from langgraph.graph.graph import CompiledGraph

from agent_graph.graph import create_graph, compile_workflow
from schema.schema import ChatMessage, UserInput

server = 'openai'
model = 'gpt-4o-mini'

graph = create_graph(server=server, model=model)
workflow = compile_workflow(graph)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = workflow
    yield

app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def check_auth_header(request: Request, call_next):
    if auth_secret := os.getenv("AUTH_SECRET"):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return Response(status_code=401, content="Missing or invalid token")
        if auth_header[7:] != auth_secret:
            return Response(status_code=401, content="Invalid token")
    return await call_next(request)


def _parse_input(user_input: UserInput) -> Tuple[Dict[str, Any], str]:
    run_id = uuid4()
    kwargs = dict(
        input={"user_question": user_input.message},
        config=RunnableConfig(
            run_id=run_id,
            recursion_limit=user_input.recursion_limit if user_input.recursion_limit else 1
        ),
    )
    return kwargs, run_id


@app.post("/invoke")
async def invoke(user_input: UserInput) -> ChatMessage:
    """
    Invoke the agent with user input to retrieve a final response.
    """
    agent: CompiledGraph = app.state.agent
    kwargs, run_id = _parse_input(user_input)
    try:
        response = await agent.ainvoke(**kwargs)
        output = ChatMessage.from_langchain(response["final_reports"][-1])
        output.run_id = str(run_id)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
