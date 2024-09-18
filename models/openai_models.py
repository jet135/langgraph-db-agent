from langchain_openai import ChatOpenAI

def get_open_ai(temperature=0, model='gpt-4o-mini', base_url=''):
    llm = ChatOpenAI(
        model=model, base_url=base_url,
        temperature=temperature,
    )
    return llm


def get_open_ai_json(temperature=0, model='gpt-4o-mini', base_url=''):
    llm = ChatOpenAI(
        model=model, base_url=base_url,
        temperature=temperature,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    return llm
