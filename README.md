# Langgraph DB Agent

A proxy for database queries using LangGraph technology

This project offers a template for you to easily build and run your own db agent using the LangGraph framework.

It also provides a FastAPI interface to offer external services.

### Quickstart

Run directly in python

```sh
# An OPENAI_API_KEY、DB_URI is required
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
echo 'DB_URI=mysql+pymysql://root:pwd@ip:port/db' >> .env
# Optional
echo 'AUTH_SECRET=your fastapi interface call auth secret' >> .env

# uv is recommended but pip also works
pip install uv
uv pip install -r pyproject.toml
python src/run_service.py

# In another shell
streamlit run src/streamlit_app.py
```

Run with docker

```sh
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
echo 'DB_URI=mysql+pymysql://root:pwd@ip:port/db' >> .env
docker-compose up -d
```

### Architecture Diagram

<img src="media/agent_architecture.png" width="600">

## Grateful
https://github.com/john-adeojo/graph_websearch_agent
https://github.com/JoshuaC215/agent-service-toolkit

## License

This project is licensed under the MIT License - see the LICENSE file for details.
