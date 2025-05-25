from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI


# Create agents using configured LLM types
def create_agent(agent_name: str, agent_type: str, tools: list, prompt_template: str):
    """Factory function to create agents with consistent configuration."""
    return create_react_agent(
        name=agent_name,
        model=agent_type,
        tools=tools,
        prompt=prompt_template,
    )