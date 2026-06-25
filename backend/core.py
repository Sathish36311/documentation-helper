import os
from typing import Dict,Any,List
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()


# Initialize Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# Initialize Pinecone
vectorstore = PineconeVectorStore(index_name="langchain-docs-2026", embedding=embeddings)
# Initialize chat model
model = init_chat_model("gpt-5.2", model_provider="openai")


@tool(response_format="content_and_artifact")
def retrieve_context(query:str):
    """Retrieve relevant documentation to help answer user queries about LangChain"""
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    # Serialize documents for the model
    serialized = "\n\n".join((f"Source: {doc.metadata.get("source", "Unknown")}\n\nContent: {doc.page_content}") for doc in retrieved_docs)
    return serialized, retrieved_docs


def run_llm(query:str) -> Dict[str,Any]: 
    """Run the RAG pipeline to answer the query using retrieved documentation.
    Args:
        query: The user's question

    Returns:
        Dictionary containing:
            - answer : The generated answer
            - contect : List of retrieved documents
    """
    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers question about LangChain documentation."
        "You have access to a tool that retrieves relevant documentaton."
        "Use the tool to find relevant information before answering the question."
        "Always cite the sources you use in your answers."
        "If you cannot find the answer in the retrieved documenation, say so"
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)
    messages = [{"role": "user", "content": query}]
    response = agent.invoke({"messages": messages})

    answer = response["messages"][-1].content

    # Extract context docs so that we can understand where the ai got answer from, make us more confident to relay on AI
    context_docs = []
    for message in response["messages"]:
        if isinstance(message,ToolMessage) and hasattr(message,'artifact'):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {"answer": answer,"context": context_docs}



if __name__ == "__main__":
    result = run_llm(query="What are deep agents")
    print(result)