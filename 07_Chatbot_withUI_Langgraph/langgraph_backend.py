from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
)
llm = ChatHuggingFace(llm=llm1)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# first streamlit
# config1 = {"configurable": {"thread_id": "1"}}
# chatbot.invoke({'messages': [HumanMessage(content='capital of France?')]}, config=config1)

# chatbot.get_state(config1)
# list(chatbot.get_state_history(config1))


# Second streamlit with streaming
# for message_chunk, metadata in chatbot.stream(
#     {'messages': [HumanMessage(content='capital of France?')]},
#     config={'configurable': {'thread_id': 'thread-1'}},
#     stream_mode='messages',
# ):
#     print(message_chunk)