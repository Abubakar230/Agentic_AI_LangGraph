from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

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

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
# Checkpointer
checkpointer = SqliteSaver(conn=conn)
# checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)

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