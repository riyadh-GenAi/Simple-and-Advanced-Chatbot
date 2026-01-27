import warnings

from langchain._api import LangChainDeprecationWarning

warnings.simplefilter("ignore", category=LangChainDeprecationWarning)

import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
groq_api_key = os.environ["GROQ_API_KEY"]

from langchain_groq import ChatGroq

chatbot = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

from langchain_core.messages import HumanMessage

messagesToTheChatbot = [
    HumanMessage(content="My Code Name is Riyadh_genAI"),
]

response = chatbot.invoke(messagesToTheChatbot)

print("\n----------\n")

print("My Code Name is Riyadh_genAI")

print("\n----------\n")
print(response.content)

print("\n----------\n")

response = chatbot.invoke([
    HumanMessage(content="What is my code name?")
])

print("\n----------\n")

print("What is my code name?")

print("\n----------\n")
print(response.content)

print("\n----------\n")

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

chatbotMemory = {}

# input: session_id, output: chatbotMemory[session_id]
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chatbotMemory:
        chatbotMemory[session_id] = ChatMessageHistory()
    return chatbotMemory[session_id]


chatbot_with_message_history = RunnableWithMessageHistory(
    chatbot, 
    get_session_history
)

session1 = {"configurable": {"session_id": "001"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="My Code Name is Riyadh_genAI")],
    config=session1,
)

print("\n----------\n")

print("My Code Name is Riyadh_genAI")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="What's my code name?")],
    config=session1,
)

print("\n----------\n")

print("What's my code name? (in session1)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

session2 = {"configurable": {"session_id": "002"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="What's my Code name?")],
    config=session2,
)

print("\n----------\n")

print("What's my code name? (in session2)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

session1 = {"configurable": {"session_id": "001"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="What's my Code Name?")],
    config=session1,
)

print("\n----------\n")

print("What's my Code Name? (in session1 again)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

session2 = {"configurable": {"session_id": "002"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="My name is Riyadh.")],
    config=session2,
)

print("\n----------\n")

print("My name is Riyadh. (in session2)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="What is my name?")],
    config=session2,
)

print("\n----------\n")

print("What is my name? (in session2)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="What is my  Name?")],
    config=session1,
)

print("\n----------\n")

print("What is my  Name? (in session1)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough


def limited_memory_of_messages(messages, number_of_messages_to_keep=2):
    return messages[-number_of_messages_to_keep:]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

limitedMemoryChain = (
    RunnablePassthrough.assign(messages=lambda x: limited_memory_of_messages(x["messages"]))
    | prompt 
    | chatbot
)

chatbot_with_limited_message_history = RunnableWithMessageHistory(
    limitedMemoryChain,
    get_session_history,
    input_messages_key="messages",
)

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="My favorite vehicles are Vespa scooters.")],
    config=session1,
)

print("\n----------\n")

print("My favorite vehicles are Vespa scooters. (in session1)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="My favorite city is San Francisco.")],
    config=session1,
)

print("\n----------\n")

print("My favorite city is San Francisco. (in session1)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

responseFromChatbot = chatbot_with_limited_message_history.invoke(
    {
        "messages": [HumanMessage(content="what is my Code Name?")],
    },
    config=session1,
)

print("\n----------\n")

print("what is my Code name? (chatbot with memory limited to the last 3 messages)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="what is my Code Name?")],
    config=session1,
)

print("\n----------\n")

print("what is my Code Name? (chatbot with unlimited memory)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

# print(chatbotMemory)
