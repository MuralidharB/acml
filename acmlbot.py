
import requests
from langchain_community.utilities import SQLDatabase
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated, Literal

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver

verbose=True
standalone=False

db = SQLDatabase.from_uri("mysql://root@localhost/acml")
if verbose:
   print(db.dialect)
   print(db.get_usable_table_names())
   db.run("SELECT * FROM Users LIMIT 10;")


toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(model="gpt-4o"))
tools = toolkit.get_tools()

list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

if verbose:
    print(list_tables_tool.invoke(""))
    print(get_schema_tool.invoke("Users"))


memory = MemorySaver()
config = {"configurable": {"thread_id": "1"}}


def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="Error"
    )


def handle_tool_error(state) -> dict:
    error = state.get("Error")
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


@tool
def db_query_tool(query: str) -> str:
    """
    Execute a SQL query against the database and get back the result.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    result = db.run_no_throw(query)
    #if 'Error' in result:
        #return "Error: Query failed. Please rewrite your query and try again."
    #print(result)
    return result


if verbose:
   print(db_query_tool.invoke("SELECT * FROM Users LIMIT 10;"))


query_check_system = """You are a SQL expert with a strong attention to detail.
Double check the MySQL query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

- Use id if name is given. You can get the id from corresponding table.
If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

You will call the appropriate tool to execute the query after running this check."""

query_check_prompt = ChatPromptTemplate.from_messages(
    [("system", query_check_system), ("placeholder", "{messages}")]
)
query_check = query_check_prompt | ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(
    [db_query_tool], tool_choice="required"
)

if verbose:
    print(query_check.invoke({"messages": [("user", "SELECT * FROM Users LIMIT 10;")]}))

# Define the state for the agent
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Define a new graph
workflow = StateGraph(State)

# Add a node for the first tool call
def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }


def model_check_query(state: State) -> dict[str, list[AIMessage]]:
    """
    Use this tool to double-check if your query is correct before executing it.
    """
    print([query_check.invoke({"messages": [("user", state["messages"][-1].tool_calls[0]['args']['query'])]})])

    return {"messages": [state["messages"][-1]]}
    #return {"messages": [query_check.invoke({"messages": [("user", state["messages"][-1].tool_calls[0]['args']['query'])]})]}


workflow.add_node("first_tool_call", first_tool_call)

# Add nodes for the first two tools
workflow.add_node(
    "list_tables_tool", create_tool_node_with_fallback([list_tables_tool])
)
workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))

# Add a node for a model to choose the relevant tables based on the question and available tables
model_get_schema = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(
    [get_schema_tool]
)
workflow.add_node(
    "model_get_schema",
    lambda state: {
        "messages": [model_get_schema.invoke(state["messages"])],
    },
)

name_to_id_system = """You are a SQL expert with a strong attention to detail.
Your objective is to parse the user input and if the input contains user names,
fetch the corresponding uuid/id of the resource from from the database.
output a syntactically correct MySQL query, look at the restults and rewrite the user input with the resoure id
If the user input does not have workload name or snapshot name or restore name, don't modify the input string.

If you get an error while executing a query, rewrite the query and try again.

If you get an empty result set, you should try to rewrite the query to get a non-empty result set. 
NEVER make stuff up if you don't have enough information to answer the query... just say you don't have enough information.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
"""

# Replace the name of either snapshot or workload or restore with uuid/id
name_parse_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", name_to_id_system
        ),
        ("placeholder", "{messages}"),
    ]
)

input_with_id = name_parse_prompt | ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(
    [db_query_tool])


def handle_name_request(state: State):
    message = input_with_id.invoke(state)
    if verbose:
        print("handle_name_request", message)

    # Sometimes, the LLM will hallucinate and call the wrong tool.
    # We need to catch this and return an error message.
    tool_messages = []
    if message.tool_calls:
        for tc in message.tool_calls:
            if tc["name"] != "db_query_tool":
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: The wrong tool was called: {tc['name']}. "
                                 "Please fix your mistakes. Remember to only call "
                                 "SubmitFinalAnswer to submit the final answer. "
                                 "Generated queries should be outputted WITHOUT a tool call.",
                        tool_call_id=tc["id"],
                    )
                )
    else:
        tool_messages = []

    if verbose:
        print("handle_name_request", "tool_messages", tool_messages)
    return {"messages": [message] + tool_messages}


# Create a greeting agent to greet users when they login. The user can be new or existing users
greetings_system_msg = """
You are a helpful AI assistant, collaborating with other assistants.
 You are SQL expert with a strong attention to detail
 You greet the users and ask for their full name, including first and lastname. 
- if the user fail to provide first and lastnames generate a message to ask for them. Append STOP to the message.
- if you have user first and last names, you output a syntactically correct MySQL query to find the user in the Users table.
- If you get an error while executing a query, rewrite the query and try again.
- If you do not find the user Users table, generate a mesage to ask if the user would like to enroll. Append STOP to the message.
- If the user agrees to enroll, generate ENROLL_USER message
- If you find the user in Users table, generate DAILY_INTAKE
"""

# Create greetings agent
greetings_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", greetings_system_msg
        ),
        ("placeholder", "{messages}"),
    ]
)
greetings_agent = greetings_prompt | ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(
    [db_query_tool])


def greetings_node(state: State):
    message = greetings_agent.invoke(state)
    if verbose:
        print("greetings_node", message)

    # Sometimes, the LLM will hallucinate and call the wrong tool.
    # We need to catch this and return an error message.
    tool_messages = []
    if message.tool_calls:
        for tc in message.tool_calls:
            if tc["name"] != "db_query_tool":
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: The wrong tool was called: {tc['name']}. "
                                 "Please fix your mistakes. Remember to only call "
                                 "SubmitFinalAnswer to submit the final answer. "
                                 "Generated queries should be outputted WITHOUT a tool call.",
                        tool_call_id=tc["id"],
                    )
                )
    else:
        tool_messages = []

    if verbose:
        print("greetings_node", "tool_messages", tool_messages)
    return {"messages": [message] + tool_messages}


# Create a new user agent to to enroll user into the system
newuser_system_msg = """
You are a helpful AI assistant, collaborating with other assistants.
You are helping new user to enroll into the program and you prompt the user to
enter relavant information so you can fill the user record in Users table.
Please ask all relavant questions so all columns of the user in Users table can be
filled correctly. Once the user record is successfully created, generate message DAILY_INTAKE to delegate the
request to  daily dietary recording agent so the user daily intake can be recorded
"""

# Create newuser agent
newuser_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", newuser_system_msg
        ),
        ("placeholder", "{messages}"),
    ]
)
newuser_agent = newuser_prompt | ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(
    [db_query_tool])


def newuser_node(state: State):
    message = newuser_agent.invoke(state)
    if verbose:
        print("newuser_node", message)

    # Sometimes, the LLM will hallucinate and call the wrong tool.
    # We need to catch this and return an error message.
    tool_messages = []
    if message.tool_calls:
        for tc in message.tool_calls:
            if tc["name"] != "db_query_tool":
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: The wrong tool was called: {tc['name']}. "
                                 "Please fix your mistakes. Remember to only call "
                                 "SubmitFinalAnswer to submit the final answer. "
                                 "Generated queries should be outputted WITHOUT a tool call.",
                        tool_call_id=tc["id"],
                    )
                )
    else:
        tool_messages = []

    if verbose:
        print("newuser_node", "tool_messages", tool_messages)
    return {"messages": [message] + tool_messages}


# Describe a tool to represent the end state
class SubmitFinalAnswer(BaseModel):
    """Submit the final answer to the user based on the query results."""

    final_answer: str = Field(..., description="The final answer to the user")


# Create an agent that records his/her daily intake
daily_intake_system_msg = """
You are a helpful AI assistant, collaborating with other assistants.
You are helping the user to input his daily intake and populate the
Diet Responses Table. Encourage user to input as many responses as possible
until user says he is done with the inputs for the day.
Call SubmitFinalAnswer to submit the final answer.
"""

# Create newuser agent
daily_intake_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", daily_intake_system_msg
        ),
        ("placeholder", "{messages}"),
    ]
)
daily_intake_agent = daily_intake_prompt | ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(
    [db_query_tool, SubmitFinalAnswer])


def daily_intake_node(state: State):
    message = daily_intake_agent.invoke(state)
    if verbose:
        print("daily_intake_node", message)

    # Sometimes, the LLM will hallucinate and call the wrong tool.
    # We need to catch this and return an error message.
    tool_messages = []
    if message.tool_calls:
        for tc in message.tool_calls:
            if tc["name"] != "db_query_tool":
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: The wrong tool was called: {tc['name']}. "
                                 "Please fix your mistakes. Remember to only call "
                                 "SubmitFinalAnswer to submit the final answer. "
                                 "Generated queries should be outputted WITHOUT a tool call.",
                        tool_call_id=tc["id"],
                    )
                )
    else:
        tool_messages = []

    if verbose:
        print("daily_intake_node", "tool_messages", tool_messages)
    return {"messages": [message] + tool_messages}


workflow.add_node("daily_intake_node", daily_intake_node)
workflow.add_node("newuser_node", newuser_node)
workflow.add_node("greetings_node", greetings_node)

# Add a node for the model to check the query before executing it
workflow.add_node("correct_query_greetings", model_check_query)
workflow.add_node("correct_query_newuser", model_check_query)
workflow.add_node("correct_query_dailyintake", model_check_query)

# Add node for executing the query
workflow.add_node("execute_query_greetings", create_tool_node_with_fallback([db_query_tool]))
workflow.add_node("execute_query_newuser", create_tool_node_with_fallback([db_query_tool]))
workflow.add_node("execute_query_dailyintake", create_tool_node_with_fallback([db_query_tool]))


# Define a conditional edge to decide whether to continue or end the workflow
def should_continue_greetings_node(state: State) -> Literal[END, "correct_query_greetings", "greetings_node"]:
    messages = state["messages"]
    last_message = messages[-1]

    if verbose:
        print("should_continue_greetings_node", last_message)

    # If there is a tool call, then we finish
    if last_message.content.startswith("Error:"):
        return "greetings_node"
    if getattr(last_message, "tool_calls", None):
        return "correct_query_greetings"
    else:
        return END


# Define a conditional edge to decide whether to continue or end the workflow
def should_continue_newuser_node(state: State) -> Literal[END, "correct_query_newuser", "newuser_node"]:
    messages = state["messages"]
    last_message = messages[-1]

    if verbose:
        print("should_continue_newuser_node", last_message)

    # If there is a tool call, then we finish
    if getattr(last_message, "tool_calls", None):
        return END
    if last_message.content.startswith("Error:"):
        return "newuser_node"
    else:
        return "correct_query_newuser"

# Define a conditional edge to decide whether to continue or end the workflow
def should_continue_daily_intake_node(state: State) -> Literal[END, "correct_query_dailyintake", "daily_intake_node"]:
    messages = state["messages"]
    last_message = messages[-1]

    if verbose:
        print("should_continue_daily_intake_node", last_message)

    # If there is a tool call, then we finish
    if getattr(last_message, "tool_calls", None):
        return END
    if last_message.content.startswith("Error:"):
        return "daily_intake_node"
    else:
        return "correct_query_dailyintake"

# Add a node for the model to check the query before executing it
#workflow.add_node("replace_name_with_id", handle_name_request)

# Add node for executing the query
#workflow.add_node("execute_query_to_replace_name", create_tool_node_with_fallback([

# Specify the edges between the nodes
workflow.add_edge(START, "first_tool_call")
workflow.add_edge("first_tool_call", "list_tables_tool")
workflow.add_edge("list_tables_tool", "model_get_schema")
workflow.add_edge("model_get_schema", "get_schema_tool")
workflow.add_edge("get_schema_tool", "greetings_node")
#workflow.add_edge( "greetings_node", "newuser_node")
#workflow.add_edge("newuser_node", "daily_intake_node")
#workflow.add_edge("greetings_node", "daily_intake_node")
workflow.add_edge("daily_intake_node", END)

#workflow.add_edge("replace_name_with_id", "execute_query_to_replace_name")
#workflow.add_edge("execute_query_to_replace_name", "query_gen")
workflow.add_conditional_edges(
    "daily_intake_node",
    should_continue_daily_intake_node,
)

workflow.add_conditional_edges(
    "greetings_node",
    should_continue_greetings_node,
)

workflow.add_conditional_edges(
    "newuser_node",
    should_continue_newuser_node,
)
workflow.add_edge("correct_query_greetings", "execute_query_greetings")
workflow.add_edge("execute_query_greetings", "greetings_node")

workflow.add_edge("correct_query_newuser", "execute_query_newuser")
workflow.add_edge("execute_query_newuser", "newuser_node")

workflow.add_edge("correct_query_dailyintake", "execute_query_dailyintake")
workflow.add_edge("execute_query_dailyintake", "daily_intake_node")

# Compile the workflow into a runnable
app = workflow.compile(checkpointer=memory)

if standalone:
    messages = app.invoke(
            {"messages": [("user", "Hi, My name is Murali Balcha?")]},
            config
            )
    json_str = messages["messages"][-1].content
    print(json_str)

else:
     import streamlit as st
     x = """
     with st.sidebar:
         openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
         "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
         "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
         "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"
     """

     st.title("💬 ACML")
     st.caption("🚀 A Chatbot powered by OpenAI")
     if "messages" not in st.session_state:
         st.session_state["messages"] = [{"role": "assistant", "content": "Welcome to ACML. Who am I speaking with?"}]

     for msg in st.session_state.messages:
         st.chat_message(msg["role"]).write(msg["content"])

     if prompt := st.chat_input():
         st.session_state.messages.append({"role": "user", "content": prompt})
         st.chat_message("user").write(prompt)
         prompt += ". Also generate follow up questions"
         messages = app.invoke(
                 {"messages": [("user", prompt)]},
                 config
                 )
         msg = messages["messages"][-1].content.strip("STOP")
         st.session_state.messages.append({"role": "assistant", "content": msg})
         st.chat_message("assistant").write(msg)
