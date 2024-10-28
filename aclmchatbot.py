from  datetime import datetime
import os # read API key from environment variables. Not required if you are specifying the key in notebook.
import re
import json # used to create a json to store snippets and embeddings
import numpy as np
from numpy import dot # used to match user questions with snippets.

from typing import Annotated, Literal, Optional

from typing_extensions import TypedDict

from langchain_community.utilities import SQLDatabase
from langgraph.graph.message import AnyMessage, add_messages

from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.runnables import RunnableLambda

from langgraph.prebuilt import ToolNode
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import tool

from pydantic import BaseModel, Field
from typing import Callable

from PyPDF2 import PdfReader # used to extract text from pdf
from langchain.text_splitter import CharacterTextSplitter # split text in smaller snippets

import openai
from openai import OpenAI # used to access openai api

standalone = True

##################################################################
#  Create embedding for questionaire                             #
##################################################################

"""### Parameters specifying different variables used in the code"""

EXTRACTED_TEXT_FILE_PATH = "pdf_text.txt" # text extracted from pdf
EXTRACTED_JSON_PATH = "extracted.json" # snippets and embeddings
OPENAI_API_KEY = os.environ['OPENAI_API_KEY'] # replace this with your openai api key or store the api key in env
EMBEDDING_MODEL = "text-embedding-ada-002" # embedding model used
GPT_MODEL = "gpt-4-turbo-preview" # gpt model used. alternatively you can use gpt-4 or other models.
CHUNK_SIZE = 1000 # chunk size to create snippets
CHUNK_OVERLAP = 200 # check size to create overlap between snippets
CONFIDENCE_SCORE = 0.75 # specify confidence score to filter search results. [0,1] prefered: 0.75
PDF_FILE_PATH = "ACLM-Diet-Screener-V1 in color.pdf"

def extract_text_from_pdf(pdf_file_path: str):

    # Open the PDF file using the specified file_path
    reader = PdfReader(pdf_file_path)
    # Get the total number of pages in the PDF
    number_of_pages = len(reader.pages)

    # Initialize an empty string to store extracted text
    pdf_text = ""

    # Loop through each page of the PDF
    for i in range(number_of_pages):
        # Get the i-th page
        page = reader.pages[i]
        # Extract text from the page and append it to pdf_text
        pdf_text += page.extract_text()
        # Add a newline after each page's text for readability
        pdf_text += "\n"

    # Specify the file path for the new text file
    file_path = EXTRACTED_TEXT_FILE_PATH

    # Write the content to the text file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(pdf_text)

    return pdf_text


docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", extract_text_from_pdf(PDF_FILE_PATH))]


class VectorStoreRetriever:
    def __init__(self, docs: list, vectors: list, oai_client):
        self._arr = np.array(vectors)
        self._docs = docs
        self._client = oai_client

    @classmethod
    def from_docs(cls, docs, oai_client):
        embeddings = oai_client.embeddings.create(
            model="text-embedding-3-small", input=[doc["page_content"] for doc in docs]
        )
        vectors = [emb.embedding for emb in embeddings.data]
        return cls(docs, vectors, oai_client)

    def query(self, query: str, k: int = 5) -> list[dict]:
        embed = self._client.embeddings.create(
            model="text-embedding-3-small", input=[query]
        )
        # "@" is just a matrix multiplication in python
        scores = np.array(embed.data[0].embedding) @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        return [
            {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
        ]


retriever = VectorStoreRetriever.from_docs(docs, openai.Client())


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


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


db = SQLDatabase.from_uri("mysql://root@localhost/aclm")

toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(model=GPT_MODEL))
tools = toolkit.get_tools()

list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

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


@tool
def lookup_questionaire(query: str) -> str:
    """Consult the ACLM questionaire that a newly enrolled must fill in.
    The user enrollment is not completed successfully until all the questions in the
    questionaire are answered"""
    docs = retriever.query(query, k=2)
    return "\n\n".join([doc["page_content"] for doc in docs])


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    dialog_state: Annotated[
        list[
            Literal[
                "assistant",
                "new_user",
                "daily_intake",
            ]
        ],
        update_dialog_stack,
    ]


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

    cancel: bool = True
    reason: str

    class Config:
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information.",
            },
        }


llm = ChatOpenAI(model=GPT_MODEL)
# Flight booking assistant

new_user_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for creating new user. "
            "The primary assistant delegates work to you whenever the user needs to be enrolled into the system. "
            "Ask any information need to enroll the user including name, email id, dietary restrictions as defined in the User table schema. "
            "If the user changes their mind, escalate the task back to the main assistant."
            " Remember that the user enrollment isn't completed until after the relevant tool has successfully been used."
            "\n\nIf the user needs help, and none of your tools are appropriate for it, then"
            ' "CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.',
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

new_user_safe_tools = [db_query_tool]
new_user_sensitive_tools = [db_query_tool]
new_user_tools = new_user_safe_tools + new_user_sensitive_tools
new_user_runnable = new_user_prompt | llm.bind_tools(
    new_user_tools + [CompleteOrEscalate]
)

# Hotel Booking Assistant
dietary_intake_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for handling user daily daiatery intake. You will encourage the user to "
            "fill in all the questions in ACLM questionaire. Since remembering last four weeks of diet is hard "
            "you will encourage user to fill in most common food that user consumes daily and then ask for any exceptions "
            "including attending parties, going to a restaurant and other ways people consume food."
            "The primary assistant delegates work to you whenever the user needs record his daily dietery intake. "
            "When asking user, Be persistent. Encourage user to enter as much accurate information as possible. "
            "If the user changes their mind, escalate the task back to the main assistant."
            " Remember that recording dietary intake isn't completed until after the relevant tool has successfully been used."
            '\n\nIf the user needs help, and none of your tools are appropriate for it, then "CompleteOrEscalate" the dialog to the host assistant.'
            " Do not waste the user's time. Do not make up invalid tools or functions."
            "\n\nSome examples for which you should CompleteOrEscalate:\n"
            " - 'nevermind i think I'll fill them later'\n"
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

dietary_intake_safe_tools = [db_query_tool]
dietary_intake_sensitive_tools = [db_query_tool]
dietary_intake_tools = dietary_intake_safe_tools + dietary_intake_sensitive_tools
dietary_intake_runnable = dietary_intake_prompt | llm.bind_tools(
    dietary_intake_tools + [CompleteOrEscalate]
)


# Primary Assistant
class ToNewUserAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle new user enrollment."""

    request: str = Field(
        description="Any necessary followup questions the new user assistant should clarify before proceeding."
    )


class ToRecordDietaryIntake(BaseModel):
    """Transfers work to a specialized assistant to record daily dietary intake."""

    start_date: str = Field(description="The start date of the car rental.")
    end_date: str = Field(description="The end date of the car rental.")
    request: str = Field(
        description="Any additional information or requests from the user regarding the car rental."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "location": "Basel",
                "start_date": "2023-07-01",
                "end_date": "2023-07-05",
                "request": "I need a compact car with automatic transmission.",
            }
        }


# The top-level assistant performs general Q&A and delegates specialized tasks to other assistants.
# The task delegation is a simple form of semantic routing / does simple intent detection
# llm = ChatAnthropic(model="claude-3-haiku-20240307")
# llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for American College of Lifestyle Medicine. "
            "Your primary role is to help answer users basic questions about their diet, exising medical conditions and any dietart restrictions associated with it. "
            "You are also a SQL expert with a strong attention to detail"
            "You greet the users and ask for their full name, including first and lastname. "
                "- if the user fail to provide first and lastnames generate a message to ask for them."
                "- if you have user first and last names, you output a syntactically correct MySQL query to find the user in the Users table."
                "- If you get an error while executing a query, rewrite the query and try again."
                "- If you do not find the user Users table, generate a mesage to ask if the user would like to enroll."
            "If a user wish to enroll into the system, "
            "delegate the task to the appropriate specialized assistant by invoking the corresponding tool. You are not able to make these types of changes yourself."
            " Only the specialized assistants are given permission to do this for the user."
            "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
            "Provide detailed information to the user, and always double-check the database before concluding that information is unavailable. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "User enrollment is not complete until the user fills in their dietary intake for the four weeks so you will delegate the request to a assistant who "
            "can guide the user into filling the questionaire"

        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

primary_assistant_tools = [
    TavilySearchResults(max_results=1),
    db_query_tool,
]
assistant_runnable = primary_assistant_prompt | llm.bind_tools(
    primary_assistant_tools
    + [
        ToNewUserAssistant,
        ToRecordDietaryIntake,
    ]
)



def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                    " and the booking, update, other other action is not complete until after you have successfully invoked the appropriate tool."
                    " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition

builder = StateGraph(State)


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


builder.add_node("first_tool_call", first_tool_call)

# Add nodes for the first two tools
builder.add_node(
    "list_tables_tool", create_tool_node_with_fallback([list_tables_tool])
)
builder.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))

# Add a node for a model to choose the relevant tables based on the question and available tables
model_get_schema = ChatOpenAI(model=GPT_MODEL, temperature=0).bind_tools(
    [get_schema_tool]
)
builder.add_node(
    "model_get_schema",
    lambda state: {
        "messages": [model_get_schema.invoke(state["messages"])],
    },
)

builder.add_edge(START, "first_tool_call")

# Flight booking assistant
builder.add_node(
    "enter_new_user",
    create_entry_node("Enter New User Info", "new_user"),
)
builder.add_node("new_user", Assistant(new_user_runnable))
builder.add_edge("enter_new_user", "new_user")
builder.add_node(
    "new_user_sensitive_tools",
    create_tool_node_with_fallback(new_user_sensitive_tools),
)
builder.add_node(
    "new_user_safe_tools",
    create_tool_node_with_fallback(new_user_safe_tools),
)


def route_new_user(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    safe_toolnames = [t.name for t in new_user_safe_tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        return "new_user_safe_tools"
    return "new_user_sensitive_tools"


builder.add_edge("new_user_sensitive_tools", "new_user")
builder.add_edge("new_user_safe_tools", "new_user")
builder.add_conditional_edges(
    "new_user",
    route_new_user,
    ["new_user_sensitive_tools", "new_user_safe_tools", "leave_skill", END],
)


# This node will be shared for exiting all specialized assistants
def pop_dialog_state(state: State) -> dict:
    """Pop the dialog stack and return to the main assistant.

    This lets the full graph explicitly track the dialog flow and delegate control
    to specific sub-graphs.
    """
    messages = []
    if state["messages"][-1].tool_calls:
        # Note: Doesn't currently handle the edge case where the llm performs parallel tool calls
        messages.append(
            ToolMessage(
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "dialog_state": "pop",
        "messages": messages,
    }


builder.add_node("leave_skill", pop_dialog_state)
builder.add_edge("leave_skill", "primary_assistant")

# Car rental assistant

builder.add_node(
    "enter_dietary_intake",
    create_entry_node("Ditary Intake Assitant", "dietary_intake"),
)
builder.add_node("dietary_intake", Assistant(dietary_intake_runnable))
builder.add_edge("enter_dietary_intake", "dietary_intake")
builder.add_node(
    "dietary_intake_safe_tools",
    create_tool_node_with_fallback(dietary_intake_safe_tools),
)
builder.add_node(
    "dietary_intake_sensitive_tools",
    create_tool_node_with_fallback(dietary_intake_sensitive_tools),
)


def route_dietary_intake(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    safe_toolnames = [t.name for t in dietary_intake_safe_tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        return "dietary_intake_safe_tools"
    return "dietary_intake_sensitive_tools"


builder.add_edge("dietary_intake_sensitive_tools", "dietary_intake")
builder.add_edge("dietary_intake_safe_tools", "dietary_intake")
builder.add_conditional_edges(
    "dietary_intake",
    route_dietary_intake,
    [
        "dietary_intake_safe_tools",
        "dietary_intake_sensitive_tools",
        "leave_skill",
        END,
    ],
)


# Primary assistant
builder.add_node("primary_assistant", Assistant(assistant_runnable))
builder.add_node(
    "primary_assistant_tools", create_tool_node_with_fallback(primary_assistant_tools)
)


def route_primary_assistant(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToNewUserAssistant.__name__:
            return "enter_new_user"
        elif tool_calls[0]["name"] == ToRecordDietaryIntake.__name__:
            return "enter_dietary_intake"
        return "primary_assistant_tools"
    raise ValueError("Invalid route")


# The assistant can route to one of the delegated assistants,
# directly use a tool, or directly respond to the user
builder.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant,
    [
        "enter_new_user",
        "enter_dietary_intake",
        "primary_assistant_tools",
        END,
    ],
)
builder.add_edge("primary_assistant_tools", "primary_assistant")


# Each delegated workflow can directly respond to the user
# When the user responds, we want to return to the currently active workflow
def route_to_workflow(
    state: State,
) -> Literal[
    "primary_assistant",
    "new_user",
    "dietary_intake",
]:
    """If we are in a delegated state, route directly to the appropriate assistant."""
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return "primary_assistant"
    return dialog_state[-1]


builder.add_edge("first_tool_call", "list_tables_tool")
builder.add_edge("list_tables_tool", "model_get_schema")
builder.add_edge("model_get_schema", "get_schema_tool")
builder.add_conditional_edges("get_schema_tool", route_to_workflow)

# Compile graph
memory = MemorySaver()
part_4_graph = builder.compile(
    checkpointer=memory,
    # Let the user approve or deny the use of sensitive tools
    interrupt_before=[
        "new_user_sensitive_tools",
        "dietary_intake_sensitive_tools",
    ],
)

import shutil
import uuid

# Let's create an example conversation a user might have with the assistant
tutorial_questions1 = [
    "Hi there, I am Dan Doss",
    "Yes, Please",
    "My email is ddoss@gmail.com, I am 53 years old and Male",
    "I have hypertension condition and I am allegic to eggs",
]

tutorial_questions = [
    "Hi there, I am Dan Doss",
    "I would like to enter my today's dietary intake",
    "I had a bag of potato chips, 16oz soda, half a pound steak",
    "Thank you",
]

tutorial_questions = [
    "Hi there, I am Mike Rodolph",
    "Yes, Please",
    "My email is mrodolph@gmail.com, I am 53 years old and Male",
    "I have hypertension condition and I am allegic to eggs",
    "Thank You",
]
# Update with the backup file so we can restart from the original place in each section
thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight information
        "user_name": "Murali Balcha",
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}


_printed = set()
if standalone:
    # We can reuse the tutorial questions from part 1 to see how it does.
    #for question in tutorial_questions:
    question = input("I am your ACLM assistant, How may I help you today?")
    while not "bye" in question:
        events = part_4_graph.stream(
            {"messages": ("user", question)}, config, stream_mode="values"
        )
        for event in events:
            pass
            #_print_event(event, _printed)
        print(event['messages'][-1].content)
        snapshot = part_4_graph.get_state(config)
        while snapshot.next:
            # We have an interrupt! The agent is trying to use a tool, and the user can approve or deny it
            # Note: This code is all outside of your graph. Typically, you would stream the output to a UI.
            # Then, you would have the frontend trigger a new run via an API call when the user has provided input.
            try:
                user_input = input(
                    "Do you approve of the above actions? Type 'y' to continue;"
                    " otherwise, explain your requested changed.\n\n"
                )
            except:
                user_input = "y"
            if user_input.strip() == "y":
                # Just continue
                result = part_4_graph.invoke( None, config,)
            else:
                # Satisfy the tool invocation by
                # providing instructions on the requested changes / change of mind
                result = part_4_graph.invoke(
                    {
                        "messages": [
                            ToolMessage(
                                tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                                content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                            )
                        ]
                    },
                    config,
                )
            snapshot = part_4_graph.get_state(config)
        question = input("")
else:
     import streamlit as st
     x = """
     with st.sidebar:
         openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
         "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
         "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
         "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"
     """

     st.title("💬 ACLM")
     st.caption("🚀 A Chatbot powered by OpenAI")
     if "messages" not in st.session_state:
         st.session_state["messages"] = [{"role": "assistant", "content": "Welcome to ACML. Who am I speaking with?"}]

     for msg in st.session_state.messages:
         st.chat_message(msg["role"]).write(msg["content"])

     if prompt := st.chat_input():
         st.session_state.messages.append({"role": "user", "content": prompt})
         st.chat_message("user").write(prompt)
         #prompt += ". Also generate follow up questions"
         events = part_4_graph.stream(
            {"messages": ("user", prompt)}, config, stream_mode="values"
         )
         for event in events:
             _print_event(event, _printed)
         msg = event['messages'][-1].content
         st.session_state.messages.append({"role": "assistant", "content": msg})
         st.chat_message("assistant").write(msg)
