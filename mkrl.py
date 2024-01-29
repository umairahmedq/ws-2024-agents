# Import the necessary libraries
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from a .env file
import os
assert os.environ.get("OPENAI_API_KEY") != None, "OPENAI_API_KEY not set"  # Ensure the OPENAI_API_KEY is set

import streamlit as st  # Import Streamlit for web app development
from langchain_experimental.tools import PythonREPLTool # Import a tool for running Python code and arXiv papers
import streamlit.components.v1 as components
import time
from langchain_community.document_loaders import ArxivLoader


from unstructured.partition.auto import partition  # For partitioning unstructured data
from langchain_community.document_loaders import UnstructuredFileLoader  # For loading unstructured documents
from langchain.chains import create_extraction_chain

import tempfile  # For creating temporary files
import shutil  # For high-level file operations

from langchain_openai import ChatOpenAI  # For using OpenAI's chat models
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, create_react_agent, load_tools
from langchain.chains import LLMChain  # For creating chains with language models
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import HumanInputRun
from langchain_community.callbacks import StreamlitCallbackHandler
import uuid
from langchain_core.prompts import PromptTemplate

# Initialize the language model with specific parameters
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# Define the tools to be used with the language model
tools = load_tools(['arxiv']) 
other_tools = [PythonREPLTool()]
# Define a template for the prompts
tools = tools + other_tools
template = open("pages/query.txt", "r").read()

# Convert the template into a PromptTemplate object
prompt = PromptTemplate.from_template(template)

# Function to save an uploaded file
def save_uploaded_file(uploaded_file):
    temp_directory = os.path.join(os.getcwd(), 'temp')
    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name, dir=temp_directory) as tmp_file:
            shutil.copyfileobj(uploaded_file, tmp_file)
            return tmp_file.name
    except Exception as e:
        st.error(f"Error occurred: {e}")
        return None

# Create the agent and executor for running the language model
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, max_iterations=30, handle_parsing_errors=True
)

# Streamlit code to create a web app interface
st.title("Cost manager")
uploaded_file = st.file_uploader("Choose a file...", type=["jpg", "png", "pdf",'xlsx','xls','docx','doc','txt','csv'])

if uploaded_file is not None:
    temp_file_path = save_uploaded_file(uploaded_file)
    if temp_file_path:
        st.success(f"Saved file to {temp_file_path}")
        loader = UnstructuredFileLoader(temp_file_path)
        docs = loader.load()
        doc = [inp.page_content for inp in docs][0]
        st.write(doc)
        query = """
        what is the date and name of the restaurant in the following. doc:
        :""" + doc

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor({"question": prompt, 
                                   'tool_names': [tool.name for tool in tools]
                                  }, return_only_outputs=True, callbacks=[st_callback])
        mkrl_response = response
        st.write(mkrl_response['output'])
