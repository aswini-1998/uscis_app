# app.py

import os
import streamlit as st
import asyncio

# --- ADK Imports ---
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

# --- NEST_ASYNCIO PATCH ---
# This is still a good idea, as it prevents other
# potential asyncio conflicts.
import nest_asyncio
nest_asyncio.apply()

# --- 1. Load Secrets and Set Environment Variables ---
try:
    os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]
    os.environ['GOOGLE_CSE_ID'] = st.secrets["GOOGLE_CSE_ID"]
except (KeyError, FileNotFoundError):
    st.error("Error: GOOGLE_API_KEY or GOOGLE_CSE_ID not found in st.secrets.")
    st.info("Please add your API keys to the .streamlit/secrets.toml file.")
    st.stop()

# --- 2. Define the Agent (No Caching) ---
# We removed @st.cache_resource. We must create a new
# agent *inside* the new event loop to avoid conflicts.
def create_agent():
    print("✅ Creating new agent definition...")
    
    retry_config=types.HttpRetryOptions(
        attempts=5,
        exp_base=7,
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504]
    )
    
    # This is your exact agent definition from the notebook
    root_agent = Agent(
        name="uscis_assistant",
        model=Gemini(
            model="gemini-2.5-pro", # Using the correct model
            retry_options=retry_config
        ),
        description="An agent that answers immigration questions based *only* on www.uscis.gov.",
        instruction=(
            "You are a helpful assistant for US immigration questions. "
            "You MUST use the `Google Search` tool to answer ALL user questions. "
            "For EVERY search, you MUST append 'site:www.uscis.gov' to the user's query. "
            "Base your answer *strictly* and *only* on the snippets provided by the search results from 'www.uscis.gov'. "
            "Do not use any outside knowledge. "
            "After providing the answer, you MUST cite all the sources you used. "
            "The search tool will provide a list of search results, each with a 'url'. "
            "You MUST list the URLs of the sources you used in a ranked list, ordered by relevance (i.e., the order they appeared in the search results). "
            "You MUST format the citations *exactly* like this: "
            "\\nSources:"
            "\\n1. <full URL of the first source>"
            "\\n2. <full URL of the second source>"
            "If the search results are empty, or if the snippets do not contain a clear answer, "
            "you MUST respond with the exact phrase: "
            "'Details not found in website 'www.uscis.gov', check other sources!'"
        ),
        tools=[google_search],
    )
    return root_agent

# --- 3. Define Async Helper Function ---

async def get_agent_response_async(prompt: str, chat_history: list):
    """
    Creates an agent and runner *inside* the async event loop,
    loads the history, and runs the agent.
    """
    print("✅ Creating new Agent and Runner in async context.")
    
    # 1. Create the agent *inside* this loop
    agent = create_agent()
    
    # 2. Create the runner *inside* this loop
    runner = InMemoryRunner(agent=agent)
    
    # 3. "Re-hydrate" the runner with the past conversation
    # We use the 'chat_history' (a simple list) from st.session_state
    for message in chat_history:
        # Use the simple .add_message() method
        runner.chat_history.add_message(
            role=message["role"], 
            content=message["content"]
        )

    # 4. Run the *new* prompt
    # run_debug will add the new prompt to the history and run the agent
    result_events = await runner.run_debug(prompt)
    
    # 5. Return the last event's content
    return result_events[-1].content.parts[0].text

# --- 4. Define Synchronous Wrapper ---

def get_agent_response(prompt: str, chat_history: list):
    """
    A synchronous wrapper to call the async function.
    We use asyncio.run() which creates a *new* event loop.
    This works now because the agent and runner are *also*
    created inside that new loop, so there is no conflict.
    """
    try:
        # Pass the prompt AND the history
        return asyncio.run(get_agent_response_async(prompt, chat_history))
    except Exception as e:
        print(f"Error running agent: {e}")
        return f"An error occurred while processing your request: {e}"

# --- 5. Build the Streamlit Chat UI ---

st.title("USCIS Assistant 🤖")
st.caption("I answer questions based *only* on the www.uscis.gov website.")

# Initialize chat history (st.session_state.messages)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get new user input
if prompt := st.chat_input("What is the I-90 form for?"):
    
    # Create a *copy* of the history to pass to the agent
    # We don't want the agent to get the *new* prompt twice
    history_to_pass = list(st.session_state.messages)
    
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the agent's response
    with st.chat_message("assistant"):
        with st.spinner("Searching www.uscis.gov..."):
            
            # Pass the prompt and the history *before* this prompt
            response = get_agent_response(prompt, history_to_pass)
            
            st.markdown(response)
    
    # Add assistant's response to session state
    st.session_state.messages.append({"role": "assistant", "content": response})