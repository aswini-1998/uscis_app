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
# This patches asyncio to allow nested event loops.
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
def create_agent():
    print("✅ Creating new agent definition...")
    
    retry_config=types.HttpRetryOptions(
        attempts=5,
        exp_base=7,
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504]
    )
    
    root_agent = Agent(
        name="uscis_assistant",
        model=Gemini(
            model="gemini-2.5-flash",
            retry_options=retry_config
        ),
        description="An agent that answers immigration questions based *only* on www.uscis.gov.",
        instruction=(
            "You are a helpful assistant for US immigration questions. "
            "You MUST use the `Google Search` tool to answer ALL user questions. "
            "After you get the search results, you MUST filter them. "
            "You MUST ONLY use information from search results where the URL **starts with** `https://www.uscis.gov`. "
            "You MUST IGNORE any and all search results from other websites like boundless.com, citizenpath.com, etc. "
            "Base your answer *strictly* and *only* on the snippets from the `https.www.uscis.gov` results. "
            "Do not use any outside knowledge. "
            "After providing the answer, you MUST cite all the sources you used. "
            "You MUST list the URLs of the sources you used (which MUST be from `https://www.uscis.gov`) in a ranked list, ordered by relevance. "
            "You MUST format the citations *exactly* like this: "
            "\\nSources:"
            "\\n1. <full URL of the first source>"
            "\\n2. <full URL of the second source>"
            "If the search results are empty, or if there are no search results from `https://www.uscis.gov`, or if the `www.uscis.gov` snippets do not contain a clear answer, "
            "you MUST respond with the exact phrase: "
            "'Details not found in website 'www.uscis.gov', check other sources!'"
        ),
        tools=[google_search],
    )
    return root_agent

# --- 3. Define Async Helper Function ---
async def get_agent_response_async(prompt: str):
    """
    Creates a *fresh* agent and runner (stateless)
    and runs the prompt.
    """
    print("✅ Creating new *stateless* Agent and Runner in async context.")
    
    # 1. Create a fresh agent *inside* this loop
    agent = create_agent()

    # 2. Create a fresh runner *inside* this loop
    runner = InMemoryRunner(agent=agent)
    
    # 3. Run the *new* prompt
    result_events = await runner.run_debug(prompt)
    
    # --- 4. THE FINAL PARSING FIX ---
    # The last event is the one we want. Its .content is a complex
    # object (not a string), and we must get the text from its 'parts'.
    try:
        # Get the last event (the agent's reply)
        last_event = result_events[-1]
        
        # Access the content (which is a Content object)
        content = last_event.content
        
        # Safely extract the text from the first part
        if content.parts and content.parts[0].text:
            return content.parts[0].text
        else:
            # This happens if the agent's reply is empty
            print("Error: Agent event was empty (content.parts is None or empty).")
            return "An error occurred: The agent returned an empty response."
        
    except Exception as e:
        print(f"Error parsing agent response: {e}")
        return f"An error occurred while parsing the agent's response: {e}"

# --- 4. Define Synchronous Wrapper ---
def get_agent_response(prompt: str):
    """
    A synchronous wrapper to call the async function.
    """
    try:
        # Pass *only* the prompt
        return asyncio.run(get_agent_response_async(prompt))
    except Exception as e:
        print(f"Error running agent: {e}")
        return f"An error occurred while processing your request: {e}"

# --- 5. Build the Streamlit Chat UI ---
st.title("USCIS Assistant 🤖")
st.caption("I answer questions based *only* on the www.uscis.gov website.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is the I-90 form for?"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching www.uscis.gov..."):
            
            # Call with *only* the prompt
            response = get_agent_response(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
