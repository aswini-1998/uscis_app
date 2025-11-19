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
            model="gemini-2.5-flash", # Using the flash-lite model
            retry_options=retry_config
        ),
        description="An agent that answers immigration questions based *only* on www.uscis.gov.",
        
        # --- HERE IS THE FIX ---
        instruction=(
            "You are a specialized assistant. Your **only** function is to answer questions based **only** on information from the `www.uscis.gov` website. "
            "**Do not, under any circumstances, answer any question from your own internal knowledge.** "
            
            "**Your process for EVERY prompt MUST be:**"
            "1.  Use the `Google Search` tool to find relevant information. "
            "2.  Analyze the search results. "
            "3.  You MUST filter out and **completely ignore** any result that is **not** from `https://www.uscis.gov`. "
            "4.  If, after filtering, there are **no relevant results from `https://www.uscis.gov`**, OR if the user's prompt is off-topic (like weather, geography, or travel to other countries), you **MUST** respond with the exact phrase: "
            "'Details not found in 'www.uscis.gov', check other sources!'"
            
            "**If, and only if, you find valid results from `https://www.uscis.gov`:**"
            "1.  Base your answer *strictly* and *only* on the snippets from those `https://www.uscis.gov` results. "
            "2.  After the answer, you MUST cite your sources in this exact format: "
            "\\nSources:"
            "\\n1. <full URL of the first source>"
            "\\n2. <full URL of the second source>"
        ),
        # --- END FIX ---
        
        tools=[google_search],
    )
    return root_agent
    
# --- 3. Define Async Helper Function (With Resource Cleanup) ---
async def get_agent_response_async(prompt: str):
    """
    Creates a *fresh* agent and runner (stateless), runs the prompt,
    and ensures the client connection is closed to prevent warnings.
    """
    print("✅ Creating new *stateless* Agent and Runner in async context.")
    
    agent = None
    try:
        # 1. Create a fresh agent *inside* this loop
        agent = create_agent()

        # 2. Create a fresh runner *inside* this loop
        runner = InMemoryRunner(agent=agent)
        
        # 3. Run the *new* prompt
        result_events = await runner.run_debug(prompt)
        
        # 4. Parse the response
        try:
            last_event = result_events[-1]
            content = last_event.content
            
            if content.parts and content.parts[0].text:
                return content.parts[0].text
            else:
                print("Error: Agent event was empty (content.parts is None or empty).")
                return "An error occurred: The agent returned an empty response."
            
        except Exception as e:
            print(f"Error parsing agent response: {e}")
            return f"An error occurred while parsing the agent's response: {e}"

    finally:
        # --- FIX: Explicitly close the client connection ---
        # This prevents the "Task was destroyed but it is pending" warnings
        # by ensuring the AsyncClient is properly closed before we exit.
        if agent and hasattr(agent.model, "client"):
            await agent.model.client.close()
            
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
