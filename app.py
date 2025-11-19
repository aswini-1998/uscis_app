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
            model="gemini-2.5-flash-lite", 
            retry_options=retry_config
        ),
        description="An agent that answers immigration questions based *only* on www.uscis.gov.",
        # --- FIX: FEW-SHOT PROMPTING FOR TOOL GROUNDING ---
        instruction=(
            "You are a helpful assistant for US immigration questions. "
            "You MUST use the `Google Search` tool to answer ALL user questions. "
            
            "**CRITICAL TOOL USAGE RULE:**"
            "When calling `Google Search`, you **MUST** include the `site:www.uscis.gov` filter inside the query argument. "
            "Do not rely on the user's prompt to have it. You must ensure it is there."
            
            "**EXAMPLES OF CORRECT TOOL USAGE:**"
            "User: 'Green card'"
            "✅ CORRECT Call: `Google Search(query='Green card site:www.uscis.gov')`"
            "❌ INCORRECT Call: `Google Search(query='Green card')`"
            
            "User: 'H-1B cap'"
            "✅ CORRECT Call: `Google Search(query='H-1B cap site:www.uscis.gov')`"
            
            "Base your answer *strictly* and *only* on the snippets provided by the search tool. "
            "Do not use any outside knowledge. "
            "After providing the answer, you MUST cite all the sources you used. "
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

# --- 3. Define Async Helper Function (With Resource Cleanup) ---
async def get_agent_response_async(prompt: str):
    """
    Creates a *fresh* agent and runner (stateless), runs the prompt,
    and ensures the client connection is closed.
    """
    print("✅ Creating new *stateless* Agent and Runner in async context.")
    
    agent = None
    try:
        # 1. Create a fresh agent
        agent = create_agent()

        # 2. Create a fresh runner
        runner = InMemoryRunner(agent=agent)
        
        # --- SEARCH QUERY INJECTION (Keep this as a safety net) ---
        restricted_prompt = f"{prompt} site:www.uscis.gov"
        
        # 3. Run the *restricted* prompt
        result_events = await runner.run_debug(restricted_prompt)
        
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
        # Explicitly close the client connection
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
            
            response = get_agent_response(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
