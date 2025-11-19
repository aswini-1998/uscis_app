# app.py

import os
import streamlit as st
import asyncio
import requests

# --- ADK Imports ---
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
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

# --- 2. Define Custom Tool (True Tool Grounding) ---
def search_uscis(query: str) -> str:
    """
    Searches the official USCIS website (www.uscis.gov) for information.
    """
    # --- FIX: Relaxed filter (removed 'www') for better results ---
    # We also put the 'site:' operator at the front, which is sometimes more reliable.
    if "site:uscis.gov" not in query:
        query = f"site:uscis.gov {query}"
    
    print(f"🔎 DEBUG: Tool running with query: '{query}'") # Debug print
    
    try:
        api_key = os.environ["GOOGLE_API_KEY"]
        cse_id = os.environ["GOOGLE_CSE_ID"]
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": cse_id,
            "q": query
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        items = data.get("items", [])
        
        print(f"🔎 DEBUG: Google Search returned {len(items)} results.") # Debug print
        
        if not items:
            return "No results found on www.uscis.gov."
            
        # Format the results for the agent
        results_text = ""
        for item in items[:5]:
            results_text += f"Title: {item.get('title')}\n"
            results_text += f"Link: {item.get('link')}\n"
            results_text += f"Snippet: {item.get('snippet')}\n\n"
            
        return results_text

    except Exception as e:
        error_msg = f"Error searching USCIS: {str(e)}"
        print(f"❌ DEBUG: {error_msg}")
        return error_msg

# --- 3. Define the Agent ---
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
        instruction=(
            "You are a helpful assistant for US immigration questions. "
            "You MUST use the `search_uscis` tool to answer ALL user questions. "
            
            "**Base your answer *strictly* and *only* on the search results provided by the tool.** "
            "Do not use any outside knowledge. "
            
            "After providing the answer, you MUST cite all the sources you used. "
            "You MUST format the citations *exactly* like this: "
            "\\nSources:"
            "\\n1. <full URL of the first source>"
            "\\n2. <full URL of the second source>"
            
            "If the tool returns 'No results found', or if the snippets do not contain a clear answer, "
            "you MUST respond with the exact phrase: "
            "'Details not found in website 'www.uscis.gov', check other sources!'"
        ),
        tools=[search_uscis],
    )
    return root_agent

# --- 4. Define Async Helper Function ---
async def get_agent_response_async(prompt: str):
    """
    Creates a *fresh* agent and runner (stateless), runs the prompt,
    and ensures cleanup.
    """
    print("✅ Creating new *stateless* Agent and Runner in async context.")
    
    agent = None
    try:
        agent = create_agent()
        runner = InMemoryRunner(agent=agent)
        
        # We simply pass the prompt. The TOOL handles the filtering.
        result_events = await runner.run_debug(prompt)
        
        try:
            # Find the last event that is from the model
            for event in reversed(result_events):
                if hasattr(event, 'content') and event.content and event.content.parts:
                    text = event.content.parts[0].text
                    if text and not text.strip().startswith('{'): 
                         return text

            if result_events and result_events[-1].content.parts:
                 return result_events[-1].content.parts[0].text
            
            return "An error occurred: The agent returned an empty response."
            
        except Exception as e:
            print(f"Error parsing agent response: {e}")
            return f"An error occurred while parsing the agent's response: {e}"

    finally:
        if agent and hasattr(agent.model, "client"):
            await agent.model.client.close()

# --- 5. Define Synchronous Wrapper ---
def get_agent_response(prompt: str):
    try:
        return asyncio.run(get_agent_response_async(prompt))
    except Exception as e:
        print(f"Error running agent: {e}")
        return f"An error occurred while processing your request: {e}"

# --- 6. Build the Streamlit Chat UI ---
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