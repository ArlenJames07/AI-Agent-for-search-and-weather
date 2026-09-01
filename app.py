import os

import certifi
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch


# ==========================================
# ENVIRONMENT
# ==========================================

os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# ==========================================
# STREAMLIT PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Agentic AI Assistant")
st.markdown(
    "Real-time Search + Weather AI Agent using "
    "LangChain, Gemini and Tavily"
)


# ==========================================
# VALIDATE ENVIRONMENT VARIABLES
# ==========================================

required_keys = {
    "GOOGLE_API_KEY": GOOGLE_API_KEY,
    "WEATHERSTACK_API_KEY": WEATHERSTACK_API_KEY,
    "TAVILY_API_KEY": TAVILY_API_KEY,
}

missing_keys = [
    key
    for key, value in required_keys.items()
    if not value
]

if missing_keys:
    st.error(
        "Missing environment variables: "
        + ", ".join(missing_keys)
    )
    st.stop()


# ==========================================
# SEARCH TOOL
# ==========================================

search_tool = TavilySearch(
    max_results=5,
    search_depth="basic",
)


# ==========================================
# WEATHER TOOL
# ==========================================

@tool
def get_weather_data(city: str) -> str:
    """
    Get the current weather for a city.

    Use this tool whenever the user asks about
    current temperature, humidity, or weather
    conditions.
    """

    url = "http://api.weatherstack.com/current"

    params = {
        "access_key": WEATHERSTACK_API_KEY,
        "query": city,
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as exc:
        return f"Weather API request failed: {exc}"

    except ValueError:
        return "Weather API returned invalid JSON."

    if "current" not in data:

        error = data.get(
            "error",
            {},
        ).get(
            "info",
            "Unknown Weatherstack error",
        )

        return (
            f"Could not fetch weather data "
            f"for {city}: {error}"
        )

    current = data["current"]

    descriptions = current.get(
        "weather_descriptions",
        [],
    )

    description = (
        descriptions[0]
        if descriptions
        else "Unknown"
    )

    location = data.get("location", {})

    return (
        f"Location: "
        f"{location.get('name', city)}, "
        f"{location.get('country', '')}\n"
        f"Temperature: "
        f"{current.get('temperature')} °C\n"
        f"Feels like: "
        f"{current.get('feelslike')} °C\n"
        f"Weather: {description}\n"
        f"Humidity: "
        f"{current.get('humidity')}%\n"
        f"Wind speed: "
        f"{current.get('wind_speed')} km/h"
    )


# ==========================================
# TOOLS
# ==========================================

tools = [
    search_tool,
    get_weather_data,
]


# ==========================================
# GEMINI
# ==========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    thinking_level="low",
    max_retries=2,
)


# ==========================================
# SYSTEM PROMPT
# ==========================================

SYSTEM_PROMPT = """
You are an AI research assistant with access to
real-time web search and weather information.

Rules:

1. For current events, news, politics, wars,
   markets, sports, scientific developments,
   recent events, or anything time-sensitive,
   use Tavily search before answering.

2. Never claim that information is current unless
   you have searched for it.

3. When the user asks about current weather,
   temperature, humidity, or weather conditions,
   use the weather tool.

4. You may use multiple tools when necessary.

5. When using web search, distinguish facts from
   speculation.

6. Prefer recent and reputable sources.

7. If different sources disagree, mention the
   disagreement instead of pretending there is
   certainty.

8. Give concise but informative answers.
"""


# ==========================================
# CREATE AGENT
# ==========================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)


# ==========================================
# UI
# ==========================================

user_query = st.text_input(
    "Enter your query:",
    placeholder=(
        "Example: What is the latest news "
        "about Iran and the USA?"
    ),
)


# ==========================================
# RUN AGENT
# ==========================================

if st.button("Run Agent"):

    if not user_query:
        st.warning("Please enter a query.")

    else:

        with st.spinner("Agent is thinking..."):

            try:

                response = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": user_query,
                            }
                        ]
                    }
                )

                final_message = response[
                    "messages"
                ][-1]

                st.success(
                    "Response Generated"
                )

                st.markdown(
                    "## Final Response"
                )

                # Gemini 3 returns structured
                # content blocks.
                st.write(final_message.text)

            except Exception as exc:

                st.error(
                    f"Error: {exc}"
                )