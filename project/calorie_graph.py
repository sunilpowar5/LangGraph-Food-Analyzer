import os
import requests
from typing import Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv

from google import genai
from google.genai import types as genai_types
from langgraph.graph import START, StateGraph, END

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI   
from langgraph.prebuilt import create_react_agent

load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NUTRITIONIX_APP_ID = os.getenv("NUTRITIONIX_APP_ID")
NUTRITIONIX_API_KEY = os.getenv("NUTRITIONIX_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=GEMINI_API_KEY)

class CalorieState(TypedDict):
    image_bytes: bytes
    mime: str
    food_items: str
    result: Optional[str]

def Identify_foods(state: CalorieState):
    """Node function to identify food items from image."""
    part = genai_types.Part.from_bytes(data=state['image_bytes'], mime_type=state['mime'])
    prompt = """
                Identify the food items in this image and list them clearly
                example: if image has bananas you should list how many 
                bananas are there and their size etc which required to 
                find out the calories of the food items. Just list the
                food items.If the food items in image is not clear, suggest
                user to upload a image with clear food items.If image does not
                have a food item ask user to upload a food image and explain 
                the user why it is not a food image. 
            """
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[part, prompt]
    )
    
    foods = response.text
    return {'food_items': foods}

@tool(description="Fetch nutrition facts for the food item")
def nutritionix_fetching(query: str) -> dict:
    headers = {
        "x-app-id": NUTRITIONIX_APP_ID,
        "x-app-key": NUTRITIONIX_API_KEY,
        "Content-Type": "application/json",
    }
    resp = requests.post(
        "https://trackapi.nutritionix.com/v2/natural/nutrients",
        headers=headers,
        json={"query": query}
    )
    if resp.status_code == 200:
        return resp.json()
    else:
        return {"error": resp.text}

agent = create_react_agent(
    model=llm,
    tools=[nutritionix_fetching],
    prompt="""Get the calories and proteins for the given input 
            food from nutritionix website. Don't add extra information"""
)

def fetch_calories(state: CalorieState):
    """
    Node function to fetch calories using the agent.
    Takes state as parameter and uses food_items from state.
    """
    food_query = state['food_items']
    
    prompt = f"""list the calories and proteins for each food item and 
                finally give total calories and proteins of {food_query}
                 don't add extra information like how you got the results.
                retrive only the calories and proteins"""

    response = agent.invoke({
        'messages': [("human", prompt)]
    })
    
    final_response = response['messages'][-1].content
    return {'result': final_response}

# creating the graph
def create_calorie_graph():
    builder = StateGraph(CalorieState)
    
    # nodes
    builder.add_node("identify_foods", Identify_foods)
    builder.add_node("fetch_calories", fetch_calories)
    
    # edges
    builder.add_edge(START, "identify_foods")
    builder.add_edge("identify_foods", "fetch_calories")
    builder.add_edge("fetch_calories", END)
    
    return builder.compile()


calorie_graph = create_calorie_graph()
