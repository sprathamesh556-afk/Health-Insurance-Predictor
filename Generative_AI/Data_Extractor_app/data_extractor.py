from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

# Ensure this model is valid on your Groq account
llm = ChatGroq(model="openai/gpt-oss-120b")

def extract_movie_data(article_text):
    prompt = '''
    From the below article, extract the movie details in a valid JSON output format.
    Use EXACTLY these keys: 
    'Actor Name', 'Director', 'Movie Title', 'Budget', 'Revenue', 'Studio', 'Release Year', 'Genre', 'IMDb Rating'.
    
    If a value is not found, return null for that key.
    Format the Budget and Revenue with the currency and unit if available (e.g., "200 million USD").
    Only return the valid JSON. No preamble, no markdown formatting (do not use json).

    Article:
    =========
    {article_text}
    '''
    
    pt = PromptTemplate.from_template(prompt)
    chain = pt | llm
    
    try:
        response = chain.invoke({'article_text': article_text})
        
        parser = JsonOutputParser()
        # Clean up the response just in case
        content = response.content.replace("json", "").replace("```", "").strip()
        
        res = parser.parse(content)
        return res
        
    except Exception as e:
        raise Exception(f"An error occurred during extraction: {e}")