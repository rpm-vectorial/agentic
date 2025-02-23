from crewai import Agent
from duckduckgo_search import DDGS
from config import llm
from typing import List, Dict, Optional
from langchain.tools import Tool
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def search_web(query: str) -> List[Dict[str, str]]:
    """Search the web using DuckDuckGo."""
    search_results = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5)
            for result in results:
                search_results.append({
                    'title': result['title'],
                    'link': result['link'],
                    'snippet': result['body']
                })
        logger.info(f"Successfully retrieved {len(search_results)} results for query: {query}")
        return search_results
    except Exception as e:
        logger.error(f"Error during web search: {str(e)}")
        raise Exception(f"Failed to perform web search: {str(e)}")

def validate_results(results: List[Dict[str, str]]) -> bool:
    """Validate search results."""
    is_valid = len(results) > 0 and all(
        all(key in result for key in ['title', 'link', 'snippet'])
        for result in results
    )
    logger.info(f"Search results validation: {'passed' if is_valid else 'failed'}")
    return is_valid

def analyze_results(results: List[Dict[str, str]], context: Optional[str] = None) -> Dict[str, any]:
    """Analyze search results and extract key insights."""
    try:
        if not results:
            logger.warning("No results to analyze")
            return {"error": "No results provided for analysis"}

        themes = _extract_themes(results)
        summary = _create_summary(results, themes, context)
        
        logger.info("Successfully analyzed search results")
        return {
            "themes": themes,
            "summary": summary,
            "source_count": len(results)
        }
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise Exception(f"Failed to analyze results: {str(e)}")

def _extract_themes(results: List[Dict[str, str]]) -> List[str]:
    """Extract main themes from search results using LLM."""
    try:
        combined_text = "\n".join([result['snippet'] for result in results if 'snippet' in result])
        
        prompt = f"""Analyze the following text and identify the main themes or key topics discussed.
        Focus on extracting 3-5 major themes.
        
        Text:
        {combined_text}
        
        Return only the themes as a comma-separated list."""
        
        response = llm.invoke(prompt)
        themes = [theme.strip() for theme in response.split(',')]
        return themes
    except Exception as e:
        logger.error(f"Error in theme extraction: {str(e)}")
        return []

def _create_summary(results: List[Dict[str, str]], themes: List[str], context: Optional[str]) -> str:
    """Create a comprehensive summary using LLM."""
    try:
        combined_text = "\n\n".join([f"Source: {result['title']}\n{result['snippet']}" 
                                   for result in results if 'snippet' in result])
        
        themes_text = ", ".join(themes) if themes else "No specific themes identified"
        context_text = f"\nAdditional context: {context}" if context else ""
        
        prompt = f"""Based on the following sources and identified themes, create a comprehensive summary.
        
        Themes: {themes_text}
        {context_text}
        
        Sources:
        {combined_text}
        
        Create a well-structured summary that:
        1. Synthesizes the main points
        2. Highlights key findings
        3. Maintains objectivity
        4. Cites multiple sources when possible"""
        
        return llm.invoke(prompt)
    except Exception as e:
        logger.error(f"Error in summary creation: {str(e)}")
        return "Failed to create summary due to an error."

def create_search_agent() -> Agent:
    """Create and configure a search agent."""
    search_tool = Tool(
        name="web_search",
        func=search_web,
        description="Search the web for information"
    )
    validate_tool = Tool(
        name="validate_results",
        func=validate_results,
        description="Validate search results"
    )
    
    return Agent(
        role='Search Expert',
        goal='Search the web for accurate and relevant information',
        backstory='An expert at web searching with keen eye for reliable sources',
        llm=llm,
        tools=[search_tool, validate_tool],
        verbose=True
    )

def create_reasoning_agent() -> Agent:
    """Create and configure a reasoning agent."""
    analysis_tool = Tool(
        name="analyze_results",
        func=analyze_results,
        description="Analyze and synthesize search results"
    )
    
    return Agent(
        role='Research Analyst',
        goal='Analyze and synthesize information from search results',
        backstory='''An analytical expert who excels at understanding and summarizing complex information.
                    Skilled in identifying key insights and presenting them in a clear, concise manner.''',
        llm=llm,
        tools=[analysis_tool],
        verbose=True
    )