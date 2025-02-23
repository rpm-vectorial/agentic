from crewai import Crew, Task
from agents import create_search_agent, create_reasoning_agent

def create_research_crew(topic):
    # Initialize agents
    search_agent = create_search_agent()
    reasoning_agent = create_reasoning_agent()

    # Create tasks
    search_task = Task(
        description=f"Search the web for information about: {topic}",
        agent=search_agent
    )

    analysis_task = Task(
        description=f"Analyze and synthesize the search results about {topic} into a comprehensive summary",
        agent=reasoning_agent
    )

    # Create crew
    crew = Crew(
        agents=[search_agent, reasoning_agent],
        tasks=[search_task, analysis_task],
        verbose=True
    )

    return crew

def main():
    # Get user input
    topic = input("Enter a topic to research: ")
    
    # Create and run the crew
    crew = create_research_crew(topic)
    result = crew.kickoff()
    
    print("\nFinal Analysis:")
    print(result)

if __name__ == "__main__":
    main()