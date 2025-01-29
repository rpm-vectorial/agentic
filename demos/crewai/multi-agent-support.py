# Warning control
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')



# Set and validate Azure OpenAI environment variables
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get environment variables with error checking
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

# Validate environment variables
if not endpoint:
    raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is not set. Please set it in your .env file")
if not deployment:
    raise ValueError("DEPLOYMENT_NAME environment variable is not set. Please set it in your .env file")
if not subscription_key:
    raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set. Please set it in your .env file")

# Import LangChain's ChatOpenAI
from langchain_openai import AzureChatOpenAI

# Initialize Azure OpenAI Service client through LangChain
llm = AzureChatOpenAI(
    openai_api_version="2024-05-01-preview",
    azure_deployment=deployment,
    azure_endpoint=endpoint,
    api_key=subscription_key,
    temperature=0.7,
)

# Importing libraries
from crewai import Agent, Task, Crew    


# Create the Senior Support Representative Agent
senior_support = Agent(
    role='Senior Support Representative',
    goal='To be the most friendly and helpful support representative in the team',
    backstory='''
        You are a dedicated support representative at crewAI (https://crewai.com).
        Your current assignment is to assist {customer}, who is a VIP client.
        Your responsibilities include:
        - Providing comprehensive, detailed answers
        - Never making assumptions about the customer's needs
        - Ensuring the highest quality of support
        - Going above and beyond to exceed customer expectations
    ''',
    verbose=True,
    allow_delegation=False,
    llm=llm  # Using the LangChain Azure OpenAI wrapper
)

# Creates a Support Quality Assurance Agent for crewAI support team
support_quality_assurance_agent = Agent(
	role="Support Quality Assurance Specialist",
	goal="Get recognition for providing the "
    "best support quality assurance in your team",
	backstory=(
		"You work at crewAI (https://crewai.com) and "
        "are now working with your team "
		"on a request from {customer} ensuring that "
        "the support representative is "
		"providing the best support possible.\n"
		"You need to make sure that the support representative "
        "is providing full"
		"complete answers, and make no assumptions."
	),
	verbose=True
)

# Import required tools
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool

# Create web search tools
# web_search = SerperDevTool() # For general web searches
website_search = WebsiteSearchTool(website_url="https://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/") # For searching specific websites

# Add tools to the agents
senior_support.tools = [website_search]
# support_quality_assurance_agent.tools = [website_search]


# Task to handle customer inquiries and provide comprehensive support
inquiry_resolution = Task(
    description=(
        "{customer} just reached out with a super important ask:\n"
        "{inquiry}\n\n"
        "{person} from {customer} is the one that reached out. "
        "Make sure to use everything you know "
        "to provide the best support possible. "  # Added missing space
        "You must strive to provide a complete "
        "and accurate response to the customer's inquiry."
    ),
    expected_output=(
        "A detailed, informative response to the "
        "customer's inquiry that addresses "
        "all aspects of their question.\n"
        "The response should include references "
        "to everything you used to find the answer, "
        "including external data or solutions. "
        "Ensure the answer is complete, "
        "leaving no questions unanswered, and maintain a helpful and friendly "
        "tone throughout."
    ),
    # Fixed indentation and referenced correct tools from above
    tools=[website_search],
    # Fixed to reference correct agent name defined above
    agent=senior_support,
)


# Task for quality assurance review of support responses
quality_assurance_review = Task(
    description=(
        "Review the response drafted by the Senior Support Representative for {customer}'s inquiry. "
        "Ensure that the answer is comprehensive, accurate, and adheres to the "
        "high-quality standards expected for customer support.\n"
        "Verify that all parts of the customer's inquiry "
        "have been addressed "
        "thoroughly, with a helpful and friendly tone.\n"
        "Check for references and sources used to "
        "find the information, "
        "ensuring the response is well-supported and "
        "leaves no questions unanswered."
    ),
    expected_output=(
        "A final, detailed, and informative response "
        "ready to be sent to the customer.\n"
        "This response should fully address the "
        "customer's inquiry, incorporating all "
        "relevant feedback and improvements.\n"
        "Don't be too formal, we are a chill and cool company "
        "but maintain a professional and friendly tone throughout."
    ),
    agent=support_quality_assurance_agent
)

# Create the crew
support_crew = Crew(
    agents=[senior_support, support_quality_assurance_agent],
    tasks=[inquiry_resolution, quality_assurance_review],
    verbose=True
)


inputs = {
    "customer": "AI Vectorial",
    "person": "Pradeep Menon",
    "inquiry": "I need help with setting up a Crew "
               "and kicking it off, specifically "
               "how can I add memory to my crew? "
               "Can you provide guidance?"
}
result = support_crew.kickoff(inputs=inputs)

print(result)