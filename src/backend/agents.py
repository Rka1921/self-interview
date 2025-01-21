from crewai import Agent, LLM, Crew, Task
# from custom_analysis_tool import CustomAnalysisTool
from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()

# Set environment variables for Groq
os.environ["GEMINI_API_KEY"] = "AIzaSyAl5Mskw4aX30olESCr_ddPg7rA4_E43jU"

# Initialize the custom analysis tool
# custom_analysis_tool = CustomAnalysisTool()

# Define the agent
fraud_transaction_detector = Agent(
    role='your are a chat agent who will chat with user',
    goal='chat with user',
    backstory="""chat with user normally""",
    verbose=True,
    llm=LLM(
        model="gemini/gemini-1.5-flash",
        temperature=0.5
    )
)

# Define the task with expected_output
task = Task(
    name="Fraud Detection",
    # context=dummy_transactions,  # Pass transactions as a list of dictionaries
    description="Detect and flag potentially fraudulent transactions.",
    agent=fraud_transaction_detector,
    expected_output="A JSON string indicating flagged transactions and reasons."
)

# Initialize the Crew with the task and agent
crew = Crew(
    agents=[fraud_transaction_detector],  # Ensure the agent is passed here
    tasks=[task],                         # Link task to the crew
    verbose=True,
    llm=LLM(
        model="gemini/gemini-1.5-flash",
        temperature=0.5
    )
)

def main():
    # Kick off the Crew
    result = crew.kickoff()
    print("Analysis Results:")
    print(result)

if __name__ == "__main__":
    main()