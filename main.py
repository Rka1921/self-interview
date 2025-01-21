from crewai import Agent, LLM, Crew, Task
from dotenv import load_dotenv
import os
import PyPDF2
import tempfile

# Load environment variables
load_dotenv()

# Set environment variables for Groq
os.environ["GEMINI_API_KEY"] = "AIzaSyBms_NwgHqoCzGUb9yyS7hOpazhtIcmj8U"

from crewai_tools import FileReadTool

def convert_pdf_to_temp_text(pdf_path):
    # Read PDF
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text_content = ""
        
        # Extract text from all pages
        for page in pdf_reader.pages:
            text_content += page.extract_text()
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt')
    temp_file.write(text_content)
    temp_file.close()
    
    return temp_file.name

# Convert PDF to text and get temp file path
pdf_path = './test.pdf'  # Update with your PDF path
temp_text_path = convert_pdf_to_temp_text(pdf_path)

# Use the temp text file with FileReadTool
file_read_tool = FileReadTool(file_path=temp_text_path)

interview_question_generator = Agent(
    role='Expert Technical Interviewer and CV Analyzer',
    goal='Generate relevant technical interview questions based on CV content',
    backstory="""You are an experienced technical interviewer who specializes in 
    creating targeted questions based on candidate's experience and skills. 
    Your questions should cover both technical expertise and practical experience.""",
    verbose=True,
    llm=LLM(
        model="gemini/gemini-1.5-flash",
        temperature=0.7
    ),
    tools=[file_read_tool]
)

task = Task(
    name="Interview Question Generation",
    description="""
    1. Read and analyze the provided CV
    2. Generate 10 technical interview questions based on the candidate's background
    3. Output must be in the following JSON format:
    {
        "questions": [
            {
                "id": 1,
                "question": "question text",
                "category": "technical/experience/project",
                "difficulty": "easy/medium/hard"
            },
            ...
        ]
    }""",
    agent=interview_question_generator,
    expected_output="A JSON object containing 10 structured interview questions"
)

crew = Crew(
    agents=[interview_question_generator],
    tasks=[task],
    verbose=True,
)

def main():
    try:
        # Kick off the Crew
        result = crew.kickoff()
        print("Analysis Results:")
        print(result)
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_text_path):
            os.unlink(temp_text_path)

if __name__ == "__main__":
    main()