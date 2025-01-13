import os
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Set the Groq API Key
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual API key

# Initialize Groq LLM
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="groq/llama-3.1-70b-versatile")

# Define script paths
SCRIPT_PATHS = {
    "Medical Classifier": os.path.join("Final_Build", "medical_process.py"),
    "Computer Science Classifier": os.path.join("Final_Build", "cs_process.py"),
    "SEO Classifier": os.path.join("Final_Build", "seo_process.py"),
    "Legal Classifier": os.path.join("Final_Build", "legal_process.py"),
}

# Initialize Agents
def initialize_agents():
    medical_agent = Agent(
        role="Medical Classifier",
        goal="Tell me if the input query is 'Medical' or 'Other'.",
        backstory="You are an AI assistant trained to identify whether user is asking for medical problems or users. Your answer should only be 'Medical' or 'Other' DO NOT SAY ANYTHING OTHER THAN 'MEDICAL' OR 'Other'.",
        llm=llm,
        verbose=True,
    )
    cs_agent = Agent(
        role="Computer Science Classifier",
        goal="Tell me if the input query is 'CS' or 'Other'.",
        backstory="You are an AI assistant trained to identify whether a given query is related to CS/Programming or not. Answer should only be 'CS' or 'Other' DO NOT SAY ANYTHING OTHER THAN 'CS' OR 'Other'.",
        llm=llm,
        verbose=True,
    )
    seo_agent = Agent(
        role="SEO Classifier",
        goal="Tell me if the input query is 'SEO' or 'Other'.",
        backstory="You are an AI assistant trained to determine if a user specifically requests for better ranking or SEO. Answer should only be 'SEO' or 'Other' DO NOT SAY ANYTHING OTHER THAN 'SEO' OR 'Other'.",
        llm=llm,
        verbose=True,
    )
    legal_agent = Agent(
        role="Legal Classifier",
        goal="Tell me if the input query is 'Legal' or 'Other'.",
        backstory="You are an AI assistant trained to identify whether a query is related to the legal field. Answer should only be 'Legal' or 'Other' DO NOT SAY ANYTHING OTHER THAN 'Legal' OR 'Other'.",
        llm=llm,
        verbose=True,
    )

    return [medical_agent, cs_agent, seo_agent, legal_agent]

# Create tasks for agents
def create_task(agent, query):
    return Task(
        description=f"Classify the query: '{query}'",
        agent=agent,
        expected_output="classification",
    )

# Classify the query using parallelized agents
def classify_query_with_agents(query):
    agents = initialize_agents()
    tasks = []

    # Create tasks concurrently
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(create_task, agent, query): agent for agent in agents}
        for future in as_completed(futures):
            task = future.result()
            tasks.append(task)

    # Execute tasks using Crew
    crew = Crew(agents=agents, tasks=tasks, verbose=True)
    results = crew.kickoff(inputs={"query": query})

    # Parse the results
    classifications = {}
    for task_output in results.tasks_output:
        agent_name = task_output.agent  # Extract the agent name
        classification_result = task_output.raw.strip()  # Extract and clean the classification result
        classifications[agent_name] = classification_result  # Store the result

    return classifications

# Forward query to the appropriate processing script
def forward_to_script(script_path, query):
    try:
        # Pass current environment variables to the subprocess
        env = os.environ.copy()
        env["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")  # Ensure the API key is included
        
        subprocess.run([sys.executable, script_path, query], check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_path}: {e}")
    except FileNotFoundError:
        print(f"Script {script_path} not found.")

# Main function
def main():
    query = input("Enter your query: ").strip()

    # Classify the query
    classifications = classify_query_with_agents(query)

    # Output the classification results
    for role, classification in classifications.items():
        print(f"{role}: {classification}")

    # Check if all classifications returned "Other"
    if all(result == "Other" for result in classifications.values()):
        print("The query is either too ambiguous or does not contain enough information to classify to an available professional.")
        return

    # Forward to the appropriate script
    for agent_name, result in classifications.items():
        if result != "Other" and agent_name in SCRIPT_PATHS:
            script_path = SCRIPT_PATHS[agent_name]
            forward_to_script(script_path, query)
            return

    print("No further action required.")

if __name__ == "__main__":
    main()
