import os
import sys
import pickle
import numpy as np
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew
import warnings

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# ============================
# API Keys and Initialization
# ============================

# Main API Key for General Tasks
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    print("Error: GROQ_API_KEY is not set in the environment.")
    sys.exit(1)

# Initialize Groq LLM for general tasks
llm = ChatGroq(model_name="groq/llama-3.1-70b-versatile", groq_api_key=GROQ_API_KEY)

# ============================
# Agent Definitions
# ============================

# Define Legal Follow-up Agent
legal_followup_agent = Agent(
    role="Legal Classification Follow-up",
    goal="Generate up to 3 follow-up questions to better classify the Legal query.",
    backstory="You generate follow-up questions to clarify a legal query, focusing on the Legal aspect, ONLY RESPOND WITH QUESTIONS, SAY NOTHING ELSE.",
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

# Define Synthesis Agent
synthesis_agent = Agent(
    role="Legal Information Synthesis",
    goal="Synthesize provided information to create a comprehensive Legal query analysis and recommendations.",
    backstory="Combine all relevant information to suggest one legal specialist (Crime, Divorce, Business, Car Accident) and explain their suitability based on the context. Keep it short and in 1 line, KEEP IT SHORT.",
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

# Define Formatting Agent
formatting_agent = Agent(
    role="Answer Formatting Agent",
    goal="Clean and format the provided answers into a structured and readable format.",
    backstory="You ensure that the answers are well-organized, free of errors, and presented in a clear and professional manner.",
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

# Define Conversation Agent
conversation_agent = Agent(
    role="Legal Information Conversation",
    goal="Answer user questions about the top-ranked Lawyers, such as their experience, services, and specialties.",
    backstory="You provide detailed information about the top-ranked Lawyers based on the user's inquiries. ALWAYS GIVE SHORT CONCISE ANSWERS.",
    allow_delegation=False,
    verbose=True,
    llm=llm,  # Use the general LLM since ranking agent is removed
)

# ============================
# Memory Management (Remade)
# ============================

# Memory is now handled using local variables within the process_legal_query function

# ============================
# Data Loading Functions
# ============================

# Load the legal-specific vector database
def load_legal_vectordb():
    DB_FAISS_PATH = os.path.join("vectorstore", "lawyers_db_faiss3")
    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(f"Could not find the VectorDB at {DB_FAISS_PATH}")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

# ============================
# Agent Task Functions
# ============================

# Generate follow-up questions using the Legal Follow-up Agent
def get_legal_followup_questions(query):
    followup_task = Task(
        description=f"Generate up to 3 follow-up questions for better classifying the Legal based query: '{query}'",
        agent=legal_followup_agent,
        expected_output="questions",
    )
    crew = Crew(agents=[legal_followup_agent], tasks=[followup_task], verbose=True)
    followup_result = crew.kickoff()
    if followup_result.tasks_output:
        return followup_result.tasks_output[0].raw.strip()
    else:
        raise ValueError("No follow-up questions generated.")

# Synthesize information using the Synthesis Agent
def synthesize_information(initial_query, answers):
    synthesis_task = Task(
        description=(
            f"Synthesize all information and provide recommendations: "
            f"Original Query: {initial_query}, Additional Information: {answers}"
        ),
        agent=synthesis_agent,
        expected_output="synthesis",
    )
    crew = Crew(agents=[synthesis_agent], tasks=[synthesis_task], verbose=True)
    synthesis_result = crew.kickoff()
    if synthesis_result.tasks_output:
        return synthesis_result.tasks_output[0].raw.strip()
    else:
        raise ValueError("No synthesis result.")

# Format the combined results using the Formatting Agent
def format_combined_results(combined_results):
    combined_text = "\n".join([str(result) for result in combined_results])

    format_task = Task(
        description=(
            f"Clean and format the following combined results:\n{combined_text}"
        ),
        agent=formatting_agent,
        expected_output="formatted_results",
    )
    crew = Crew(agents=[formatting_agent], tasks=[format_task], verbose=True)
    format_result = crew.kickoff()
    if format_result.tasks_output:
        return format_result.tasks_output[0].raw.strip()
    else:
        raise ValueError("No formatted results generated.")

# ============================
# Conversation Agent Task Function
# ============================

# Answer logic for the Conversation Agent
def handle_conversation(memory_data):
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Ending conversation.")
            break

        conversation_task = Task(
            description=(
                f"Synthesis Result: {memory_data['synthesis_result']}\n"
                f"Formatted Lawyers: {memory_data['formatted_lawyers']}\n\n"
                f"User Question: {user_input}\n"
                f"You provide detailed information about the top-ranked Lawyers based on the user's inquiries. ALWAYS GIVE SHORT CONCISE ANSWERS."
            ),
            agent=conversation_agent,
            expected_output="conversation_response",
        )
        crew = Crew(agents=[conversation_agent], tasks=[conversation_task], verbose=True)
        conversation_result = crew.kickoff()
        if conversation_result.tasks_output:
            response = conversation_result.tasks_output[0].raw.strip()
            print(f"Agent: {response}")
        else:
            print("Agent: I'm sorry, I couldn't generate a response.")

# ============================
# Main Processing Function
# ============================

def process_legal_query(query):
    try:
        # Step 1: Generate follow-up questions
        followup_questions = get_legal_followup_questions(query).split("\n")
        answers = {}
        for i, q in enumerate(followup_questions[:3]):
            answer = input(f"{i+1}. {q.strip()}: ").strip()
            answers[q.strip()] = answer

        # Step 2: Synthesize information
        synthesis_result = synthesize_information(query, answers)

        # Step 3: Retrieve relevant documents using RAG
        vectors = load_legal_vectordb()
        retriever = vectors.as_retriever()

        rag_results = retriever.get_relevant_documents(synthesis_result, k=5)
        print("\n=== RAG Results ===")
        rag_combined = []
        for doc in rag_results:
            print(f"Document: {doc.page_content}\n{'-'*50}")
            rag_combined.append(doc.page_content)

        # Step 4: Combine RAG results for formatting
        combined_results = rag_combined

        # Step 5: Format the combined results
        formatted_combined = format_combined_results(combined_results)
        print("\n=== Formatted Combined Results ===")
        print(formatted_combined)

        # Step 6: Store synthesis and formatted results into memory
        memory_data = {
            'synthesis_result': synthesis_result,
            'formatted_lawyers': formatted_combined,  # Ensure it's a string
        }
        print("\nResults have been processed and stored in memory for further queries.")

        # Step 7: Initiate conversation with the Conversation Agent
        print("\nYou can now ask questions about the top-ranked lawyers. Type 'exit' to end the conversation.")
        handle_conversation(memory_data)

    except Exception as e:
        print(f"Error processing query: {e}")

# ============================
# Main Execution Block
# ============================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
        process_legal_query(query)
    else:
        print("No query provided. Please provide a legal query as a command-line argument.")
