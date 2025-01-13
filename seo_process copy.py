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
from sentence_transformers import SentenceTransformer

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

# Define SEO Follow-up Agent
seo_followup_agent = Agent(
    role="SEO Classification Follow-up",
    goal="Generate up to 3 follow-up questions to better classify the SEO query.",
    backstory="You generate follow-up questions to clarify an SEO query, focusing on the SEO aspect. ONLY RESPOND WITH QUESTIONS, SAY NOTHING ELSE",
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

# Define Synthesis Agent
synthesis_agent = Agent(
    role="SEO Information Synthesis",
    goal="Synthesize provided information to create a comprehensive SEO query analysis and recommendations.",
    backstory="Combine all relevant information to suggest one SEO specialist and explain their suitability based on the context. Keep it short and in 1 line, KEEP IT SHORT.",
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
    role="SEO Information Conversation",
    goal="Answer user questions about the top-ranked SEO individuals, such as their experience, services, and specialties.",
    backstory="You provide detailed information about the top-ranked SEO individuals based on the user's inquiries. ALWAYS GIVE SHORT CONCISE ANSWERS.",
    allow_delegation=False,
    verbose=True,
    llm=llm,  # Use the general LLM since ranking agent is removed
)

# ============================
# Memory Management (Remade)
# ============================

# Memory is now handled using local variables within the process_seo_query function

# ============================
# Data Loading Functions
# ============================

# Load the SEO-specific vector database and trained model
def load_seo_model():
    """
    Load the trained NearestNeighbors model and embeddings for SEO gig retrieval.
    """
    model_file = "seo_nbrs_model.pkl"  # Path to the saved NearestNeighbors model
    embeddings_file = "seo_gig_embeddings.pkl"  # Path to the saved embeddings

    try:
        with open(embeddings_file, "rb") as f:
            embedding_data = pickle.load(f)
        with open(model_file, "rb") as f:
            nbrs = pickle.load(f)
        print("SEO embeddings and Nearest Neighbors model loaded successfully.")
        return nbrs, embedding_data
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required file not found: {e.filename}")
    except Exception as e:
        raise RuntimeError(f"Error loading SEO model: {e}")

# Load the dataset used for SEO gigs
def load_seo_dataset():
    """
    Load the dataset containing SEO gigs information.
    """
    file_path = "All-marketing2.csv"  # Update with your file path
    try:
        data = pd.read_csv(file_path, on_bad_lines='skip')
        print(f"SEO dataset loaded successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        raise RuntimeError(f"Error loading the SEO dataset: {e}")

# ============================
# Agent Task Functions
# ============================

# Generate follow-up questions using the SEO Follow-up Agent
def get_seo_followup_questions(query):
    followup_task = Task(
        description=f"Generate up to 3 follow-up questions for better classifying the SEO based query: '{query}'",
        agent=seo_followup_agent,
        expected_output="questions",
    )
    crew = Crew(agents=[seo_followup_agent], tasks=[followup_task], verbose=True)
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
# SEO Ranking Function
# ============================

# Retrieve and rank top SEO gigs using the trained NearestNeighbors model
def retrieve_top_seo_gigs(synthesis_query, data, nbrs, embedding_data, top_k=5):
    """
    Retrieve top-k most relevant SEO gigs for the given synthesis query.
    """
    # Initialize SentenceTransformer model
    model_name = "all-MiniLM-L12-v2"
    sbert_model = SentenceTransformer(model_name)

    # Encode the synthesis query
    query_embedding = sbert_model.encode(synthesis_query, convert_to_tensor=True).detach().cpu().numpy()

    # Find the nearest neighbors
    distances, indices = nbrs.kneighbors([query_embedding])
    results = []
    for i, idx in enumerate(indices[0][:top_k]):
        gig = data.iloc[idx]
        results.append({
            'Name': gig['Name'],
            'Gig Title': gig['Gig Title'],
            'Rating': gig['Rating'],
            'About': gig['About'],
            'Table Data': gig['Table Data'],
            'Package Title': gig['Package Title'],
            'Package Price': gig['Package Price'],
            'Review': gig['Review'],
            'Category': gig['Category'],
            'Link': gig['Link'],
            'Similarity': 1 - distances[0][i]  # Cosine similarity
        })
    return results

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
                f"Ranked SEO Gigs: {memory_data['ranked_seo_gigs']}\n\n"
                f"User Question: {user_input}\n"
                f"You provide detailed information about the top-ranked SEO individuals based on the user's inquiries. ALWAYS GIVE SHORT CONCISE ANSWERS."
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

def process_seo_query(query):
    try:
        # Step 1: Generate follow-up questions
        followup_questions = get_seo_followup_questions(query).split("\n")
        answers = {}
        for i, q in enumerate(followup_questions[:3]):
            answer = input(f"{i+1}. {q.strip()}: ").strip()
            answers[q.strip()] = answer

        # Step 2: Synthesize information
        synthesis_result = synthesize_information(query, answers)

        # Step 3: Load SEO model and dataset
        nbrs, embedding_data = load_seo_model()
        data = load_seo_dataset()

        # Step 4: Retrieve top SEO gigs
        top_seo_gigs = retrieve_top_seo_gigs(synthesis_result, data, nbrs, embedding_data, top_k=5)
        print("\n=== Top SEO Gigs ===")
        for gig in top_seo_gigs:
            print(f"Name: {gig['Name']}")
            print(f"Gig Title: {gig['Gig Title']}")
            print(f"Rating: {gig['Rating']}")
            print(f"About: {gig['About']}")
            print(f"Table Data: {gig['Table Data']}")
            print(f"Package Title: {gig['Package Title']}")
            print(f"Package Price: {gig['Package Price']}")
            print(f"Review: {gig['Review']}")
            print(f"Category: {gig['Category']}")
            print(f"Link: {gig['Link']}")
            print(f"Similarity: {gig['Similarity']:.2f}\n")

        # Step 5: Format the combined results
        # Here, combined_results consist of the top SEO gigs
        formatted_combined = format_combined_results(top_seo_gigs)
        print("\n=== Formatted Combined Results ===")
        print(formatted_combined)

        # Step 6: Store synthesis and ranked SEO gigs into memory
        memory_data = {
            'synthesis_result': synthesis_result,
            'ranked_seo_gigs': formatted_combined,  # Ensure it's a string
        }
        print("\nResults have been processed and stored in memory for further queries.")

        # Step 7: Initiate conversation with the Conversation Agent
        print("\nYou can now ask questions about the top-ranked SEO individuals. Type 'exit' to end the conversation.")
        handle_conversation(memory_data)

    except Exception as e:
        print(f"Error processing query: {e}")

# ============================
# Main Execution Block
# ============================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
        process_seo_query(query)
    else:
        print("No query provided. Please provide an SEO-based query as a command-line argument.")
