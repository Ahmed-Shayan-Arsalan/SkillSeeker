# cs_query_model.py

import os
import sys
import pickle
import numpy as np
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
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

# API Key for Ranking Agent's LLaMA 3.1 8B Model
RANKING_AGENT_API_KEY = ""
ranking_llm = ChatGroq(model_name="groq/llama-3.1-8b-instant", groq_api_key=RANKING_AGENT_API_KEY)

# ============================
# Agent Definitions
# ============================

# Define CS Follow-up Agent
cs_followup_agent = Agent(
    role="CS Classification Follow-up",
    goal="Generate up to 3 follow-up questions to better classify the CS query.",
    backstory="You generate follow-up questions to clarify a CS query, focusing on the technical aspects. ONLY RESPOND WITH QUESTIONS, SAY NOTHING ELSE",
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

# Define Synthesis Agent
synthesis_agent = Agent(
    role="CS Information Synthesis",
    goal="Synthesize provided information to create a comprehensive CS query analysis and recommendations.",
    backstory="Combine all relevant information to suggest the most suitable CS professional and explain their suitability based on the context.",
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

# Define One-Line Answer Generator Agent
one_line_agent = Agent(
    role="One-Line Answer Generator",
    goal="Provide a concise one-line summary based on the initial query and follow-up answers.",
    backstory="You generate a one-line response that will guide ranking and final selection of CS professionals.",
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

# Define CS Ranking Agent
cs_ranking_agent = Agent(
    role="CS Ranking Agent",
    goal="Rank the combined results from vector embeddings and trained model outputs based on the original query.",
    backstory="You analyze the combined outputs to rank the top 4 CS professionals by relevance to the query. GIVE ALL DETAILS OF THE PROFESSIONAL IN PROPER STRUCTURE, DO NOT LEAVE DETAILS BEHIND ALWAYS GIVE THE LINK, ALWAYS GIVE AT LEAST 4 PROFESSIONALS.",
    allow_delegation=False,
    verbose=True,
    llm=ranking_llm,  # Use the new model with the provided API key
)

# Define Conversation Agent
cs_conversation_agent = Agent(
    role="CS Information Conversation",
    goal="Answer user questions about the top-ranked CS professionals, such as their experience, services, and specialties.",
    backstory="You provide detailed information about the top-ranked CS professionals based on the user's inquiries, ALWAYS GIVE SHORT CONCISE ANSWERS.",
    allow_delegation=False,
    verbose=True,
    llm=ranking_llm,  # Use the ranking_llm as specified
)

# ============================
# Memory Management (Remade)
# ============================

# Memory is now handled using local variables within the process_cs_query function

# ============================
# Data Loading Functions
# ============================

# Load the CS-specific vector database
def load_cs_vectordb():
    DB_FAISS_PATH = os.path.join("vectorstore", "cs_based_db_faiss")
    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(f"Could not find the VectorDB at {DB_FAISS_PATH}")
    embeddings = HuggingFaceEmbeddings(
        model_name="dunzhang/stella_en_1.5B_v5",
        model_kwargs={"device": "cpu"}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

# Load the trained model and associated data for ranking
def load_trained_model(data_path="Final_Build/Trained_models"):
    model_file = os.path.join(data_path, "logistic_regression_model.pkl")
    embeddings_file = os.path.join(data_path, "engineering_gig_embeddings.pkl")
    dataset_file = os.path.join(data_path, "Cleaned-All-engneering.csv")

    try:
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        with open(embeddings_file, "rb") as f:
            embedding_data = pickle.load(f)
        data = pd.read_csv(dataset_file, on_bad_lines="skip")
        data["embeddings"] = embedding_data["embeddings"]
        print("Trained model and dataset loaded successfully.")
        return model, data
    except Exception as e:
        raise RuntimeError(f"Error loading trained model: {e}")

# ============================
# Agent Task Functions
# ============================

# Generate follow-up questions using the CS Follow-up Agent
def get_cs_followup_questions(query):
    followup_task = Task(
        description=f"Generate up to 3 follow-up questions for better classifying the CS query: '{query}'",
        agent=cs_followup_agent,
        expected_output="questions",
    )
    crew = Crew(agents=[cs_followup_agent], tasks=[followup_task], verbose=True)
    followup_result = crew.kickoff()
    if followup_result.tasks_output:
        return followup_result.tasks_output[0].raw.strip()
    else:
        raise ValueError("No follow-up questions generated.")

# Synthesize information using the Synthesis Agent
def synthesize_cs_information(initial_query, answers):
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

# Generate a one-line summary using the One-Line Answer Generator Agent
def generate_one_line_summary(query, answers):
    one_line_task = Task(
        description=(
            f"Provide a one-line summary based on query: '{query}' and answers: {answers}."
        ),
        agent=one_line_agent,
        expected_output="one-line summary",
    )
    crew = Crew(agents=[one_line_agent], tasks=[one_line_task], verbose=True)
    one_line_result = crew.kickoff()
    if one_line_result.tasks_output:
        return one_line_result.tasks_output[0].raw.strip()
    else:
        raise ValueError("No one-line summary generated.")

# Rank the combined results using the CS Ranking Agent
def rank_combined_results(query, combined_results):
    # Convert combined_results to a textual format
    if isinstance(combined_results, list):
        combined_text = "\n".join([str(result) for result in combined_results])
    elif isinstance(combined_results, dict):
        combined_text = "\n".join([f"{key}: {value}" for key, value in combined_results.items()])
    else:
        combined_text = str(combined_results)

    ranking_task = Task(
        description=(
            f"Rank the following combined results based on query: '{query}'. "
            f"MAKE SURE TO GIVE EXTREME EMPHASIS ON RELEVANCE TO THE QUERY.You analyze the combined outputs to rank the top 4 CS professionals by relevance to the query. GIVE ALL DETAILS OF THE PROFESSIONAL IN PROPER STRUCTURE, DO NOT LEAVE DETAILS BEHIND, ALWAYS GIVE AT LEAST 4 PROFESSIONALS\n"
            f"Results: {combined_text}"
        ),
        agent=cs_ranking_agent,
        expected_output="ranking",
    )
    crew = Crew(agents=[cs_ranking_agent], tasks=[ranking_task], verbose=True)
    ranking_result = crew.kickoff()
    if ranking_result.tasks_output:
        return ranking_result.tasks_output[0].raw.strip()
    else:
        raise ValueError("No ranking result generated.")

# Retrieve and rank top CS professionals using Sentence-BERT and the trained logistic regression model
def retrieve_and_rank(query, data, model, top_k=5):
    # Load Sentence-BERT model
    model_name = "all-MiniLM-L12-v2"
    sbert_model = SentenceTransformer(model_name)

    # Encode query
    query_embedding = sbert_model.encode(query, convert_to_tensor=True).detach().cpu().numpy()

    # Compute cosine similarity
    dataset_embeddings = np.vstack(data["embeddings"].to_list())
    similarities = cosine_similarity([query_embedding], dataset_embeddings)[0]
    data["similarity"] = similarities

    # Use logistic regression to rank professionals
    # Assuming the model was trained with binary labels and higher probability indicates higher rank
    rank_scores = model.predict_proba(dataset_embeddings)[:, 1]
    data["rank_score"] = rank_scores
    ranked_data = data.sort_values(by=["rank_score", "similarity"], ascending=[False, False]).head(top_k)

    return ranked_data

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
                f"Ranked Professionals: {memory_data['ranked_professionals']}\n\n"
                f"User Question: {user_input}\n"
                f"You provide detailed information about the top-ranked CS professionals based on the user's inquiries. ALWAYS GIVE SHORT CONCISE ANSWERS."
            ),
            agent=cs_conversation_agent,
            expected_output="conversation_response",
        )
        crew = Crew(agents=[cs_conversation_agent], tasks=[conversation_task], verbose=True)
        conversation_result = crew.kickoff()
        if conversation_result.tasks_output:
            response = conversation_result.tasks_output[0].raw.strip()
            print(f"Agent: {response}")
        else:
            print("Agent: I'm sorry, I couldn't generate a response.")

# ============================
# Main Processing Function
# ============================

def process_cs_query(query):
    try:
        # Step 1: Generate follow-up questions
        followup_questions = get_cs_followup_questions(query).split("\n")
        answers = {}
        for i, q in enumerate(followup_questions[:3]):
            answer = input(f"{i+1}. {q.strip()}: ").strip()
            answers[q.strip()] = answer

        # Step 2: Synthesize information
        synthesis_result = synthesize_cs_information(query, answers)

        # Step 3: Retrieve relevant documents using RAG
        vectors = load_cs_vectordb()
        retriever = vectors.as_retriever()

        rag_results = retriever.get_relevant_documents(synthesis_result, k=5)
        print("\n=== RAG Results ===")
        rag_combined = []
        for doc in rag_results:
            print(f"Document: {doc.page_content}\n{'-'*50}")
            rag_combined.append(doc.page_content)

        # Step 4: Generate one-line summary
        one_line_summary = generate_one_line_summary(query, answers)
        print(f"\nOne-Line Summary: {one_line_summary}")

        # Step 5: Load trained model and retrieve ranked professionals
        model, data = load_trained_model()
        trained_results = retrieve_and_rank(one_line_summary, data, model)

        # Step 6: Print detailed information of each ranked professional
        print("\n=== Trained Model Results ===")
        trained_textual_results = []
        for _, professional in trained_results.iterrows():
            professional_details = {
                "Name": professional["Name"],
                "Gig Title": professional["Gig Title"],
                "Rating": professional["Rating"],
                "About": professional["About"],
                "Table Data": professional["Table Data"],
                "Package Title": professional["Package Title"],
                "Package Price": professional["Package Price"],
                "Review": professional["Review"],
                "Category": professional["Category"],
                "Link": professional["Link"],
                "Similarity": f"{professional['similarity']:.2f}",
                "Rank Score": f"{professional['rank_score']:.2f}",
            }
            trained_textual_results.append(professional_details)
            # Print details
            print(f"Name: {professional_details['Name']}")
            print(f"Gig Title: {professional_details['Gig Title']}")
            print(f"Rating: {professional_details['Rating']}")
            print(f"About: {professional_details['About']}")
            print(f"Table Data: {professional_details['Table Data']}")
            print(f"Package Title: {professional_details['Package Title']}")
            print(f"Package Price: {professional_details['Package Price']}")
            print(f"Review: {professional_details['Review']}")
            print(f"Category: {professional_details['Category']}")
            print(f"Link: {professional_details['Link']}")
            print(f"Similarity: {professional_details['Similarity']}")
            print(f"Rank Score: {professional_details['Rank Score']}")
            print('-' * 50)

        # Step 7: Combine RAG results and trained model results for ranking (used internally)
        combined_results = rag_combined + trained_textual_results

        # Step 8: Rank the combined results
        ranked_combined = rank_combined_results(query, combined_results)

        print("\n=== Final Ranked Top 4 Professionals ===")
        print(ranked_combined)

        # Convert ranked_combined to a string if it's a DataFrame
        if isinstance(ranked_combined, pd.DataFrame):
            ranked_combined_str = ranked_combined.to_string(index=False)
        else:
            ranked_combined_str = str(ranked_combined)

        # Step 9: Store only synthesis and ranking results into memory
        memory_data = {
            'synthesis_result': synthesis_result,
            'ranked_professionals': ranked_combined_str,  # Ensure it's a string
        }
        print("\nResults have been processed and stored in memory for further queries.")

        # Step 10: Initiate conversation with the Conversation Agent
        print("\nYou can now ask questions about the top-ranked CS professionals. Type 'exit' to end the conversation.")
        handle_conversation(memory_data)

    except Exception as e:
        print(f"Error processing query: {e}")

# ============================
# Main Execution Block
# ============================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
        process_cs_query(query)
    else:
        print("No query provided. Please provide a CS query as a command-line argument.")
