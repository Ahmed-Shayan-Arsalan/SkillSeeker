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
RANKING_AGENT_API_KEY = "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"
ranking_llm = ChatGroq(model_name="groq/llama-3.1-8b-instant", groq_api_key=RANKING_AGENT_API_KEY)

# ============================
# Agent Definitions
# ============================

# Define Medical Follow-up Agent
medical_followup_agent = Agent(
    role="Medical Classification Follow-up",
    goal="Generate up to 3 follow-up questions to better classify the medical query.",
    backstory="You generate follow-up questions to clarify a medical query, focusing on the medical aspect. ONLY RESPOND WITH QUESTIONS, SAY NOTHING ELSE",
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

# Define Details Verification Agent
details_verification_agent = Agent(
    role="User Details Verification",
    goal="Check if the original query contains any location reference.",
    backstory="You verify if a location is mentioned in the original query. Respond with 'Needed' if location is missing, 'Not Needed' if present.",
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

# Define Synthesis Agent
synthesis_agent = Agent(
    role="Medical Information Synthesis",
    goal="Synthesize provided information to create a comprehensive medical query analysis and recommendations.",
    backstory="Combine all relevant information to suggest one medical specialist and explain their suitability based on the context.",
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

# Define One-Line Answer Generator Agent
one_line_agent = Agent(
    role="One-Line Answer Generator",
    goal="Provide a concise one-line summary based on the initial query, further questions, and location.",
    backstory="You generate a one-line response that will guide ranking and final selection of doctors.",
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

# Define Doctor Ranking Agent
ranking_agent = Agent(
    role="Doctor Ranking Agent",
    goal="Rank the combined results from vector embeddings and trained model outputs based on the original query and location.",
    backstory="You analyze the combined outputs to rank the top 4 doctors by relevance to the query and location, GIVE ALL DETAILS OF THE DOCTOR IN PROPER STRUCTURE, DO NOT LEAVE DETAILS BEHIND, ALWAYS GIVE AT LEAST 4 DOCTORS.",
    allow_delegation=False,
    verbose=True,
    llm=ranking_llm,  # Use the new model with the provided API key
)

# Define Conversation Agent
conversation_agent = Agent(
    role="Doctor Information Conversation",
    goal="Answer user questions about the top-ranked doctors, such as their experience, services, and specialties.",
    backstory="You provide detailed information about the top-ranked doctors based on the user's inquiries, ALWAYS GIVE SHORT CONCISE ANSWERS.",
    allow_delegation=False,
    verbose=True,
    llm=ranking_llm,  # Use the ranking_llm as specified
)

# ============================
# Memory Management (Remade)
# ============================

# Memory is now handled using local variables within the process_medical_query function

# ============================
# Data Loading Functions
# ============================

# Load the medical-specific vector database
def load_medical_vectordb():
    DB_FAISS_PATH = os.path.join("vectorstore", "doctors_db_faiss2")
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
    embeddings_file = os.path.join(data_path, "doctor_embeddings.pkl")
    dataset_file = os.path.join(data_path, "doctors_clinic_timings.csv")

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

# Generate follow-up questions using the Medical Follow-up Agent
def get_medical_followup_questions(query):
    followup_task = Task(
        description=f"Generate up to 3 follow-up questions for better classifying the medical query: '{query}'",
        agent=medical_followup_agent,
        expected_output="questions",
    )
    crew = Crew(agents=[medical_followup_agent], tasks=[followup_task], verbose=True)
    followup_result = crew.kickoff()
    if followup_result.tasks_output:
        return followup_result.tasks_output[0].raw.strip()
    else:
        raise ValueError("No follow-up questions generated.")

# Verify if the user provided location details using the Details Verification Agent
def verify_user_details(query):
    details_task = Task(
        description=f"Check if this query contains any location reference: '{query}'",
        agent=details_verification_agent,
        expected_output="verification",
    )
    crew = Crew(agents=[details_verification_agent], tasks=[details_task], verbose=True)
    details_result = crew.kickoff()
    if details_result.tasks_output:
        return details_result.tasks_output[0].raw.strip()
    else:
        raise ValueError("No location verification result.")

# Synthesize information using the Synthesis Agent
def synthesize_information(initial_query, location, answers):
    synthesis_task = Task(
        description=(
            f"Synthesize all information and provide recommendations: "
            f"Original Query: {initial_query}, Location: {location}, Additional Information: {answers}"
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
def generate_one_line_summary(query, location, answers):
    one_line_task = Task(
        description=(
            f"Provide a one-line summary based on query: '{query}', "
            f"location: '{location}', and answers: {answers}."
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

# Rank the combined results using the Doctor Ranking Agent
def rank_combined_results(query, location, combined_results):
    # Convert combined_results to a textual format
    if isinstance(combined_results, list):
        combined_text = "\n".join([str(result) for result in combined_results])
    elif isinstance(combined_results, dict):
        combined_text = "\n".join([f"{key}: {value}" for key, value in combined_results.items()])
    else:
        combined_text = str(combined_results)

    ranking_task = Task(
        description=(
            f"Rank AND FORMAT the following combined results based on query: '{query}' and location: '{location}' MAKE SURE TO PUT EXTREME EMPHASIS ON LOCATION.\n"
            f"Results: {combined_text}"
        ),
        agent=ranking_agent,
        expected_output="ranking",
    )
    crew = Crew(agents=[ranking_agent], tasks=[ranking_task], verbose=True)
    ranking_result = crew.kickoff()
    if ranking_result.tasks_output:
        return ranking_result.tasks_output[0].raw.strip()
    else:
        raise ValueError("No ranking result generated.")

# Retrieve and rank top doctors using Sentence-BERT and the trained logistic regression model
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

    # Use logistic regression to rank doctors
    data["location_rank"] = model.predict(np.vstack(data["embeddings"].to_list()))
    ranked_data = data.sort_values(by=["location_rank", "similarity"], ascending=[False, False]).head(top_k)

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
                f"Ranked Doctors: {memory_data['ranked_doctors']}\n\n"
                f"User Question: {user_input}\n"
                f"You provide detailed information about the top-ranked doctors based on the user's inquiries. ALWAYS GIVE SHORT CONCISE ANSWERS."
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

def process_medical_query(query):
    try:
        # Step 1: Verify if location is needed
        location_needed = verify_user_details(query)
        if location_needed.lower() == "needed":
            location = input("Please specify your preferred location for the doctor: ").strip()
            if not location:
                raise ValueError("Location input cannot be empty.")
        elif location_needed.lower() == "not needed":
            location = "Location specified in query"
        else:
            raise ValueError("Invalid response from location verification agent.")

        # Step 2: Generate follow-up questions
        followup_questions = get_medical_followup_questions(query).split("\n")
        answers = {}
        for i, q in enumerate(followup_questions[:3]):
            answer = input(f"{i+1}. {q.strip()}: ").strip()
            answers[q.strip()] = answer

        # Step 3: Synthesize information
        synthesis_result = synthesize_information(query, location, answers)

        # Step 4: Retrieve relevant documents using RAG
        vectors = load_medical_vectordb()
        retriever = vectors.as_retriever()

        rag_results = retriever.get_relevant_documents(synthesis_result, k=5)
        print("\n=== RAG Results ===")
        rag_combined = []
        for doc in rag_results:
            print(f"Document: {doc.page_content}\n{'-'*50}")
            rag_combined.append(doc.page_content)

        # Step 5: Generate one-line summary (Not included in memory_data)
        one_line_summary = generate_one_line_summary(query, location, answers)
        print(f"\nOne-Line Summary: {one_line_summary}")

        # Step 6: Load trained model and retrieve ranked doctors
        model, data = load_trained_model()
        trained_results = retrieve_and_rank(one_line_summary, data, model)

        # Step 7: Print detailed information of each ranked doctor
        print("\n=== Trained Model Results ===")
        trained_textual_results = []
        for _, doctor in trained_results.iterrows():
            doctor_details = {
                "Doctor Name": doctor["Doctor Name"],
                "Similarity": f"{doctor['similarity']:.2f}",
                "Education": doctor.get("Education", "N/A"),
                "Review Details": doctor.get("Review Details", "N/A"),
                "Experience": doctor.get("Experience", "N/A"),
                "Qualifications": doctor.get("Qualifications", "N/A"),
                "Contact Number": doctor.get("Contact Number", "N/A"),
                "Maps Locations": doctor.get("Maps Locations", "N/A"),
                "Locations": f"{doctor.get('Locations', 'N/A')} (Rank: {doctor.get('location_rank', 'N/A')})",
                "Services": doctor.get("Services", "N/A"),
                "Diseases": doctor.get("Diseases", "N/A"),
                "Symptoms": doctor.get("Symptoms", "N/A"),
                "Interests": doctor.get("Interests", "N/A"),
            }
            trained_textual_results.append(doctor_details)
            # Print details
            print(f"Doctor: {doctor_details['Doctor Name']}")
            for key, value in doctor_details.items():
                if key != "Doctor Name":
                    print(f"{key}: {value}")
            print('-' * 50)

        # Step 8: Combine RAG results and trained model results for ranking (used internally)
        combined_results = rag_combined + trained_textual_results

        # Step 9: Rank the combined results
        ranked_combined = rank_combined_results(query, location, combined_results)

        print("\n=== Final Ranked Top 4 Professionals ===")
        print(ranked_combined)

        # Convert ranked_combined to a string if it's a DataFrame
        if isinstance(ranked_combined, pd.DataFrame):
            ranked_combined_str = ranked_combined.to_string(index=False)
        else:
            ranked_combined_str = str(ranked_combined)

        # Step 10: Store only synthesis and ranking results into memory
        memory_data = {
            'synthesis_result': synthesis_result,
            'ranked_doctors': ranked_combined_str,  # Ensure it's a string
        }
        print("\nResults have been processed and stored in memory for further queries.")

        # Step 11: Initiate conversation with the Conversation Agent
        print("\nYou can now ask questions about the top-ranked doctors. Type 'exit' to end the conversation.")
        handle_conversation(memory_data)

    except Exception as e:
        print(f"Error processing query: {e}")

# ============================
# Main Execution Block
# ============================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
        process_medical_query(query)
    else:
        print("No query provided. Please provide a medical query as a command-line argument.")
