import os
from litellm import completion

# Set the Groq API key
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual API key

def classify_query(query, model_name):
    """Classify the query using Groq's LiteLLM."""
    response = completion(
        model=model_name,
        messages=[
            {"role": "user", "content": f"Classify this query: {query}"}
        ],
    )
    return response["choices"][0]["message"]["content"]

def load_vector_database():
    """Stub for vector database functionality."""
    print("Loading vector database... (Functionality not yet implemented)")

# Example models
models = {
    "medical": "groq/llama3-8b-8192",
    "cs": "groq/llama3-8b-8192",
    "legal": "groq/llama3-8b-8192",
    "seo": "groq/llama3-8b-8192",
    "journalism": "groq/llama3-8b-8192",
}

# Example usage
if __name__ == "__main__":
    query = "What is the treatment for flu?"
    category = classify_query(query, models["medical"])
    print(f"Classification Result: {category}")
