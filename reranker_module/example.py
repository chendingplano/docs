from reranker import QwenReranker
import traceback

def run_example():
    print("Starting QwenReranker example...")
    try:
        # Initialize the reranker
        # You can specify a device or let it auto-detect
        # reranker = QwenReranker(device="cpu") # Example for CPU
        reranker = QwenReranker()

        # Define a sample query and a list of documents
        sample_query = "What are the main challenges in renewable energy?"
        sample_docs = [
            "Solar power is becoming increasingly affordable.",
            "Wind turbines can impact bird populations.",
            "The integration of renewable sources into the grid is a significant challenge.",
            "Geothermal energy offers a consistent power supply.",
            "Battery storage is crucial for addressing the intermittency of renewables like solar and wind.",
            "Policy and regulatory hurdles often slow down renewable energy projects."
        ]

        print(f"\nQuery: {sample_query}")
        print("\nOriginal documents:")
        for i, doc in enumerate(sample_docs):
            print(f"{i+1}. {doc}")

        # Get reranked documents and their scores
        reranked_docs, reranked_scores = reranker.rerank(sample_query, sample_docs)

        print("\nReranked documents (and scores):")
        for i, (doc, score) in enumerate(zip(reranked_docs, reranked_scores)):
            print(f"{i+1}. Score: {score:.4f} - Document: {doc}")

        # Example with an empty query or documents
        print("\nTesting with empty query and documents:")
        empty_query_docs, empty_query_scores = reranker.rerank("", [])
        print(f"Reranked with empty query/docs: Docs: {empty_query_docs}, Scores: {empty_query_scores}")

        empty_docs, empty_scores = reranker.rerank("Test query", [])
        print(f"Reranked with empty docs: Docs: {empty_docs}, Scores: {empty_scores}")

        no_query_docs, no_query_scores = reranker.rerank("", ["Doc 1", "Doc 2"])
        print(f"Reranked with empty query: Docs: {no_query_docs}, Scores: {no_query_scores}")

    except Exception as e:
        print(f"An error occurred during the example: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    run_example()
