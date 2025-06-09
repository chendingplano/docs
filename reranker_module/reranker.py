import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class QwenReranker:
    def __init__(self, model_name="Qwen/Qwen2-0.5B-Reranker", device=None):
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = None
        self.model = None

        try:
            # Try to load from local files first
            print(f"Attempting to load {self.model_name} from local files...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, local_files_only=True)
            print(f"Successfully loaded {self.model_name} from local files.")
        except (OSError, EnvironmentError) as e: # OSError is common for model not found, EnvironmentError for other issues
            print(f"Model {self.model_name} not found locally or error loading: {e}. Downloading and caching...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            print(f"Successfully downloaded and loaded {self.model_name}.")

        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device} and set to evaluation mode.")

    def rerank(self, query: str, documents: list[str]):
        if not query or not documents:
            return [], []

        pairs = []
        for doc in documents:
            pairs.append([query, doc])

        all_scores = []
        with torch.no_grad():
            for pair in pairs:
                inputs = self.tokenizer(pair[0], pair[1], return_tensors='pt', max_length=512, truncation=True, padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                # For rerankers, the score is often the logit of the second class (index 1)
                # or a specific output head. Assuming it's the logit for the "relevant" class.
                # It's good practice to confirm this from model documentation if available.
                # For Qwen reranker, it's typically the score from the output without applying softmax.
                score = outputs.logits.squeeze().item() # Squeeze to remove unnecessary dimensions
                all_scores.append(score)

        # Combine documents with their scores
        scored_documents = list(zip(documents, all_scores))

        # Sort documents by score in descending order
        reranked_documents_with_scores = sorted(scored_documents, key=lambda x: x[1], reverse=True)

        # Separate documents and scores for return if needed, or return as tuples
        reranked_docs = [doc for doc, score in reranked_documents_with_scores]
        reranked_scores = [score for doc, score in reranked_documents_with_scores]

        return reranked_docs, reranked_scores
