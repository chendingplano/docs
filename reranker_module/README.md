# Qwen Reranker Module

This module provides a `QwenReranker` class for reranking a list of documents based on their relevance to a given query. It utilizes a pre-trained model from the Hugging Face Transformers library, specifically designed for reranking tasks. The default model is "Qwen/Qwen2-0.5B-Reranker".

## Features

- Easy-to-use interface for reranking documents.
- Automatic download and caching of the reranking model.
- Attempts to load local models first to save time and bandwidth.
- Supports GPU (CUDA) if available, otherwise falls back to CPU.
- Includes an example script to demonstrate usage.

## Requirements

- Python 3.7+
- PyTorch
- Transformers library

You can install the necessary Python libraries using pip:
```bash
pip install torch transformers
```

## Structure

- `reranker.py`: Contains the `QwenReranker` class.
- `example.py`: Provides a runnable example of how to use the `QwenReranker`.
- `README.md`: This file, providing documentation for the module.

## Usage

1.  **Import the `QwenReranker` class:**
    ```python
    from reranker import QwenReranker
    ```

2.  **Initialize the reranker:**
    ```python
    reranker = QwenReranker()
    # Or specify a device:
    # reranker = QwenReranker(device="cpu")
    # reranker = QwenReranker(model_name="another/reranker-model") # To use a different model
    ```
    The first time you run this, the model (`Qwen/Qwen2-0.5B-Reranker` by default) will be downloaded and cached. Subsequent runs will load the model from the cache. If you have the model files stored locally in a directory structure that `from_pretrained` can recognize (e.g., a previous download or a manual download), you can point `model_name` to that local path, and it will attempt to load it from there first.

3.  **Prepare your query and documents:**
    ```python
    query = "What are the benefits of using a reranker?"
    documents = [
        "Rerankers improve search relevance.",
        "Machine learning models can be complex.",
        "A reranker filters and sorts documents based on a query.",
        "The sky is blue."
    ]
    ```

4.  **Call the `rerank` method:**
    ```python
    reranked_documents, reranked_scores = reranker.rerank(query, documents)
    ```

5.  **Process the results:**
    ```python
    print(f"Query: {query}\n")
    for doc, score in zip(reranked_documents, reranked_scores):
        print(f"Score: {score:.4f} - Document: {doc}")
    ```

## Running the Example

To see the reranker in action, you can run the provided example script:
```bash
cd path/to/reranker_module
python example.py
```
This will initialize the reranker (downloading the model if it's the first time) and then rerank a sample set of documents against a query, printing the results.

## Local Model Files

The reranker will first attempt to load the specified `model_name` using `local_files_only=True`. This means if you have previously downloaded the model or have a local copy of it (and its associated tokenizer files), you can provide the path to that directory as `model_name`. For example, if you have "Qwen/Qwen2-0.5B-Reranker" saved in `./my_local_qwen_reranker_model`, you could initialize with `QwenReranker(model_name="./my_local_qwen_reranker_model")`. If loading locally fails, it will proceed to download the model from Hugging Face Hub.

## Error Handling

The constructor includes basic error handling for model loading. If a model cannot be found locally, it attempts to download it. Other runtime errors during reranking should be caught by your own application code. The `example.py` includes a `try-except` block to catch potential errors during its execution.
