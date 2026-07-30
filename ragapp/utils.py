import json
import sys
import time
import torch
import os
import requests
from openai import OpenAI
from sentence_transformers import util
import textwrap
import pickle
from decouple import config

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)


def prompt_formatter(query: str,
                     context_items: list[dict]) -> str:
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    base_prompt = f"""Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following example as reference for the ideal answer style.
\nExample 1:
Query: What is the importance of hydration for physical performance?
Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
\nNow use the following context items to answer the user query:
{context}
User query: {query}
Answer:""" 
    
    
    return base_prompt

def embed(context):
  # --- Configuration ---
    embed_api_key = config("EMBED_API_KEY")  # Get from openrouter.ai/settings/keys
    url = "https://integrate.api.nvidia.com/v1/embeddings"

    headers = {
        "Authorization": f"Bearer {embed_api_key}",
        "Content-Type": "application/json",
        "accept": "application/json",
    }
    payload = {
        "input": str(context),
        "model": "nvidia/nemotron-3-embed-1b",
        "input_type": "passage",
        "encoding_format": "float",
        "truncate": "NONE",
        "user": "string"
    }
    
    # --- Make Request ---
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    

    # --- Extract Embedding ---
    data = response.json()
    embedding = data["data"][0]["embedding"]
    return embedding








def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                n_resources_to_return: int=8,
                                print_time: bool=True):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """

    # Embed the query
    # query_embedding = model.encode(query, convert_to_tensor=True)
    t0 = time.time()
    query_embedding = embed(query)
    print(f"[TIMING] embedding request: {time.time() - t0:.2f}s")

    # Get dot product scores on embeddings
    t1 = time.time()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    print(f"[TIMING] dot scores: {time.time() - t1:.2f}s")

    # if print_time:
    #     print(f"[INFO] Time taken to get scores on ({len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")
    t2 = time.time()
    scores, indices = torch.topk(input=dot_scores,
                                 k=n_resources_to_return)
    print(f"[TIMING] top k: {time.time() - t2:.2f}s")
    return scores, indices

def print_top_results_and_scores(query: str,
                                 embeddings: torch.tensor,
                                 pages_and_chunks: list[dict],
                                 n_resources_to_return: int=5):
    """
    Finds relevant passages given a query and prints them out along with their scores.
    """
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  n_resources_to_return=n_resources_to_return)

    t3 = time.time()
    pages_and_chunks_save_path_pickle = "./ragapp/pages_and_chunks1.pkl"
    with open(pages_and_chunks_save_path_pickle, "rb") as f:
        pages_and_chunks = pickle.load(f)
    print(f"[TIMING] retrieval of ./ragapp/pages_and_chunks1.pkl: {time.time() - t3:.2f}s")
    # Loop through zipped together scores and indices from torch.topk
    for score, idx in zip(scores, indices):
        # print(f"Score: {score:.4f}")
        # print("Text:")
        print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
        # print(f"Page number: {pages_and_chunks[idx]['page_number']}")
        print("\n")

def glm(prompt: str):
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=config("GLM_API_KEY")
    )

    t4 = time.time()
    completion = client.chat.completions.create(
        model="z-ai/glm-5.2",
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.78,
        top_p=0.81,
        max_tokens=16384,
        seed=42,
        # extra_body={"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}},
        stream=True
    )
    print(f"[TIMING] glm: {time.time() - t4:.2f}s")
    t13 = time.time()
    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        if len(chunk.choices) == 0 or getattr(chunk.choices[0], "delta", None) is None:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        content = getattr(delta, "content", None)

        if reasoning:
            yield f"data: {json.dumps({'type': 'reasoning', 'text': reasoning})}\n\n"
        if content:
            yield f"data: {json.dumps({'type': 'content', 'text': content})}\n\n"
    
    yield "data: [DONE]\n\n"
    print(f"[TIMING] glm: {time.time() - t13:.2f}s")
def ask(query: str,
        temperature: float=0.7,
        max_new_tokens:int=256,
        format_answer_text=True,
        return_answer_only=True):
    """
    Takes a query, finds relevant resources/context and generates an answer to the query based on the relevant resources.
    """

    # RETRIEVAL
    # Get just the scores and indices of top related results
    t5 = time.time()
    embeddings = torch.load('./ragapp/embeddings1.pt')
    print(f"[TIMING] torch.load: {time.time() - t5:.2f}s")
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings)

    # Create a list of context items
    pages_and_chunks_save_path_pickle = "./ragapp/pages_and_chunks1.pkl"
    t6 = time.time()
    with open(pages_and_chunks_save_path_pickle, "rb") as f:
        pages_and_chunks = pickle.load(f)
    context_items = [pages_and_chunks[i] for i in indices] 
    print(f"[TIMING] pickle and iteration: {time.time() - t6:.2f}s")

    # Add score to context item
    t7 = time.time()
    for i, item in enumerate(context_items): 
        item["score"] = scores[i].cpu()
    print(f"[TIMING] Add score to context item: {time.time() - t7:.2f}s")

    # AUGMENTATION
    # Create the prompt and format it with context items
    t8 = time.time()
    prompt = prompt_formatter(query=query,
                              context_items=context_items)
    print(f"[TIMING] Create the prompt and format it with context items: {time.time() - t8:.2f}s.  {os.cpu_count()}")
    return glm(prompt)