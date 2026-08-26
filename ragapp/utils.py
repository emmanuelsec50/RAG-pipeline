import json
import sys
import time
import torch
import os
import requests
from openai import OpenAI
import textwrap
import pickle
from decouple import config

PAGES_AND_CHUNKS_SAVE_PATH_PICKLE = "./ragapp/pages_and_chunks5.pkl"
EMBEDDINGS_PATH = './ragapp/embeddings5.pt'


def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)


def prompt_formatter(query: str,
                     context_items: list[dict]) -> str:
    context = "- " + "\n- ".join([item["chunk"] for item in context_items])

    base_prompt = f"""You are an Embu University Chat Bot. Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following example as reference for the ideal answer style.
\nExample 1:
Query: What is the importance of hydration for physical performance?
Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
\n If the retrieved context doesn't contain a clear answer, say so directly rather than reasoning through unrelated information.
\nNow use the following context items to answer the user query:
{context}
User query: {query}
Answer:""" 
    
    
    return base_prompt

def embed(context):
  # --- Configuration ---
    embed_api_key = config("EMBED_API_KEY")
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

    
    
    query_embedding = embed(query)
    

    # Get dot product scores on embeddings
    
    dot_scores = dot_score(query_embedding, embeddings)[0]
    

    
    scores, indices = torch.topk(input=dot_scores,
                                 k=n_resources_to_return)
    
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

    
    
    with open(PAGES_AND_CHUNKS_SAVE_PATH_PICKLE, "rb") as f:
        pages_and_chunks = pickle.load(f)
    
    # Loop through zipped together scores and indices from torch.topk
    for score, idx in zip(scores, indices):
        
        print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
        
        print("\n")

def glm(prompt: str):
    client = OpenAI(
    api_key=config('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "disabled"}}
    )
    
    for chunk in response:
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
    
    embeddings = torch.load(EMBEDDINGS_PATH)
    
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings)

    # Create a list of context items
    
    
    with open(PAGES_AND_CHUNKS_SAVE_PATH_PICKLE, "rb") as f:
        pages_and_chunks = pickle.load(f)
    
    context_items = [pages_and_chunks[i] for i in indices] 
    

    # Add score to context item
    
    for i, item in enumerate(context_items): 
        item["score"] = scores[i].cpu()
    

    # AUGMENTATION
    # Create the prompt and format it with context items
    
    prompt = prompt_formatter(query=query,
                              context_items=context_items)
    
    return glm(prompt)



def dot_score(a, b):
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    return torch.mm(a, b.transpose(0, 1))

def cos_sim(a, b):
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))