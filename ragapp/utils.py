import json
import sys
import time
import torch
import os
import requests
from .clients import openrouter_client, deepseek_client
import textwrap
import pickle
from decouple import config
from django.core.cache import cache
import asyncio
from .knowledge_base import get as get_knowledge_base
from asgiref.sync import sync_to_async


PAGES_AND_CHUNKS_SAVE_PATH_PICKLE = "./ragapp/pages_and_chunks_mongo.pkl"
EMBEDDINGS_PATH = './ragapp/embeddings_mongo.pt'


def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)


def prompt_formatter(query: str,
                     context_items: list[dict], conversation: str):
    context = "- " + "\n- ".join([item["chunk"] for item in context_items])

    base_prompt = f"""You are an Embu University Chat Bot. Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following example as reference for the ideal answer style.
\n CONVERSATION HISTORY
\n {conversation}
\n If the retrieved context doesn't contain a clear answer, say so directly rather than reasoning through unrelated information.
\nNow use the following context items to answer the user query:
{context}
User: {query}
Answer:""" 
    
    
    return base_prompt

async def embed(context):
  # --- Configuration ---
    # embed_api_key = config("EMBED_API_KEY")
    # url = "https://integrate.api.nvidia.com/v1/embeddings"

    # headers = {
    #     "Authorization": f"Bearer {embed_api_key}",
    #     "Content-Type": "application/json",
    #     "accept": "application/json",
    # }
    # payload = {
    #     "input": str(context),
    #     "model": "nvidia/nemotron-3-embed-1b",
    #     "input_type": "passage",
    #     "encoding_format": "float",
    #     "truncate": "NONE",
    #     "user": "string"
    # }
    
    # # --- Make Request ---
    # response = requests.post(url, headers=headers, json=payload)
    # response.raise_for_status()
    

    # # --- Extract Embedding ---
    # data = response.json()
    # embedding = data["data"][0]["embedding"]
    # return embedding

    embedding = await openrouter_client.embeddings.create(
    extra_headers={
        "HTTP-Referer": "ai.vixxon.online", # Optional. Site URL for rankings on openrouter.ai.
        "X-OpenRouter-Title": "vixxon", # Optional. Site title for rankings on openrouter.ai.
    },
    model="voyageai/voyage-4-large",
    input= context,
    # input: ["text1", "text2", "text3"] # batch embeddings also supported!
    encoding_format="float"
    )
    return embedding.data[0].embedding







async def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                n_resources_to_return: int=8,
                                print_time: bool=True):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """

    
    
    query_embedding = await embed(query)
    # print('embedded')

    # Get dot product scores on embeddings
    
    # dot_scores = dot_score(query_embedding, embeddings)[0]
    dot_scores = await asyncio.to_thread(dot_score, query_embedding, embeddings)

    
    scores, indices = torch.topk(input=dot_scores[0],
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

    
    
    
    embeddings, pages_and_chunks = get_knowledge_base()
    
    # Loop through zipped together scores and indices from torch.topk
    for score, idx in zip(scores, indices):
        
        print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
        
        print("\n")

async def glm(prompt: str):
    

    response = await deepseek_client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "disabled"}}
    )
    
    async for chunk in response:
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
   
async def ask(query: str,
        conversation,
        temperature: float=0.7,
        max_new_tokens:int=256,
        format_answer_text=True,
        return_answer_only=True):
    """
    Takes a query, finds relevant resources/context and generates an answer to the query based on the relevant resources.
    """



    convo = get_last_five_convo(conversation)
    prompt = await prompt_rewrite(query, convo)
    


    # RETRIEVAL
    # Get just the scores and indices of top related results
    # ----------------------------------------------------------------------------------
    embeddings, pages_and_chunks = get_knowledge_base()
    
    scores, indices = await retrieve_relevant_resources(query=prompt,
                                                  embeddings=embeddings)

    # Create a list of context items
    
    
    
    context_items = [pages_and_chunks[i] for i in indices] 
    

    # Add score to context item
    
    for i, item in enumerate(context_items): 
        item["score"] = scores[i].cpu()
    # --------------------------------------------------------------------------------------------

    # AUGMENTATION
    # Create the prompt and format it with context items
    # convo = get_last_five_convo(conversation)
    
    prompt = prompt_formatter(query=prompt,
                              context_items=context_items, conversation=convo)
    
    return glm(prompt)

async def stream_and_save(chat_id, prompt):
    accumulated = []
    # messages = sync_to_async(get_messages)(str(chat_id))
    async for chunk in await ask(prompt, await get_messages(str(chat_id))):
        if chunk.startswith("data: "):
            try:
                payload = json.loads(chunk[6:].strip())
                if payload.get("type") == "content":
                    accumulated.append(payload["text"])

            except (json.JSONDecodeError, ValueError):
                pass
        yield chunk
    # save_message(str(chat_id), 'user', prompt)
    # save_message(str(chat_id), 'assistant', "".join(accumulated))
    await sync_to_async(save_message)(str(chat_id),  'user', prompt)
    await sync_to_async(save_message)(str(chat_id), 'assistant', "".join(accumulated))



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


def save_message(chat_id: str, role: str, content: str):
    conversation = cache.get(chat_id)
    if conversation:
        conversation.append({
            "role": role,
            "content": content
             })
        cache.set(chat_id, conversation, 60*15)
    else:
        cache.set(chat_id, [{
            "role": role,
            "content": content
             }], 60*15)

async def get_messages(chat_id):
    conversation = await cache.aget(chat_id)
    if conversation:
        return conversation
    else:
        return []

def clear_messages(chat_id):
    cache.delete(chat_id)

def get_last_five_convo(conversation) -> str:
    convo = ''
    if not conversation:
        return f'No conversation yet'
    for item in conversation[-5:]:
        convo += f'{item['role']} - {item['content']}\n'
    return convo

async def prompt_rewrite(prompt, conversation):
    if conversation == 'No conversation yet':
        return prompt

    
    query_rewrite_prompt = """You are a query rewriter for a retrieval-augmented generation (RAG) system.

Your task is to transform the user's query into an optimized, standalone search query for a semantic document search engine.
Do NOT answer the query. Only rewrite it.Preserve the user's ORIGINAL INTENT. Never change what the user is actually asking for.
If the query is already clear and self-contained, return it unchanged or only lightly polished.
Output ONLY the rewritten query as plain text. No explanations, no quotes, no prefixes, no bullet points.
"""
   
    query_rewrite_prompt += f'\n {conversation}'
    query_rewrite_prompt += f'\n user: {prompt}'
    query_rewrite_prompt += f'\n Rewritten query:'
    
    



    
    response = await openrouter_client.chat.completions.create(
    model="deepseek/deepseek-v4-flash",
    messages=[
            {"role": "system", "content": "You are a query rewriter for a RAG system"},
            {
                "role": "user",
                "content": query_rewrite_prompt
            }
            ],
    extra_body={"reasoning": {"enabled": False}}
    )

    # Extract the assistant message with reasoning_details
    return response.choices[0].message.content