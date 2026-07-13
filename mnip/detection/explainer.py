import torch
from typing import List, Dict, Any

def get_token_attributions(model, tokenizer, text: str) -> List[Dict[str, Any]]:
    """
    Computes gradient-based attributions for each token in the input text
    with respect to the negligence class logit (index 8).
    """
    # 1. Tokenize text
    inputs = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # 2. Get device of model
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # 3. Get input embeddings
    # In HuggingFace BERT, word embeddings are at model.bert.embeddings.word_embeddings
    word_embeddings = model.bert.embeddings.word_embeddings(input_ids).clone().detach()
    word_embeddings.requires_grad = True
    
    # 4. Forward pass
    # Temporarily enable grads just for this computation
    with torch.enable_grad():
        logits = model(inputs_embeds=word_embeddings, attention_mask=attention_mask)
        
        # Target logit index 8 is the binary negligence prediction
        negligence_logit = logits[0, 8]
        
        # 5. Backward pass to compute gradients
        model.zero_grad()
        negligence_logit.backward()
        
        # 6. Extract gradients w.r.t. embeddings
        grads = word_embeddings.grad  # Shape: [1, seq_len, 768]
        
    if grads is None:
        # Fallback to zero attributions if gradients are not computed
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        return [{"token": t, "attribution": 0.0} for t in tokens if t != "[PAD]"]
        
    # Calculate attribution score per token: L2 norm of gradient
    # Shape: [seq_len]
    attribution_scores = torch.norm(grads[0], dim=-1)
    
    # Normalize scores between 0 and 1
    max_score = attribution_scores.max().item()
    if max_score > 0:
        attribution_scores = (attribution_scores / max_score).tolist()
    else:
        attribution_scores = attribution_scores.tolist()
        
    # Convert token IDs back to strings
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Filter out [PAD] tokens to reduce size of payload
    attributions = []
    for token, score in zip(tokens, attribution_scores):
        if token == "[PAD]":
            continue
        attributions.append({
            "token": token,
            "attribution": float(score)
        })
        
    return attributions
