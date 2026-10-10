import torch
import logging
from typing import List, Dict, Any

logger = logging.getLogger("backend.detection.explainer")

def get_token_attributions(model, tokenizer, text: str) -> List[Dict[str, Any]]:
    """
    Computes signed Input x Gradient attributions for each clinical token with respect
    to the negligence screening logit (index 8).
    
    Tokens with positive attributions contributed toward the model flagging potential negligence;
    tokens with negative attributions contributed against flagging.
    Special tokens ([CLS], [SEP], [PAD]) are filtered out to prevent pooling artifact inflation.
    """
    if not text or not text.strip():
        return []

    try:
        # 1. Tokenize text with attention mask
        inputs = tokenizer(
            text.strip(),
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # 2. Extract input word embeddings and enable gradient computation
        word_embeddings = model.bert.embeddings.word_embeddings(input_ids).clone().detach()
        word_embeddings.requires_grad = True
        
        # 3. Forward pass targeting negligence logit (index 8)
        model.eval()
        with torch.enable_grad():
            logits = model(inputs_embeds=word_embeddings, attention_mask=attention_mask)
            negligence_logit = logits[0, 8]
            model.zero_grad()
            negligence_logit.backward()
            grads = word_embeddings.grad
            
        if grads is None:
            return []
            
        # 4. Compute signed Input x Gradient attribution per token: sum(embed * grad) across embedding dimensions
        input_x_grad = (word_embeddings[0] * grads[0]).sum(dim=-1).detach().cpu().numpy() # Shape [seq_len]
        
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        seq_length = int(attention_mask[0].sum().item()) # Only actual tokens, ignoring padding
        
        # 5. Normalize signed scores relative to maximum absolute contribution
        valid_scores = input_x_grad[:seq_length]
        max_abs = float(np_max := max(abs(float(s)) for s in valid_scores)) if len(valid_scores) > 0 else 0.0
        
        attributions = []
        for idx in range(seq_length):
            tok = tokens[idx]
            # Exclude BERT structural markers
            if tok in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
                
            raw_val = float(input_x_grad[idx])
            norm_val = round(raw_val / max_abs, 4) if max_abs > 1e-8 else 0.0
            
            attributions.append({
                "token": tok,
                "attribution": norm_val,
                "direction": "positive" if norm_val > 0 else "negative" if norm_val < 0 else "neutral"
            })
            
        return attributions
        
    except Exception as e:
        logger.error(f"Error computing token attributions: {str(e)}")
        return []
