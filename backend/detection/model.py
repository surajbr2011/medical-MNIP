import os
import torch
import torch.nn as nn
from functools import lru_cache
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Tuple

from backend.config import settings
from backend.api.schemas import NegligenceResult

# Device selection: CUDA, MPS (Apple Silicon), or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

WHO_ICPS_CATEGORIES = [
    "Clinical process/procedure",
    "Medical device/equipment",
    "Documentation",
    "Medication/IV fluids",
    "Clinical administration",
    "Healthcare-associated infection",
    "Resources/organizational",
    "Nutrition/systemic functions"
]

class BioClinicalBERTMultiTaskClassifier(nn.Module):
    """
    Bio_ClinicalBERT multi-task model with a custom classification head:
    Linear(768 -> 256) -> GELU -> Linear(256 -> 9)
    The 9 outputs represent 8 WHO ICPS categories + 1 binary negligence label.
    """
    def __init__(self, model_name_or_path: str = "emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, 256)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(256, 9) # 8 categories + 1 binary negligence

    def forward(self, input_ids: torch.Tensor = None, attention_mask: torch.Tensor = None, inputs_embeds: torch.Tensor = None, labels: torch.Tensor = None) -> torch.Tensor:
        if inputs_embeds is not None:
            outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use pooler output or mean pooling. Mean pooling is often more robust.
        # However, Bio_ClinicalBERT has a pooler, so we can use outputs.pooler_output.
        # Fallback to mean pooling if pooler_output is None.
        pooled = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state.mean(dim=1)
        x = self.dropout(pooled)
        x = self.fc1(x)
        x = self.gelu(x)
        logits = self.fc2(x)
        return logits

@lru_cache(maxsize=1)
def get_detection_model_and_tokenizer() -> Tuple[BioClinicalBERTMultiTaskClassifier, AutoTokenizer]:
    """Loads tokenizer and model weights once (singleton pattern)."""
    model_path = settings.CLINICALBERT_MODEL_PATH
    local_weights_dir = "models/clinicalbert_backend"
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        
    model = BioClinicalBERTMultiTaskClassifier("emilyalsentzer/Bio_ClinicalBERT")
    
    # Load fine-tuned weights if they exist locally
    if os.path.exists(os.path.join(local_weights_dir, "pytorch_model.bin")):
        try:
            model.load_state_dict(torch.load(os.path.join(local_weights_dir, "pytorch_model.bin"), map_location=device))
            model.eval()
            model.to(device)
            return model, tokenizer
        except Exception as e:
            print(f"Error loading local fine-tuned weights: {str(e)}. Falling back to base Bio_ClinicalBERT.")
            
    model.eval()
    model.to(device)
    return model, tokenizer

def predict_negligence(text: str) -> NegligenceResult:
    """Predict negligence class and WHO ICPS categories for a single clinical note."""
    model, tokenizer = get_detection_model_and_tokenizer()
    
    # Pre-import explainer to avoid circular dependencies
    from backend.detection.explainer import get_token_attributions
    
    inputs = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        # Apply sigmoid to logits for individual tasks
        probs = torch.sigmoid(logits)[0]
        
    # Extracted probabilities
    # Logits: [0:8] categories, [8] binary negligence
    cat_probs = probs[0:8].tolist()
    negligence_prob = float(probs[8].item())
    
    categories_dict = {
        WHO_ICPS_CATEGORIES[i]: cat_probs[i] for i in range(8)
    }
    
    # Get token attributions (gradient explanations)
    token_attributions = get_token_attributions(model, tokenizer, text)
    
    return NegligenceResult(
        negligent=negligence_prob >= 0.5,
        confidence=negligence_prob,
        categories=categories_dict,
        token_attributions=token_attributions
    )

def batch_predict_negligence(texts: List[str]) -> List[NegligenceResult]:
    """Perform batch inference for a list of clinical notes."""
    results = []
    for text in texts:
        results.append(predict_negligence(text))
    return results
