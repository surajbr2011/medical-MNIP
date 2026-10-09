import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, classification_report
from torch.utils.data import Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
import mlflow

from backend.config import settings
from backend.detection.model import BioClinicalBERTMultiTaskClassifier, WHO_ICPS_CATEGORIES, device

# Set MLflow experiment name
mlflow.set_experiment("backend_detection")

class ClinicalNotesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["note_text"])
        negligent = int(row["negligent"])
        category_str = str(row["category"])
        
        # Construct target vector of size 9
        # index 0 to 7: categories, index 8: binary negligence
        target = np.zeros(9, dtype=np.float32)
        target[8] = float(negligent)
        
        # Set category index
        if category_str in WHO_ICPS_CATEGORIES:
            cat_idx = WHO_ICPS_CATEGORIES.index(category_str)
            target[cat_idx] = 1.0
            
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(target, dtype=torch.float32)
        }

class FocalLossTrainer(Trainer):
    """Custom Trainer implementing focal loss (gamma=2.0) for multi-label class imbalance."""
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        
        # Forward pass
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Binary cross entropy with logits focal loss
        gamma = 2.0
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** gamma * bce_loss).mean()
        
        return (focal_loss, logits) if return_outputs else focal_loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Apply sigmoid to get probabilities
    probs = 1 / (1 + np.exp(-logits))
    
    # Let's evaluate metrics on the binary negligence label (index 8)
    preds = (probs[:, 8] >= 0.5).astype(int)
    true_labels = labels[:, 8].astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average="binary", zero_division=0)
    
    try:
        auroc = roc_auc_score(true_labels, probs[:, 8])
    except Exception:
        auroc = 0.5
        
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc)
    }

def create_mock_csv():
    """Create data/annotated_notes.csv if not exists so trainer runs immediately."""
    os.makedirs("data", exist_ok=True)
    csv_path = "data/annotated_notes.csv"
    if not os.path.exists(csv_path):
        print("Mocking data/annotated_notes.csv...")
        mock_data = []
        # Generate some mock clinical notes with negligent and clean labels
        notes_neg = [
            "Patient was admitted with acute appendicitis. Appendectomy was planned but delayed by 48 hours due to lack of surgeon availability, leading to appendix perforation and sepsis.",
            "Administered 50mg of Atenolol to patient instead of prescribed 5mg, resulting in severe bradycardia and hypotension.",
            "Left sponge in patient's abdomen during Caesarean section. Patient complained of pain and had fever; sponge found on X-ray 3 days later.",
            "Patient had severe chest pain. ECG showed ST elevation but doctor discharged patient as gastric irritation; patient suffered cardiac arrest at home.",
            "Cross-matching was missed; transfused B+ blood to A+ patient, leading to acute hemolytic reaction and renal failure."
        ]
        notes_non = [
            "Admitted patient for routine health checkup. Vital signs are normal. Advised discharge tomorrow.",
            "Patient underwent elective total knee replacement. Pre-operative checks normal, surgery uneventful. Pain controlled.",
            "Diagnosed with Type 2 diabetes. Prescribed Metformin 500mg daily. Patient educated on diet changes and scheduled for follow-up.",
            "Presented with mild fever and cough. Chest X-ray clear. Advised hydration and symptomatic treatment.",
            "Follow-up visit for hypertension. Blood pressure well-controlled at 120/80 mmHg. Continue current medication."
        ]
        
        for note in notes_neg:
            mock_data.append({
                "note_text": note,
                "negligent": 1,
                "category": np.random.choice(WHO_ICPS_CATEGORIES)
            })
        for note in notes_non:
            mock_data.append({
                "note_text": note,
                "negligent": 0,
                "category": np.random.choice(WHO_ICPS_CATEGORIES)
            })
            
        # Duplicate to get enough samples for split
        mock_data = mock_data * 5
        pd.DataFrame(mock_data).to_csv(csv_path, index=False)
        print(f"Created mock dataset with {len(mock_data)} records.")

def train_model():
    create_mock_csv()
    csv_path = "data/annotated_notes.csv"
    df = pd.read_csv(csv_path)
    
    # 80/20 train/val split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["negligent"])
    
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    
    train_dataset = ClinicalNotesDataset(train_df, tokenizer)
    val_dataset = ClinicalNotesDataset(val_df, tokenizer)
    
    model = BioClinicalBERTMultiTaskClassifier("emilyalsentzer/Bio_ClinicalBERT")
    model.to(device)
    
    output_dir = "models/clinicalbert_backend"
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=8,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2.3e-5,
        weight_decay=0.012,
        warmup_ratio=0.08,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=["mlflow"],
        logging_steps=1
    )
    
    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    print("Starting fine-tuning...")
    with mlflow.start_run():
        trainer.train()
        
        # Evaluate best model on validation set
        eval_results = trainer.evaluate()
        print("Validation Results:", eval_results)
        
        # Save best model state dict
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(output_dir)
        print(f"Saved best model to {output_dir}")
        
        # Final classification report on validation set
        predictions = trainer.predict(val_dataset)
        probs = 1 / (1 + np.exp(-predictions.predictions))
        preds = (probs[:, 8] >= 0.5).astype(int)
        true_labels = predictions.label_ids[:, 8].astype(int)
        
        print("\nFinal Classification Report (Validation Set):")
        print(classification_report(true_labels, preds, zero_division=0))

if __name__ == "__main__":
    train_model()
