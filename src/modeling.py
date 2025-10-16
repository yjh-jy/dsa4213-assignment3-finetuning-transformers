# src/modeling.py
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from src.config import MODEL_NAME, LORA_R, LORA_ALPHA, LORA_DROPOUT

def get_model_and_freeze(model_name=MODEL_NAME, strategy="full", num_labels=2, lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=None):
    """
    strategy: "full" or "lora"
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    if strategy == "lora":
        # Default target modules - adjust if your model uses different names
        if target_modules is None:
            target_modules = ["q_lin", "v_lin"]  # only adapts the query and value projections

        # to check the layer names in the current model
        # for name, module in model.named_modules():
        #     print(name)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="SEQ_CLS"
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
    # strategy == "full" => do nothing (all params trainable)
    return model
