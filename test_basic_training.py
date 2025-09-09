"""
Test basic NPT training without streaming to isolate issues.
"""

import torch
from transformers import AutoTokenizer, LlamaConfig
from src.npt import NPTLlamaModel, NPTConfig
from src.training import NPTTrainer, TrainingConfig
from torch.utils.data import Dataset, DataLoader

print("Testing Basic NPT Training")
print("=" * 50)

# Create small demo model
print("\n1. Creating demo model...")
config = LlamaConfig(
    hidden_size=256,
    intermediate_size=1024,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=4,
    vocab_size=128256,  # Match Llama tokenizer
)
config._attn_implementation = "eager"

model = NPTLlamaModel(config)
npt_config = NPTConfig(convert_all=True, np_rank=16)
model.convert_to_npt(npt_config)
model.freeze_base_parameters()

print(f"Model created with vocab_size={config.vocab_size}")
param_counts = model.count_parameters()
print(f"NPT parameters: {param_counts['npt']:,}")

# Load tokenizer
print("\n2. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer vocab size: {len(tokenizer)}")

# Create synthetic dataset
print("\n3. Creating synthetic dataset...")
class SyntheticDataset(Dataset):
    def __init__(self, size=20, seq_len=128, vocab_size=128256):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random token IDs within vocab range
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {
            'input_ids': input_ids,
            'attention_mask': torch.ones(self.seq_len),
            'labels': input_ids.clone()
        }

train_dataset = SyntheticDataset(size=10, seq_len=128)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

val_dataset = SyntheticDataset(size=5, seq_len=128)
val_loader = DataLoader(val_dataset, batch_size=2)

print("Dataset created successfully")

# Training config
print("\n4. Setting up training...")
training_config = TrainingConfig(
    batch_size=2,
    learning_rate=1e-3,
    max_steps=5,
    logging_steps=1,
    eval_steps=5,
    output_dir="/tmp/npt_test",
    device="cpu"  # Use CPU to avoid CUDA issues
)

trainer = NPTTrainer(
    model=model,
    config=training_config,
    train_loader=train_loader,
    val_loader=val_loader
)

print("\n5. Running training steps...")
print("-" * 40)

for i, batch in enumerate(train_loader):
    if i >= 3:
        break
    
    metrics = trainer.train_step(batch)
    trainer.global_step += 1
    print(f"Step {trainer.global_step}: Loss = {metrics.total_loss:.4f}")

print("-" * 40)
print("\n✓ Basic training works correctly!")

# Test with real text
print("\n6. Testing with real text tokenization...")
sample_text = "The Neuro-Plastic Transformer is a novel architecture that enables dynamic weight updates."
tokens = tokenizer(
    sample_text,
    return_tensors='pt',
    truncation=True,
    max_length=128,
    padding='max_length'
)

print(f"Tokenized shape: {tokens['input_ids'].shape}")
print(f"Max token ID: {tokens['input_ids'].max().item()}")
print(f"Min token ID: {tokens['input_ids'].min().item()}")

# Forward pass with real tokens
with torch.no_grad():
    outputs = model(
        input_ids=tokens['input_ids'],
        attention_mask=tokens['attention_mask']
    )
    print(f"Output shape: {outputs.logits.shape}")

print("\n✓ All tests passed!")