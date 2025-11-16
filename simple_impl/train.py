import torch
from torch.utils.data import DataLoader
from transformers import BertConfig
from tqdm import tqdm
import gc

from dataset.full_sequence_dataset import CATHFullSequenceDataset
from foldingdiff.bert_for_riemannian_diffusion import BertForRiemannianDiffusion
from schedule import LinearBetaSchedule
from sde_lib import DiffusionMixture
from losses import get_mix_loss_fn

# from likelihood import Likelihood
from util.ema import ExponentialMovingAverage


# This will stop the script and show you the exact operation that created a nan (it was a huge problem)
torch.autograd.set_detect_anomaly(True)


LEARNING_RATE = 2e-5  # Was 2e-4 but it was giving nan (chatgpt told me was a good fix to change it)
BATCH_SIZE = 2
NUM_EPOCHS = 5000
GRAD_CLIP_NORM = 1.0

# Model and Data Configuration
#ps: we may need to change that a lot, Samir 
MAX_LEN = 128
BERT_HIDDEN_SIZE = 256
BERT_NUM_HEADS = 8
BERT_NUM_LAYERS = 3


LOG_FREQ_EPOCHS = 2
SEED = 69


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#handling the data...
print(f"Loading dataset with max_len = {MAX_LEN}...")
dataset = CATHFullSequenceDataset("./data/angles/", max_len=MAX_LEN)
manifold = dataset.manifold
feature_dim = 2 * dataset.torus_dim


N = len(dataset)
N_val = N_test = N // 10
N_train = N - N_val - N_test

train_set, val_set, test_set = torch.utils.data.random_split(
    dataset,
    [N_train, N_val, N_test],
    generator=torch.Generator().manual_seed(SEED),
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Training with {N_train} examples, validating with {N_val} examples.")


#  DIFFUSION SETUP HERE: 

beta = LinearBetaSchedule(beta_0=0.2, beta_f=0.001, t0=0.0, tf=1.0)

mix = DiffusionMixture(
    manifold,
    beta,
    mix_type='log',
    drift_scale=1.0,
    pred=False,
    prior_type='unif'
)

loss_fn = get_mix_loss_fn(mix, num_steps=20, loss_type='smooth_l1', beta=0.1)


print("Initializing model...")
cfg = BertConfig(
    max_position_embeddings=MAX_LEN,
    hidden_size=BERT_HIDDEN_SIZE,
    num_attention_heads=BERT_NUM_HEADS,
    num_hidden_layers=BERT_NUM_LAYERS,
    intermediate_size=4 * BERT_HIDDEN_SIZE,
    use_cache=False,
    _attn_implementation="eager"
)

modelf = BertForRiemannianDiffusion(cfg, manifold, feature_dim).to(device)
modelb = BertForRiemannianDiffusion(cfg, manifold, feature_dim).to(device)

optf = torch.optim.Adam(modelf.parameters(), lr=LEARNING_RATE)
optb = torch.optim.Adam(modelb.parameters(), lr=LEARNING_RATE)

emaF = ExponentialMovingAverage(modelf.parameters(), decay=0.9999)
emaB = ExponentialMovingAverage(modelb.parameters(), decay=0.9999)



@torch.no_grad()
def evaluate_loss(modelf, modelb, val_loader, loss_fn):
    """Calculates the average loss on the validation set."""
    modelf.eval()
    modelb.eval()
    
    total_val_loss = 0.0
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        data_tensor = batch['x']
        
        #  FIX 2: add sanity check for nan so it doesn't return a nan scored bathc like before
        if torch.any(torch.isnan(data_tensor)):
            print("NaN found in validation data batch!")
            continue

        loss, _, _ = loss_fn(modelf, modelb, data_tensor)
        if not torch.isnan(loss): # Only add valid loss values
            total_val_loss += loss.item()
        
    return total_val_loss / len(val_loader)

# traininig loop here: 
def train():
    """Main training loop."""
    print("Starting training... LET'S ROCK! ")
    for epoch in range(1, NUM_EPOCHS + 1):
        modelf.train()
        modelb.train()
        
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False)
        
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            data_tensor = batch['x']
            
            # same fix for nan data as before
            if torch.any(torch.isnan(data_tensor)):
                print(f"NaN found in training data batch! Skipping.")
                continue

            optf.zero_grad()
            optb.zero_grad()
            
            loss, lf, lb = loss_fn(modelf, modelb, data_tensor)
            
            # Check for nan before backprop.
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch}. Skipping backward pass.")
                #  stop the process for this batch but continue the training loop to see if the model recovers.
                continue

            loss.backward()

            if GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(modelf.parameters(), GRAD_CLIP_NORM)
                torch.nn.utils.clip_grad_norm_(modelb.parameters(), GRAD_CLIP_NORM)

            optf.step()
            optb.step()

            emaF.update(modelf.parameters())
            emaB.update(modelb.parameters())

            total_train_loss += loss.item()
            pbar.set_postfix(batch_loss=loss.item())
        
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        if epoch % LOG_FREQ_EPOCHS == 0:
            emaF.copy_to(modelf.parameters())
            emaB.copy_to(modelb.parameters())
            
            avg_val_loss = evaluate_loss(modelf, modelb, val_loader, loss_fn)
            
            print(f"Epoch {epoch: >4}/{NUM_EPOCHS} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")

    print("Training finished.")
    print("Saving final model checkpoint to 'trained_diffusion.pkl'...")
    emaF.copy_to(modelf.parameters())
    emaB.copy_to(modelb.parameters())
    torch.save(
        {"f": modelf.state_dict(), "b": modelb.state_dict()},
        "trained_diffusion.pkl"
    )

if __name__ == "__main__":
    train()