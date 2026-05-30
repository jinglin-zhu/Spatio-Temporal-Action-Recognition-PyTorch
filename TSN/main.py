import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import miniUCFTSNDataset
from model import TSN

def parse_args():
    parser = argparse.ArgumentParser(description="miniUCF Temporal Segment Networks (TSN)")
    
    # Path configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    parser.add_argument('--train_list', type=str, 
                        default=os.path.join(project_root, 'data', 'train.txt'),
                        help='Path to the training list file.')
    parser.add_argument('--val_list', type=str, 
                        default=os.path.join(project_root, 'data', 'validation.txt'),
                        help='Path to the validation list file.')
    parser.add_argument('--classes', type=str, 
                        default=os.path.join(project_root, 'data', 'classes.txt'),
                        help='Path to the classes mapping file.')
    
    # Model & Optimization settings
    parser.add_argument('--modality', type=str, default='RGB', choices=['RGB', 'Flow'],
                        help="Input modality. 'RGB' or 'Flow'.")
    parser.add_argument('--init_strategy', type=str, default='imagenet', choices=['imagenet', 'random'],
                        help="Weight initialization strategy. 'imagenet' or 'random'.")
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
    parser.add_argument('--num_segments', type=int, default=4, help='Number of segments K for TSN.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of threads for DataLoader.')
    
    # Evaluation flags
    parser.add_argument('--evaluate_fusion', action='store_true',
                        help='If set, run late fusion evaluation using saved RGB and Flow checkpoints.')
    
    return parser.parse_args()

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one single epoch.

    Args:
        model (nn.Module): The TSN model to be trained.
        dataloader (DataLoader): Training DataLoader.
        criterion (nn.Module): Loss function (CrossEntropyLoss).
        optimizer (Optimizer): Optimizer (SGD).
        device (torch.device): Device to run computation on (GPU/CPU).

    Returns:
        tuple: A tuple containing:
            - float: Average training loss over the epoch.
            - float: Training accuracy over the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="  [Train]", leave=False)
    for clips, labels in progress_bar:
        clips = clips.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass on TSN
        outputs = model(clips)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item() * clips.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += clips.size(0)
        
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0 * correct / total:.2f}%")
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """
    Validate the model over the validation dataset.

    Args:
        model (nn.Module): The TSN model to be evaluated.
        dataloader (DataLoader): Validation DataLoader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run computation on (GPU/CPU).

    Returns:
        tuple: A tuple containing:
            - float: Average validation loss.
            - float: Validation top-1 accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="  [Val]", leave=False)
        for clips, labels in progress_bar:
            clips = clips.to(device)
            labels = labels.to(device)
            
            outputs = model(clips)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * clips.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += clips.size(0)
            
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0 * correct / total:.2f}%")
            
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def evaluate_per_class_accuracy(model, loader, device, num_classes=25):
    """
    Calculate the top-1 classification accuracy for each individual class.

    Args:
        model (nn.Module): Trained TSN model.
        loader (DataLoader): Aligned validation DataLoader.
        device (torch.device): Device to run computation on (GPU/CPU).
        num_classes (int, optional): Total number of unique classes. Defaults to 25.

    Returns:
        list of float: List containing accuracy [0.0 - 1.0] for each class index.
    """
    model.eval()
    class_correct = [0.0] * num_classes
    class_total = [0.0] * num_classes
    
    with torch.no_grad():
        for clips, labels in loader:
            clips = clips.to(device)
            labels = labels.to(device)
            
            outputs = model(clips)
            _, preds = torch.max(outputs, 1)
            
            for label, pred in zip(labels, preds):
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1
                
    class_accuracies = []
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracies.append(class_correct[i] / class_total[i])
        else:
            class_accuracies.append(0.0)
            
    return class_accuracies

def run_late_fusion(rgb_model, flow_model, rgb_loader, flow_loader, device):
    """
    Perform Late Fusion by averaging predictions of RGB and Flow models at test time.

    Args:
        rgb_model (nn.Module): Trained RGB TSN model.
        flow_model (nn.Module): Trained Flow TSN model.
        rgb_loader (DataLoader): Aligned validation DataLoader for RGB.
        flow_loader (DataLoader): Aligned validation DataLoader for Flow.
        device (torch.device): Device to run computation on (GPU/CPU).

    Returns:
        float: Aligned late-fusion top-1 classification accuracy.
    """
    rgb_model.eval()
    flow_model.eval()
    
    correct = 0
    total = 0
    
    # Zip loaders together. This is mathematically correct since both datasets are aligned (shuffle=False)
    with torch.no_grad():
        for (rgb_clips, labels), (flow_clips, _) in zip(rgb_loader, flow_loader):
            rgb_clips = rgb_clips.to(device)
            flow_clips = flow_clips.to(device)
            labels = labels.to(device)
            
            rgb_logits = rgb_model(rgb_clips)
            flow_logits = flow_model(flow_clips)
            
            # Convert logits to probability distributions via Softmax
            rgb_probs = torch.softmax(rgb_logits, dim=-1)
            flow_probs = torch.softmax(flow_logits, dim=-1)
            
            # Average the probability distributions (Late Fusion)
            fused_probs = (rgb_probs + flow_probs) / 2.0
            
            _, preds = torch.max(fused_probs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
    return correct / total

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ------------------- MODE 1: LATE FUSION EVALUATION -------------------
    if args.evaluate_fusion:
        print("\n=== Starting Late Fusion Evaluation (Task 1.3) ===")
        
        # 1. Instantiate aligned validation loaders
        val_rgb_dataset = miniUCFTSNDataset(args.val_list, args.classes, modality='RGB', mode='val')
        val_flow_dataset = miniUCFTSNDataset(args.val_list, args.classes, modality='Flow', mode='val')
        
        val_rgb_loader = DataLoader(val_rgb_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        val_flow_loader = DataLoader(val_flow_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        # 2. Re-instantiate RGB and Flow models
        rgb_model = TSN(num_classes=25, modality='RGB', init_strategy='random').to(device)
        flow_model = TSN(num_classes=25, modality='Flow', init_strategy='random').to(device)
        
        # 3. Load checkpoints
        rgb_ckpt = "best_tsn_RGB_imagenet.pth"
        flow_ckpt = "best_tsn_Flow_imagenet.pth"
        
        if not (os.path.exists(rgb_ckpt) and os.path.exists(flow_ckpt)):
            print(f"Error: Missing checkpoint files. Ensure {rgb_ckpt} and {flow_ckpt} exist in the current folder.")
            return
            
        rgb_model.load_state_dict(torch.load(rgb_ckpt, map_location=device))
        flow_model.load_state_dict(torch.load(flow_ckpt, map_location=device))
        print("Successfully loaded pre-trained RGB and Flow model weights.")
        
        # 4. Read class names for display
        class_names = []
        with open(args.classes, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    class_names.append(parts[1])
                    
        # 5. Compute per-class accuracy
        print("\nCalculating per-class accuracies...")
        rgb_class_accs = evaluate_per_class_accuracy(rgb_model, val_rgb_loader, device)
        flow_class_accs = evaluate_per_class_accuracy(flow_model, val_flow_loader, device)
        
        # Print comparison table
        print("-" * 65)
        print(f"{'Class Name':<25} | {'RGB Acc (%)':<15} | {'Flow Acc (%)':<15}")
        print("-" * 65)
        for idx, name in enumerate(class_names):
            print(f"{name:<25} | {rgb_class_accs[idx]*100:<15.2f} | {flow_class_accs[idx]*100:<15.2f}")
        print("-" * 65)
        
        # 6. Execute Late Fusion
        print("\nExecuting Late Fusion (averaging test probabilities)...")
        fusion_accuracy = run_late_fusion(rgb_model, flow_model, val_rgb_loader, val_flow_loader, device)
        print(f"=== Late Fusion Overall Validation Accuracy: {fusion_accuracy*100:.2f}% ===\n")
        return

    # ------------------- MODE 2: STANDARD TSN TRAINING -------------------
    print(f"Training Config | Modality: {args.modality} | Init: {args.init_strategy.upper()}")
    
    train_dataset = miniUCFTSNDataset(args.train_list, args.classes, modality=args.modality, mode='train')
    val_dataset = miniUCFTSNDataset(args.val_list, args.classes, modality=args.modality, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    
    # Initialize model
    model = TSN(num_classes=25, modality=args.modality, init_strategy=args.init_strategy)
    model = model.to(device)
    
    # Cross-entropy loss and SGD optimizer with momentum
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\nStarting TSN training, total epochs: {args.epochs}")
    print("-" * 50)
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        epoch_time = time.time() - start_time
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {100.0 * train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {100.0 * val_acc:.2f}%")
              
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"best_tsn_{args.modality}_{args.init_strategy}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best checkpoint to: {save_path}")
            
    print("-" * 50)
    print(f"TSN Training complete. Best Accuracy: {100.0 * best_val_acc:.2f}%")
    
    # Save training logs
    log_file = f"log_tsn_{args.modality}_{args.init_strategy}.txt"
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
        for i in range(args.epochs):
            f.write(f"{i+1},{history['train_loss'][i]:.6f},{history['train_acc'][i]:.6f},"
                    f"{history['val_loss'][i]:.6f},{history['val_acc'][i]:.6f}\n")
    print(f"Training logs exported to: {log_file}\n")

if __name__ == '__main__':
    main()