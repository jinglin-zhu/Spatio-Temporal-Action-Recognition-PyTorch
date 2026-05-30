import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import miniUCF3DDataset
from model import ResNet18_3D

def parse_args():
    parser = argparse.ArgumentParser(description="miniUCF 3D ResNet-18 Action Recognition")
    
    # path configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    parser.add_argument('--train_list', type=str, 
                        default=os.path.join(project_root, 'data', 'train.txt'),
                        help='training list path')
    parser.add_argument('--val_list', type=str, 
                        default=os.path.join(project_root, 'data', 'validation.txt'),
                        help='validation list path')
    parser.add_argument('--classes', type=str, 
                        default=os.path.join(project_root, 'data', 'classes.txt'),
                        help='classes list path')
    
    # hyperparameters and initialization settings
    parser.add_argument('--init_strategy', type=str, default='inflate', choices=['random', 'inflate'],
                        help="weight initialization strategies: 'random' or 'inflate'")
    parser.add_argument('--epochs', type=int, default=15, help='total training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--clip_len', type=int, default=16, help='clip length')
    parser.add_argument('--num_views', type=int, default=4, help='number of views in validation')
    parser.add_argument('--num_workers', type=int, default=4, help='number of DataLoader threads')
    
    return parser.parse_args()

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Execute a single-epoch training loop on the 3D network.

    Args:
        model (nn.Module): The 3D ResNet model to be trained.
        dataloader (DataLoader): PyTorch DataLoader providing training batches of (clips, labels).
        criterion (nn.Module): The loss function (CrossEntropyLoss).
        optimizer (optim.Optimizer): PyTorch optimizer (SGD).
        device (torch.device): Computation hardware platform device (e.g., 'cuda' or 'cpu').

    Returns:
        tuple: A tuple containing:
            - epoch_loss (float): Average training loss calculated for this epoch.
            - epoch_acc (float): Average training top-1 classification accuracy for this epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="  [Train]", mininterval=2.0, leave=False)
    for clips, labels in progress_bar:
        # clips: [Batch, 3, clip_len, 224, 224]
        clips = clips.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # 3D ResNet forward
        outputs = model(clips)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * clips.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += clips.size(0)
        
        # update progress bar display
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0 * correct / total:.2f}%")
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_epoch_multiview(model, dataloader, criterion, device, num_views):
    """
    Perform validation using Multi-view Temporal consensus testing [1, 3, 19].

    Args:
        model (nn.Module): The 3D ResNet model to be evaluated.
        dataloader (DataLoader): PyTorch DataLoader providing validation batches of (stacked_views, labels).
        criterion (nn.Module): The loss function (CrossEntropyLoss).
        device (torch.device): Computation hardware platform device (e.g., 'cuda' or 'cpu').
        num_views (int): Number of non-overlapping temporal views extracted per video.

    Returns:
        tuple: A tuple containing:
            - val_loss (float): Average validation loss calculated using integrated multi-view probabilities.
            - val_acc (float): Average top-1 accuracy calculated using integrated multi-view consensus.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="  [Val MV]", mininterval=2.0, leave=False)
        for stacked_views, labels in progress_bar:
            # stacked_views: [Batch, num_views, 3, clip_len, 224, 224]
            # labels: [Batch]
            batch_size = stacked_views.size(0)
            labels = labels.to(device)
            
            # to accelerate computation, merge batch and views dimensions
            # merged shape: [Batch * num_views, 3, clip_len, 224, 224]
            flat_views = stacked_views.view(-1, 3, stacked_views.size(3), stacked_views.size(4), stacked_views.size(5))
            flat_views = flat_views.to(device)
            
            # get predictive outputs from all views (shape: [Batch * num_views, num_classes])
            flat_outputs = model(flat_views)
            
            # restore dimensions to: [Batch, num_views, num_classes]
            outputs = flat_outputs.view(batch_size, num_views, -1)
            
            # for each video, compute the Softmax probability distribution across multi-views and take the mean
            probs = torch.softmax(outputs, dim=-1) # calculate probabilities along the category dimension
            mean_probs = torch.mean(probs, dim=1)  # compute the mean along the view dimension, shape: [Batch, num_classes]
            
            # calculate the loss using the integrated average probability
            eps = 1e-12
            loss = nn.NLLLoss()(torch.log(mean_probs + eps), labels)
            
            running_loss += loss.item() * batch_size
            _, preds = torch.max(mean_probs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += batch_size
            
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0 * correct / total:.2f}%")
            
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Current Initialization Strategy: {args.init_strategy.upper()}")
    
    # 1. instantiate datasets and dataLoaders
    train_dataset = miniUCF3DDataset(data_list_path=args.train_list, 
                                     classes_path=args.classes, 
                                     clip_len=args.clip_len, 
                                     mode='train')
    
    val_dataset = miniUCF3DDataset(data_list_path=args.val_list, 
                                   classes_path=args.classes, 
                                   clip_len=args.clip_len, 
                                   mode='val', 
                                   num_views=args.num_views)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    
    # 2. initialize the manually implemented 3D ResNet-18
    model = ResNet18_3D(num_classes=25, init_strategy=args.init_strategy)
    model = model.to(device)
    
    # 3. optimizers and loss functions
    # use SGD with momentum, combined with weight decay to prevent overfitting
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # use cosine annealing learning rate scheduler to achieve smoother convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    
    # for logging and plotting
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print(f"\nStart training, total epochs:{args.epochs}")
    print("-" * 50)
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # train for one round
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        # one round of multi-view verification
        val_loss, val_acc = validate_epoch_multiview(model, val_loader, criterion, device, args.num_views)
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # log history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {100.0 * train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc (MV): {100.0 * val_acc:.2f}%")
        
        # save best weights
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"best_3dresnet_{args.init_strategy}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  --> A better model has been found and saved to {save_path}")
            
    print("-" * 50)
    print(f"Training complete! Best validation set accuracy (multi-view): {100.0 * best_val_acc:.2f}%")
    
    # export training logs to the current directory for subsequent plotting
    log_file = f"log_3dresnet_{args.init_strategy}.txt"
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
        for i in range(args.epochs):
            f.write(f"{i+1},{history['train_loss'][i]:.6f},{history['train_acc'][i]:.6f},"
                    f"{history['val_loss'][i]:.6f},{history['val_acc'][i]:.6f}\n")
    print(f"Historical data has been saved to: {log_file}")

if __name__ == '__main__':
    main()