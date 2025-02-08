import os
import time
from util import get_model, train_epoch, validate, set_seed
from data_util import get_datasets
import torch
import torch.optim as optim
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--id_dataset', type=str, default='cifar10', help='ID dataset name')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--report_freq', type=int, default=100, help='report frequency in steps')
    parser.add_argument('--max_lr', type=float, default=0.21, help='max learning rate for one cycle policy')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--epsilon', type=float, default=8/255, help='perturbation size')
    parser.add_argument('--beta', type=float, default=3.0, help='weight for KLD loss')
    parser.add_argument('--gamma', type=float, default=0.5, help='weight for helper loss')
    parser.add_argument('--eta', type=float, default=2.0, help='weight for OE loss')
    parser.add_argument('--steps', type=int, default=10, help='PGD attack steps')
    parser.add_argument('--loss_type', type=str, default='HALO', help='loss type: STD, OE, TRADES, TRADES_OE, or HALO')
    parser.add_argument('--model', type=str, default='ResNet18', help='model architecture')
    parser.add_argument('--oe_dataset', type=str, default='TIN597', help='OE dataset')
    parser.add_argument('--std_model_path', type=str, default=None, help='path to standard model for HALO')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--run_name', type=str, default='HALO', help='run name')
    parser.add_argument('--verbose', action='store_true', help='print training details')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed(args.seed)

    train_loader, val_loader, _, oe_loader = get_datasets(args.id_dataset, args.batch_size, args.oe_dataset, verbose=args.verbose)

    if args.loss_type in ['STD', 'TRADES']:
        oe_loader = None

    model = get_model(args.model, args.id_dataset).to(device)

    # create folder in training_runs
    if not os.path.exists(f'training_runs/{args.run_name}'):
        os.makedirs(f'training_runs/{args.run_name}')

    # create a json file with the arguments
    with open(f'training_runs/{args.run_name}/args.json', 'w') as f:
        json.dump(vars(args), f)
    
    # create a json file to store training history
    history = {}
    with open(f'training_runs/{args.run_name}/history.json', 'w') as f:
        json.dump(history, f)
    
    # load standard (helper) model if using HALO
    std_model = None
    if args.loss_type == 'HALO':
        try:
            std_model = get_model(args.model, args.id_dataset).to(device)
            std_model.load_state_dict(torch.load(args.std_model_path))
        except:
            raise ValueError('Invalid standard model path for HALO')

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr, 
        anneal_strategy='cos',
        pct_start=0.25,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
    )
    
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch(model, std_model, train_loader, oe_loader, optimizer, scheduler, None, device, args)
        val_loss, val_accuracy = validate(model, val_loader, args=args, device=device)

        print(f'Epoch {epoch}/{args.epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Accuracy: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.2f}%',
              f'Time: {time.time() - epoch_start_time:.2f}s')
        history[epoch] = {'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_accuracy}
        with open(f'training_runs/{args.run_name}/history.json', 'w') as f:
            json.dump(history, f)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'training_runs/{args.run_name}/{args.run_name}_{args.id_dataset}_best.pth')
            print(f'Checkpoint saved with Val Loss: {val_loss:.4f}')

    print(f'Final Val Loss: {val_loss:.4f}, Final Val Accuracy: {val_accuracy:.2f}%')

    # Save the model
    torch.save(model.state_dict(), f'training_runs/{args.run_name}/{args.run_name}_{args.id_dataset}_last.pth')



if __name__ == '__main__':
    main()