import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tqdm import tqdm
import numpy as np
import itertools

from architectures.resnet import ResNet18
from architectures.preactresnet import PRN18
from architectures.ti_preactresnet import TIPRN18

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def KLD(p, q):
    return nn.KLDivLoss(reduction="sum")(
        F.log_softmax(p, dim=1), F.softmax(q, dim=1) + 1e-8
    ) * (1.0 / p.size(0))

def CE_to_U(logits):
    return -(logits.mean(1) - torch.logsumexp(logits, dim=1)).mean()

def _get_combined_loader(ID_train_loader, OE_loader):
    if OE_loader:
        # Reset OE loader to start at a random point
        OE_loader.dataset.offset = np.random.randint(len(OE_loader.dataset))
        return zip(ID_train_loader, itertools.cycle(OE_loader))
    else:
        return ID_train_loader

def _prepare_batch(data, OE_loader, device):
    if OE_loader:
        (X, y), (X_OE, _) = data
        X = torch.cat([X, X_OE], dim=0)
    else:
        X, y = data

    return X.to(device), y.to(device)

@torch.no_grad()
def update_bn(model, ma_model):
    for module1, module2 in zip(ma_model.modules(), model.modules()):
        if isinstance(module1, torch.nn.modules.batchnorm._BatchNorm):
            module1.running_mean.copy_(module2.running_mean)
            module1.running_var.copy_(module2.running_var)
            module1.num_batches_tracked.copy_(module2.num_batches_tracked)

def ema_update(model, ema_model, decay, global_step, warmup_steps, dynamic_decay=True):
    factor = int(global_step >= warmup_steps)
    if dynamic_decay:
        delta = global_step - warmup_steps
        decay = min(decay, (1. + delta) / (10. + delta)) if 10. + delta != 0 else decay
    decay = decay * factor
    for p_ema, p_model in zip(ema_model.parameters(), model.parameters()):
        p_ema.data = decay * p_ema.data + (1 - decay) * p_model.data

def train_epoch(model, std_model, ID_train_loader, OE_loader, optimizer, scheduler, ma_params, device, args):
    model.train()
    total_loss = 0
    n_correct = 0
    combined_loader = _get_combined_loader(ID_train_loader, OE_loader)

    for batch_idx, data in tqdm(enumerate(combined_loader)):
        X, y = _prepare_batch(data, OE_loader, device)
        optimizer.zero_grad()
        loss, _ = multi_loss(model, X, y, epsilon=args.epsilon, beta=args.beta, 
                             gamma=args.gamma, eta=args.eta, steps=args.steps, 
                             device=device, std_model=std_model, 
                             OE=OE_loader is not None, loss_type=args.loss_type)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update moving average model if needed
        if ma_params:
            global_step = (ma_params["epoch"] - 1) * ma_params["update_steps"] + batch_idx
            ema_update(model, ma_params["ma_model"], ma_params["decay"], global_step, ma_params["warmup_steps"])

        total_loss += loss.item()
        pred = model(X[:len(y)]).argmax(dim=1, keepdim=True)
        n_correct += pred.eq(y.view_as(pred)).sum().item()
        if batch_idx % args.report_freq == 0:
            print(f"Batch {batch_idx}/{len(ID_train_loader)}, Loss: {loss.item():.4f}")

    return total_loss / len(ID_train_loader), 100.0 * n_correct / len(ID_train_loader.dataset)

def validate(model, val_loader, args, device):
    model.eval()
    val_loss = 0
    correct = 0
    for data, target in tqdm(val_loader):
        data, target = data.to(device), target.to(device)
        if not args.loss_type == "STD":
            # perfrom adversarial attack on validation samples
            data = generate_adversarial_examples(
                model, data, target, epsilon=args.epsilon, steps=args.steps, attack_type="CE", device=device
            )
        output = model(data)
        val_loss += F.cross_entropy(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_loader)
    accuracy = 100.0 * correct / len(val_loader.dataset)
    return val_loss, accuracy

def generate_adversarial_examples(
    model,
    X,
    y=None,
    epsilon=8/255,
    steps=10,
    attack_type="CE",
    detection_attack=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    seed=None,
    clip=True,
):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    alpha = 2.5 * epsilon / steps

    random = torch.FloatTensor(*X.shape).uniform_(-epsilon, epsilon).to(device)
    X_adv = X.detach() + random
    X_adv.requires_grad_()

    best_loss = float("-inf")
    best_adv = None

    for i in range(steps):
        if detection_attack == "OOD->ID":
            logits = model(X_adv)
            loss = CE_to_U(logits)
        elif detection_attack == "ID->OOD":
            logits = model(X_adv)
            loss = -CE_to_U(logits)
        elif attack_type == "CE":
            loss = F.cross_entropy(model(X_adv), y)
        elif attack_type == "KLD":
            loss = KLD(
                model(X_adv),
                model(X)
            )
        else:
            raise ValueError(f"Unknown attack configuration: attack_type={attack_type}, detection_attack={detection_attack}")

        if loss.item() > best_loss:
            best_loss = loss.item()
            best_adv = X_adv.detach()

        grad = torch.autograd.grad(loss, [X_adv])[0]
        
        X_adv.data = X_adv.data + alpha * grad.sign()
        X_adv = torch.max(torch.min(X_adv, X + epsilon), X - epsilon)
        if clip:
            X_adv = torch.clamp(X_adv, 0, 1)

    return best_adv


def multi_loss(
    model,
    X,
    y,
    epsilon=8/255,
    beta=2.5,
    gamma=0.5,
    eta=0.5,
    steps=10,
    device=None,
    std_model=None,
    OE=False,
    loss_type="HALO",
):
    if device is None:
        device = X.device

    # calculate std CE loss
    logits = model(X)
    std_loss = F.cross_entropy(logits[:len(y)], y)
    
    # if OE is enabled, add OE loss for OE samples (last half of the batch)
    if OE:
        std_loss += eta * CE_to_U(logits[len(y):])

    if loss_type in ["STD", "OE"]:
        return std_loss, logits

    # calculate adversarial examples and adversarial loss
    X_adv = generate_adversarial_examples(
        model, X, y, epsilon, steps, "KLD", device=device
    )
    adv_logits = model(X_adv)
    adv_loss = KLD(adv_logits[:len(y)], logits[:len(y)])
    
    # if OE is enabled, add loss for attacked OE samples
    if OE:
        adv_loss += eta * KLD(adv_logits[len(y):], logits[len(y):])

    if loss_type in ["TRADES", "TRADES_OE"]:
        return std_loss + beta * adv_loss, logits
    
    assert loss_type == "HALO", "Invalid loss type"

    # calculate helper (HAT) loss
    if not std_model:
        raise ValueError("std_model must be provided for HAT loss")
    
    perturbation = X_adv - X
    X_helper = X + 2 * perturbation
    
    with torch.no_grad():
        logits_std = std_model(X_adv)
        pred_std = logits_std[:len(y)].argmax(dim=1)
    
    logits_hat = model(X_helper)
    helper_loss = F.cross_entropy(logits_hat[:len(y)], pred_std)
    return std_loss + beta * adv_loss + gamma * helper_loss, logits


def get_model(model_name, dataset="CIFAR10"):
    """
    Get and instantiate a model based on the model name and dataset.
    """
    model_classes = {
        "ResNet18": ResNet18,
        "PreActResNet18": PRN18,
        "TI-PreActResNet18": TIPRN18,
    }
    num_classes = {"CIFAR10": 10, "CIFAR100": 100, "TINYIMAGENET": 200}.get(dataset.upper(), 10)

    if model_name in model_classes:
        return model_classes[model_name](num_classes=num_classes)
    raise ValueError(f"Invalid model name: {model_name}")