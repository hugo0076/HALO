import argparse
import json
from util import get_model
from data_util import get_datasets
from openood_adv.model_wrapper import ModelWrapper
from openood_adv.utils import AdvEvaluator, BasePostprocessor
import torch
from tqdm import tqdm


def test_classification(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    correct = 0
    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(test_loader.dataset)

def test_ood(model, id_dataset, epsilon=8/255, n_steps=40):
    wrapped_model = ModelWrapper(model, w_features=False, id_dataset=id_dataset)
    postprocessor = BasePostprocessor({}) # use MSP as the postprocessor
    postprocessor.APS_mode = False
    evaluator = AdvEvaluator(
            wrapped_model,
            id_name=id_dataset,                   
            data_root='./data',                   
            config_root=None,                     
            preprocessor=None,
            postprocessor=postprocessor,
        )
    results = evaluator.eval_ood(fsood=False, progress=False, epsilon=epsilon, n_steps=n_steps)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Model')
    parser.add_argument('--id_dataset', type=str, default='cifar10', help='ID dataset name')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--epsilon', type=float, default=8/255, help='perturbation size')
    parser.add_argument('--steps', type=int, default=40, help='PGD attack steps')
    parser.add_argument('--run_name', type=str, default='HALO', help='run name')
    parser.add_argument('--verbose', action='store_true', help='print training details')
    parser.add_argument('--model', type=str, default='ResNet18', help='model architecture')
    args = parser.parse_args()
    if args.verbose:
        print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = get_model(args.model, args.id_dataset)
    # load model weights
    model.load_state_dict(torch.load(f'training_runs/{args.run_name}/{args.run_name}_{args.id_dataset}_best.pth'))
    model.eval()
    model.to(device)

    # load data
    _, _, test_loader, _ = get_datasets(args.id_dataset, args.batch_size, None, verbose=args.verbose)

    # test classification accuracy
    test_acc = test_classification(model, test_loader)
    if args.verbose:
        print(f'Test Accuracy: {test_acc}')

    # test OOD detection
    ood_metrics = test_ood(model, args.id_dataset, epsilon=args.epsilon, n_steps=args.steps)
    ood_metrics = ood_metrics.to_dict()
    if args.verbose:
        print(f'OOD Detection Metrics: {ood_metrics}')

    # save to json
    with open(f'training_runs/{args.run_name}/test_results.json', 'w') as f:
        json.dump({'test_acc': test_acc, 'ood_metrics': ood_metrics}, f)



    
