from typing import Callable, List, Type, Any

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.evaluators.metrics import compute_all_metrics

from openood.evaluation_api.datasets import DATA_INFO, data_setup, get_id_ood_dataloader
from openood.evaluation_api.postprocessor import get_postprocessor
from openood.evaluation_api.preprocessor import get_default_preprocessor

import openood.utils.comm as comm

NORM_VALUES = {
    'cifar10': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
    'cifar100': [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
    'tinyimagenet': [[0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]],
}

def KLD(p, q):
    return nn.KLDivLoss(reduction="sum")(
        F.log_softmax(p, dim=1), F.softmax(q, dim=1) + 1e-8
    ) * (1.0 / p.size(0))

def CE_to_U(logits):
    return -(logits.mean(1) - torch.logsumexp(logits, dim=1)).mean()

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

class BasePostprocessor:
    def __init__(self, config):
        self.config = config

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(
        self,
        net: nn.Module,
        data_loader: DataLoader,
        progress: bool = True,
        attack=False,
        epsilon=0.0,
        n_steps=10,
        id_dataset="cifar10",
    ):
        pred_list, conf_list, label_list = [], [], []
        for batch in tqdm(
            data_loader, disable=not progress or not comm.is_main_process()
        ):
            data = batch["data"].cuda()
            label = batch["label"].cuda()
            # if attack is True, then we want to attack the model in the way specified
            if attack:
                # we need to make sure the data lies in the range [0, 1]
                mean = torch.tensor(NORM_VALUES[id_dataset][0]).view(1, 3, 1, 1).to(data.device)
                std = torch.tensor(NORM_VALUES[id_dataset][1]).view(1, 3, 1, 1).to(data.device)
                data = data * std + mean
                # put the model in non-normalised mode for attack
                net.set_normalise(False)
                if attack in ["ID->OOD", "OOD->ID"]:
                    data = generate_adversarial_examples(
                        net,
                        data,
                        label,
                        epsilon=epsilon,
                        steps=n_steps,
                        attack_type="CE",
                        detection_attack=attack,
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    )
                # put the model back in normalised mode
                net.set_normalise(True)
                # We need to convert the data back to its original range
                data = (data - mean) / std

            pred, conf = self.postprocess(net, data)

            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())

        # convert values into numpy array
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        return pred_list, conf_list, label_list

class AdvEvaluator:
    def __init__(
        self,
        net: nn.Module,
        id_name: str,
        data_root: str = './data',
        config_root: str = './configs',
        preprocessor: Callable = None,
        postprocessor_name: str = None,
        postprocessor: Type[BasePostprocessor] = None,
        batch_size: int = 200,
        shuffle: bool = False,
        num_workers: int = 4,
    ) -> None:
        """A unified, easy-to-use API for evaluating (most) discriminative OOD
        detection methods with adversarial attacks.

        Args:
            net (nn.Module):
                The base classifier.
            id_name (str):
                The name of the in-distribution dataset.
            data_root (str, optional):
                The path of the data folder. Defaults to './data'.
            config_root (str, optional):
                The path of the config folder. Defaults to './configs'.
            preprocessor (Callable, optional):
                The preprocessor of input images.
                Passing None will use the default preprocessor
                following convention. Defaults to None.
            postprocessor_name (str, optional):
                The name of the postprocessor that obtains OOD score.
                Ignored if an actual postprocessor is passed.
                Defaults to None.
            postprocessor (Type[BasePostprocessor], optional):
                An actual postprocessor instance which inherits
                OpenOOD's BasePostprocessor. Defaults to None.
            batch_size (int, optional):
                The batch size of samples. Defaults to 200.
            shuffle (bool, optional):
                Whether shuffling samples. Defaults to False.
            num_workers (int, optional):
                The num_workers argument that will be passed to
                data loaders. Defaults to 4.

        Raises:
            ValueError:
                If both postprocessor_name and postprocessor are None.
            ValueError:
                If the specified ID dataset {id_name} is not supported.
            TypeError:
                If the passed postprocessor does not inherit BasePostprocessor.
        """
        # check the arguments
        if postprocessor_name is None and postprocessor is None:
            raise ValueError('Please pass postprocessor_name or postprocessor')
        if postprocessor_name is not None and postprocessor is not None:
            print(
                'Postprocessor_name is ignored because postprocessor is passed'
            )
        if id_name not in DATA_INFO:
            raise ValueError(f'Dataset [{id_name}] is not supported')

        # get data preprocessor
        if preprocessor is None:
            preprocessor = get_default_preprocessor(id_name)

        # set up config root
        if config_root is None:
            filepath = os.path.dirname(os.path.abspath(__file__))
            config_root = os.path.join(*filepath.split('/')[:-2], 'configs')

        # get postprocessor
        if postprocessor is None:
            postprocessor = get_postprocessor(config_root, postprocessor_name,
                                              id_name)
        # if not isinstance(postprocessor, BasePostprocessor):
        #     raise TypeError(
        #         'postprocessor should inherit BasePostprocessor in OpenOOD')

        # load data
        data_setup(data_root, id_name)
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers
        }
        dataloader_dict = get_id_ood_dataloader(id_name, data_root,
                                                preprocessor, **loader_kwargs)

        # postprocessor setup
        postprocessor.setup(net, dataloader_dict['id'], dataloader_dict['ood'])

        self.id_name = id_name
        self.net = net
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.dataloader_dict = dataloader_dict
        self.metrics = {
            'id_acc': None,
            'csid_acc': None,
            'ood': None,
            'fsood': None
        }
        self.scores = {
            'id': {
                'train': None,
                'val': None,
                'test': None
            },
            'csid': {k: None
                     for k in dataloader_dict['csid'].keys()},
            'ood': {
                'val': None,
                'near':
                {k: None
                 for k in dataloader_dict['ood']['near'].keys()},
                'far': {k: None
                        for k in dataloader_dict['ood']['far'].keys()},
            },
            'id_preds': None,
            'id_labels': None,
            'csid_preds': {k: None
                           for k in dataloader_dict['csid'].keys()},
            'csid_labels': {k: None
                            for k in dataloader_dict['csid'].keys()},
        }
        # perform hyperparameter search if have not done so
        if (self.postprocessor.APS_mode
                and not self.postprocessor.hyperparam_search_done):
            self.hyperparam_search()

        self.net.eval()

        # how to ensure the postprocessors can work with
        # models whose definition doesn't align with OpenOOD

    def _classifier_inference(self,
                              data_loader: DataLoader,
                              msg: str = 'Acc Eval',
                              progress: bool = True):
        self.net.eval()

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=msg, disable=not progress):
                data = batch['data'].cuda()
                logits = self.net(data)
                preds = logits.argmax(1)
                all_preds.append(preds.cpu())
                all_labels.append(batch['label'])

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        return all_preds, all_labels

    def eval_acc(self, data_name: str = 'id') -> float:
        if data_name == 'id':
            if self.metrics['id_acc'] is not None:
                return self.metrics['id_acc']
            else:
                if self.scores['id_preds'] is None:
                    all_preds, all_labels = self._classifier_inference(
                        self.dataloader_dict['id']['test'], 'ID Acc Eval')
                    self.scores['id_preds'] = all_preds
                    self.scores['id_labels'] = all_labels
                else:
                    all_preds = self.scores['id_preds']
                    all_labels = self.scores['id_labels']

                assert len(all_preds) == len(all_labels)
                correct = (all_preds == all_labels).sum().item()
                acc = correct / len(all_labels) * 100
                self.metrics['id_acc'] = acc
                return acc
        elif data_name == 'csid':
            if self.metrics['csid_acc'] is not None:
                return self.metrics['csid_acc']
            else:
                correct, total = 0, 0
                for _, (dataname, dataloader) in enumerate(
                        self.dataloader_dict['csid'].items()):
                    if self.scores['csid_preds'][dataname] is None:
                        all_preds, all_labels = self._classifier_inference(
                            dataloader, f'CSID {dataname} Acc Eval')
                        self.scores['csid_preds'][dataname] = all_preds
                        self.scores['csid_labels'][dataname] = all_labels
                    else:
                        all_preds = self.scores['csid_preds'][dataname]
                        all_labels = self.scores['csid_labels'][dataname]

                    assert len(all_preds) == len(all_labels)
                    c = (all_preds == all_labels).sum().item()
                    t = len(all_labels)
                    correct += c
                    total += t

                if self.scores['id_preds'] is None:
                    all_preds, all_labels = self._classifier_inference(
                        self.dataloader_dict['id']['test'], 'ID Acc Eval')
                    self.scores['id_preds'] = all_preds
                    self.scores['id_labels'] = all_labels
                else:
                    all_preds = self.scores['id_preds']
                    all_labels = self.scores['id_labels']

                correct += (all_preds == all_labels).sum().item()
                total += len(all_labels)

                acc = correct / total * 100
                self.metrics['csid_acc'] = acc
                return acc
        else:
            raise ValueError(f'Unknown data name {data_name}')

    def eval_ood(self, fsood: bool = False, progress: bool = True, epsilon: float = 0.0, n_steps: int = 10) -> pd.DataFrame:
        id_name = 'id' if not fsood else 'csid'
        task = 'ood' if not fsood else 'fsood'
        if self.metrics[task] is None:
            self.net.eval()

            # id score
            if self.scores['id']['test'] is None or self.scores['id']['adv_test'] is None:
                print(f'Performing inference on {self.id_name} test set...',
                      flush=True)
                id_pred, id_conf, id_gt = self.postprocessor.inference(
                    self.net, self.dataloader_dict['id']['test'], progress)
                # perform inference on id test set with ID -> OOD attack
                print(f'Performing inference on {self.id_name} test set with ID -> OOD attack...',
                      flush=True)
                adv_id_pred, adv_id_conf, adv_id_gt = self.postprocessor.inference(
                    self.net, self.dataloader_dict['id']['test'], progress, attack = 'ID->OOD', epsilon=epsilon, n_steps=n_steps, id_dataset=self.id_name)
            
                self.scores['id']['test'] = [id_pred, id_conf, id_gt]
                self.scores['id']['adv_test'] = [adv_id_pred, adv_id_conf, adv_id_gt]
            else:
                id_pred, id_conf, id_gt = self.scores['id']['test']
                adv_id_pred, adv_id_conf, adv_id_gt = self.scores['id']['adv_test']

            if fsood:
                csid_pred, csid_conf, csid_gt = [], [], []
                for i, dataset_name in enumerate(self.scores['csid'].keys()):
                    if self.scores['csid'][dataset_name] is None:
                        print(
                            f'Performing inference on {self.id_name} '
                            f'(cs) test set [{i+1}]: {dataset_name}...',
                            flush=True)
                        temp_pred, temp_conf, temp_gt = \
                            self.postprocessor.inference(
                                self.net,
                                self.dataloader_dict['csid'][dataset_name],
                                progress)
                        self.scores['csid'][dataset_name] = [
                            temp_pred, temp_conf, temp_gt
                        ]

                    csid_pred.append(self.scores['csid'][dataset_name][0])
                    csid_conf.append(self.scores['csid'][dataset_name][1])
                    csid_gt.append(self.scores['csid'][dataset_name][2])

                csid_pred = np.concatenate(csid_pred)
                csid_conf = np.concatenate(csid_conf)
                csid_gt = np.concatenate(csid_gt)

                id_pred = np.concatenate((id_pred, csid_pred))
                id_conf = np.concatenate((id_conf, csid_conf))
                id_gt = np.concatenate((id_gt, csid_gt))

            # load nearood data and compute ood metrics
            near_metrics = self._eval_ood([id_pred, id_conf, id_gt],
                                          ood_split='near',
                                          progress=progress)
            # load farood data and compute ood metrics
            far_metrics = self._eval_ood([id_pred, id_conf, id_gt],
                                         ood_split='far',
                                         progress=progress)


            # compute metrics for nearood and farood when ID -> OOD attack is performed
            near_metrics_ID_OOD = self._eval_ood([adv_id_pred, adv_id_conf, adv_id_gt],
                                            ood_split='near',
                                            progress=progress)
            
            far_metrics_ID_OOD = self._eval_ood([adv_id_pred, adv_id_conf, adv_id_gt],
                                            ood_split='far',
                                            progress=progress)
            
            # compute metrics for nearood and farood when OOD -> ID attack is performed
            near_metrics_OOD_ID = self._eval_ood([id_pred, id_conf, id_gt],
                                            ood_split='near',
                                            progress=progress,
                                            adversarial=True,
                                            epsilon=epsilon,
                                            n_steps=n_steps)
            far_metrics_OOD_ID = self._eval_ood([id_pred, id_conf, id_gt],
                                            ood_split='far',
                                            progress=progress,
                                            adversarial=True,
                                            epsilon=epsilon,
                                            n_steps=n_steps)
            
            # compute metrics for nearood and farood when both attacks are performed
            near_metrics_both = self._eval_ood([adv_id_pred, adv_id_conf, adv_id_gt],
                                            ood_split='near',
                                            progress=progress,
                                            adversarial=True,
                                            epsilon=epsilon,
                                            n_steps=n_steps)
            far_metrics_both = self._eval_ood([adv_id_pred, adv_id_conf, adv_id_gt],
                                            ood_split='far',
                                            progress=progress,
                                            adversarial=True,
                                            epsilon=epsilon,
                                            n_steps=n_steps)
            
            
            # concatenate all metrics across attack settings, excluding the accuracy
            near_metrics = np.concatenate([near_metrics, near_metrics_ID_OOD, near_metrics_OOD_ID, near_metrics_both], axis=1)
            far_metrics = np.concatenate([far_metrics, far_metrics_ID_OOD, far_metrics_OOD_ID, far_metrics_both], axis=1)

            self.metrics[task] = pd.DataFrame(
                np.concatenate([near_metrics, far_metrics], axis=0),
                index=list(self.dataloader_dict['ood']['near'].keys()) +
                ['nearood'] + list(self.dataloader_dict['ood']['far'].keys()) +
                ['farood'],
                columns=['FPR@95', 'AUROC', 'AUPR_IN', 'AUPR_OUT',
                        'FPR@95_ID_OOD', 'AUROC_ID_OOD', 'AUPR_IN_ID_OOD', 'AUPR_OUT_ID_OOD',
                        'FPR@95_OOD_ID', 'AUROC_OOD_ID', 'AUPR_IN_OOD_ID', 'AUPR_OUT_OOD_ID',
                        'FPR@95_both', 'AUROC_both', 'AUPR_IN_both', 'AUPR_OUT_both']
            )
        else:
            print('Evaluation has already been done!')

        with pd.option_context(
                'display.max_rows', None, 'display.max_columns', None,
                'display.float_format',
                '{:,.2f}'.format):  # more options can be specified also
            print(self.metrics[task])

        return self.metrics[task]

    def _eval_ood(self,
                  id_list: List[np.ndarray],
                  ood_split: str = 'near',
                  progress: bool = True,
                  adversarial: bool = False,
                  epsilon: float = 0.0,
                  n_steps: int = 10) -> np.ndarray:
        print(f'Processing {ood_split} ood...', flush=True)
        print(f'OOD->ID attack? {adversarial}', flush=True)
        [id_pred, id_conf, id_gt] = id_list
        metrics_list = []
        for dataset_name, ood_dl in self.dataloader_dict['ood'][
                ood_split].items():
            # if adversarial is true we are looking for/creating f"{dataset_name}_adv" rather than just dataset_name
            dataset_name = f"{dataset_name}_adv" if adversarial else dataset_name
            if self.scores['ood'][ood_split].get(dataset_name, None) is None:
                attack = 'OOD->ID' if adversarial else None
                print(f'Performing inference on {dataset_name} dataset with attack = {attack}...', flush=True)
                ood_pred, ood_conf, ood_gt = self.postprocessor.inference(
                    self.net, ood_dl, progress, attack, epsilon, n_steps, id_dataset=self.id_name)
                self.scores['ood'][ood_split][dataset_name] = [
                    ood_pred, ood_conf, ood_gt
                ]
            else:
                print(
                    'Inference has been performed on '
                    f'{dataset_name} dataset...',
                    flush=True)
                [ood_pred, ood_conf,
                 ood_gt] = self.scores['ood'][ood_split][dataset_name]

            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])

            print(f'Computing metrics on {dataset_name} dataset...')
            ood_metrics = compute_all_metrics(conf, label, pred)
            metrics_list.append(ood_metrics)
            self._print_metrics(ood_metrics[:4]) # print the metrics excluding accuracy

        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        # remove accuracy from the metrics
        metrics_list = np.delete(metrics_list, 4, axis=1)
        metrics_mean = np.mean(metrics_list, axis=0, keepdims=True)
        self._print_metrics(list(metrics_mean[0]))
        return np.concatenate([metrics_list, metrics_mean], axis=0) * 100

    def _print_metrics(self, metrics):
        [fpr, auroc, aupr_in, aupr_out] = metrics

        # print ood metric results
        print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc),
              end=' ',
              flush=True)
        print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
            100 * aupr_in, 100 * aupr_out),
              flush=True)
        print(u'\u2500' * 70, flush=True)
        print('', flush=True)

    def hyperparam_search(self):
        print('Starting automatic parameter search...')
        max_auroc = 0
        hyperparam_names = []
        hyperparam_list = []
        count = 0

        for name in self.postprocessor.args_dict.keys():
            hyperparam_names.append(name)
            count += 1

        for name in hyperparam_names:
            hyperparam_list.append(self.postprocessor.args_dict[name])

        hyperparam_combination = self.recursive_generator(
            hyperparam_list, count)

        final_index = None
        for i, hyperparam in enumerate(hyperparam_combination):
            self.postprocessor.set_hyperparam(hyperparam)

            id_pred, id_conf, id_gt = self.postprocessor.inference(
                self.net, self.dataloader_dict['id']['val'])
            ood_pred, ood_conf, ood_gt = self.postprocessor.inference(
                self.net, self.dataloader_dict['ood']['val'])

            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            auroc = ood_metrics[1]

            print('Hyperparam: {}, auroc: {}'.format(hyperparam, auroc))
            if auroc > max_auroc:
                final_index = i
                max_auroc = auroc

        self.postprocessor.set_hyperparam(hyperparam_combination[final_index])
        print('Final hyperparam: {}'.format(
            self.postprocessor.get_hyperparam()))
        self.postprocessor.hyperparam_search_done = True

    def recursive_generator(self, list, n):
        if n == 1:
            results = []
            for x in list[0]:
                k = []
                k.append(x)
                results.append(k)
            return results
        else:
            results = []
            temp = self.recursive_generator(list, n - 1)
            for x in list[n - 1]:
                for y in temp:
                    k = y.copy()
                    k.append(x)
                    results.append(k)
            return results
