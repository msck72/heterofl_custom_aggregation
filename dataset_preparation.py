from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split
from torchvision.datasets import CIFAR10, MNIST

# import hydra
# from hydra.core.hydra_config import HydraConfig
# from hydra.utils import call, instantiate
# from omegaconf import DictConfig, OmegaConf


# @hydra.main(config_path="conf", config_name="base", version_base=None)
# def download_and_preprocess(cfg: DictConfig) -> None:
#     """Does everything needed to get the dataset.

#     Parameters
#     ----------
#     cfg : DictConfig
#         An omegaconf object that stores the hydra config.
#     """

#     ## 1. print parsed config
#     print(OmegaConf.to_yaml(cfg))

#     # Please include here all the logic
#     # Please use the Hydra config style as much as possible specially
#     # for parts that can be customised (e.g. how data is partitioned)

# if __name__ == "__main__":

#     download_and_preprocess()


def _download_data(dataset_name) -> Tuple[Dataset, Dataset]:
    if dataset_name == "MNIST":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        trainset = MNIST("./dataset", train=True, download=True, transform=transform)
        testset = MNIST("./dataset", train=False, download=True, transform=transform)
    elif dataset_name == "CIFAR10":
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
        testset = CIFAR10("./dataset", train=False, download=True, transform=transform)
    else:
        raise ValueError(f"{dataset_name} is not valid")

    return trainset, testset


def _partition_data(
    num_clients,
    dataset_name,
    iid: Optional[bool] = False,
    shard_per_user=2,  # only in case of non-iid
    balance: Optional[bool] = False,
    seed: Optional[int] = 42,
) -> Tuple[List[Dataset], Dataset]:
    trainset, testset = _download_data(dataset_name)

    if dataset_name == "MNIST" or dataset_name == "CIFAR10":
        classes_size = 10
    # else:
    # dataset_classes_size needs to be calculated

    if balance:
        trainset = _balance_classes(trainset, seed)

    if iid:
        datasets, label_split = iid_partition(trainset, num_clients, seed=seed)
        client_testsets, _ = iid_partition(testset, num_clients, seed=seed)
    else:
        datasets, label_split = non_iid(
            trainset, num_clients, shard_per_user, classes_size
        )
        client_testsets, _ = non_iid(
            testset, num_clients, shard_per_user, classes_size, label_split
        )

        tensor_label_split = []
        for i in label_split:
            tensor_label_split.append(torch.Tensor(i))
        label_split = tensor_label_split

    return trainset, datasets, label_split, client_testsets, testset


def iid_partition(dataset, num_clients, seed=42) -> Tuple[List, List]:
    partition_size = int(len(dataset) / num_clients)
    lengths = [partition_size] * num_clients

    divided_dataset = random_split(
        dataset, lengths, torch.Generator().manual_seed(seed)
    )
    label_split = []
    for i in range(num_clients):
        label_split.append(
            torch.unique(torch.Tensor([target for _, target in divided_dataset[i]]))
        )

    return divided_dataset, label_split


def non_iid(
    dataset, num_clients, shard_per_user, classes_size, label_split=None, seed=42
) -> Tuple[List, List]:
    label = np.array(dataset.targets)
    data_split = {i: [] for i in range(num_clients)}
    label_idx_split = {}

    for i in range(len(label)):
        label_i = label[i].item()
        if label_i not in label_idx_split:
            label_idx_split[label_i] = []
        label_idx_split[label_i].append(i)

    shard_per_class = int(shard_per_user * num_clients / classes_size)

    for label_i in label_idx_split:
        label_idx = label_idx_split[label_i]
        num_leftover = len(label_idx) % shard_per_class
        leftover = label_idx[-num_leftover:] if num_leftover > 0 else []
        new_label_idx = (
            np.array(label_idx[:-num_leftover])
            if num_leftover > 0
            else np.array(label_idx)
        )
        new_label_idx = new_label_idx.reshape((shard_per_class, -1)).tolist()

        for i, leftover_label_idx in enumerate(leftover):
            new_label_idx[i] = np.concatenate([new_label_idx[i], [leftover_label_idx]])
        label_idx_split[label_i] = new_label_idx

    if label_split is None:
        label_split = list(range(classes_size)) * shard_per_class
        label_split = torch.tensor(label_split)[
            torch.randperm(len(label_split), generator = torch.Generator().manual_seed(seed))
        ].tolist()
        label_split = np.array(label_split).reshape((num_clients, -1)).tolist()

        for i in range(len(label_split)):
            label_split[i] = np.unique(label_split[i]).tolist()

    for i in range(num_clients):
        for label_i in label_split[i]:
            idx = torch.arange(len(label_idx_split[label_i]))[
                torch.randperm(len(label_idx_split[label_i]) , generator = torch.Generator().manual_seed(seed))[0]
            ].item()
            data_split[i].extend(label_idx_split[label_i].pop(idx))

    divided_dataset = [None for i in range(num_clients)]
    for i in range(num_clients):
        divided_dataset[i] = Subset(dataset, data_split[i])


    return divided_dataset, label_split


def _balance_classes(
    trainset: Dataset,
    seed: Optional[int] = 42,
) -> Dataset:
    class_counts = np.bincount(trainset.targets)
    smallest = np.min(class_counts)
    idxs = trainset.targets.argsort()
    tmp = [Subset(trainset, idxs[: int(smallest)])]
    tmp_targets = [trainset.targets[idxs[: int(smallest)]]]
    for count in np.cumsum(class_counts):
        tmp.append(Subset(trainset, idxs[int(count) : int(count + smallest)]))
        tmp_targets.append(trainset.targets[idxs[int(count) : int(count + smallest)]])
    unshuffled = ConcatDataset(tmp)
    unshuffled_targets = torch.cat(tmp_targets)
    shuffled_idxs = torch.randperm(
        len(unshuffled), generator=torch.Generator().manual_seed(seed)
    )
    shuffled = Subset(unshuffled, shuffled_idxs)
    shuffled.targets = unshuffled_targets[shuffled_idxs]

    return shuffled


def _sort_by_class(
    trainset: Dataset,
) -> Dataset:
    class_counts = np.bincount(trainset.targets)
    idxs = trainset.targets.argsort()  # sort targets in ascending order

    tmp = []  # create subset of smallest class
    tmp_targets = []  # same for targets

    start = 0
    for count in np.cumsum(class_counts):
        tmp.append(
            Subset(trainset, idxs[start : int(count + start)])
        )  # add rest of classes
        tmp_targets.append(trainset.targets[idxs[start : int(count + start)]])
        start += count
    sorted_dataset = ConcatDataset(tmp)  # concat dataset
    sorted_dataset.targets = torch.cat(tmp_targets)  # concat targets
    return sorted_dataset


def _power_law_split(
    sorted_trainset: Dataset,
    num_partitions: int,
    num_labels_per_partition: int = 2,
    min_data_per_partition: int = 10,
    mean: float = 0.0,
    sigma: float = 2.0,
) -> Dataset:
    targets = sorted_trainset.targets
    full_idx = range(len(targets))

    class_counts = np.bincount(sorted_trainset.targets)
    labels_cs = np.cumsum(class_counts)
    labels_cs = [0] + labels_cs[:-1].tolist()

    partitions_idx = []
    num_classes = len(np.bincount(targets))
    hist = np.zeros(num_classes, dtype=np.int32)

    # assign min_data_per_partition
    min_data_per_class = int(min_data_per_partition / num_labels_per_partition)
    for u_id in range(num_partitions):
        partitions_idx.append([])
        for cls_idx in range(num_labels_per_partition):
            # label for the u_id-th client
            cls = (u_id + cls_idx) % num_classes
            # record minimum data
            indices = list(
                full_idx[
                    labels_cs[cls]
                    + hist[cls] : labels_cs[cls]
                    + hist[cls]
                    + min_data_per_class
                ]
            )
            partitions_idx[-1].extend(indices)
            hist[cls] += min_data_per_class

    # add remaining images following power-law
    probs = np.random.lognormal(
        mean,
        sigma,
        (num_classes, int(num_partitions / num_classes), num_labels_per_partition),
    )
    remaining_per_class = class_counts - hist
    # obtain how many samples each partition should be assigned for each of the labels it contains
    probs = (
        remaining_per_class.reshape(-1, 1, 1)
        * probs
        / np.sum(probs, (1, 2), keepdims=True)
    )

    for u_id in range(num_partitions):
        for cls_idx in range(num_labels_per_partition):
            cls = (u_id + cls_idx) % num_classes
            count = int(probs[cls, u_id // num_classes, cls_idx])

            # add count of specific class to partition
            indices = full_idx[
                labels_cs[cls] + hist[cls] : labels_cs[cls] + hist[cls] + count
            ]
            partitions_idx[u_id].extend(indices)
            hist[cls] += count

    # construct subsets
    partitions = [Subset(sorted_trainset, p) for p in partitions_idx]
    return partitions
