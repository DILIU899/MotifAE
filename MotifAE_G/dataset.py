import copy
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from utils.configs import get_logger

logger = get_logger("SAE-Dataloader")


class ProteinData(Dataset):
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        cache_embeddings: bool = False,
        precomputed_embedding_path: Optional[str] = None,
        chunk_seq: bool = False,
        label_name: list = ["score"]
    ):
        """
        Init ProteinDataSet

        Args:
            metadata_df: protein mutation metadata
            cache_embeddings: Whether to cache all the available esm_embeddings in the RAM
            precomputed_embedding_path: Where to load the precomputed embedding
            chunk_seq: Whether to chunk the seq embedding to fit to the domain data
        """
        self.metadata_df = metadata_df
        self.cache_embeddings = cache_embeddings
        self.precomputed_embedding_path = precomputed_embedding_path
        self.chunk_seq = chunk_seq
        self.label_name = label_name

        # Group the result based on the domianIDs
        self.domain_groups = self.metadata_df.groupby("domainID")
        self.domain_ids = list(self.domain_groups.groups.keys())

        # get domain -> sample, used for sampling batch to enable one batch one domain
        self.domain_to_samples = {}

        for domain_id in self.domain_ids:
            group_indices = self.domain_groups.get_group(domain_id).index.tolist()
            self.domain_to_samples[domain_id] = group_indices

        # cache embeddings
        self.embedding_cache = {} if cache_embeddings else None

    def _get_esm2_embedding(self, uniprot_id: str, domain_id: str, seq_start: int, seq_end: int) -> torch.Tensor:
        """get esm2 embedding and chunk. We use index start from 1 to chunk."""
        if self.cache_embeddings and domain_id in self.embedding_cache:
            embedding = self.embedding_cache[domain_id]
        else:
            if not self.precomputed_embedding_path is None:
                embedding = torch.tensor(
                    np.load(f"{self.precomputed_embedding_path}/{uniprot_id}.representations.layer.48.npy")
                )
                # rule out the <cls> and <eos> token and chunk, we use index start from 1
                if self.chunk_seq:
                    embedding = embedding[1:-1, :][(seq_start - 1) : seq_end, :]
                else:
                    embedding = embedding[1:-1, :]
            else:
                raise NotImplementedError(
                    "Should add use seq to inference esm embedding option. But not implemented"
                )
            if self.cache_embeddings:
                self.embedding_cache[domain_id] = embedding

        return embedding

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, idx: str) -> Dict[str, torch.Tensor]:
        """get single sample, here we receive the index from our self-defined sampler, which is the index of the dataframe"""
        row = self.metadata_df.loc[idx]

        # get embedding
        embedding = self._get_esm2_embedding(row["uniprotID"], row["domainID"], row["seq.start"], row["seq.end"])

        # embedding: [L, 1280]
        # score: [1]
        # pos: [1]
        # alt: [1]
        # sample_idx: [1]
        return {
            "embedding": embedding,
            "score": row[self.label_name].values.astype(float),
            "pos": row["pos"] - 1,
            "wt": row["ref"],
            "alt": row["alt"],
            "sample_idx": row.name,
        }


class ProteinBatchSampler:
    def __init__(
        self,
        dataset: ProteinData,
        batch_size: int,
        domain_diversity: float = 0.0,  # 0: single domain, 1: try to use batch_size domains
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Args:
            dataset: ProteinData instance
            batch_size: batch size
            domain_diversity: float between 0-1, controls number of different domains in each batch
                          0.0: try to use single domain per batch
                          1.0: try to use batch_size different domains per batch
                          -1: strict one batch one domain
            shuffle: whether to shuffle samples, if not shuffle, only used in valid/test mode, we don't care about domain_diversity
            drop_last: whether to drop the last batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.domain_diversity = domain_diversity
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):

        if self.shuffle:
            batches = []
            available_samples = copy.deepcopy(self.dataset.domain_to_samples)

            # while we still have samples
            while available_samples:
                batch = []

                if self.domain_diversity != -1:
                    # how many domains we need for each batch
                    n_domain_per_batch = max(
                        1, min(len(available_samples), int(np.ceil(self.batch_size * self.domain_diversity)))
                    )
                else:
                    n_domain_per_batch = 1
                available_domains = list(available_samples.keys())

                selected_domains = list(np.random.choice(available_domains, n_domain_per_batch, replace=False))
                if self.domain_diversity != -1:
                    # we need to make sure for domains we want to have, we should at least have enough sample size
                    domain_total_counts = sum([len(available_samples[i]) for i in selected_domains])
                    rest_domains = list(set(available_domains) - set(selected_domains))
                    while domain_total_counts < self.batch_size and rest_domains:
                        # if we still have new domains, sample a new domain from the rest domains
                        new_domain = np.random.choice(rest_domains)
                        domain_total_counts += len(available_samples[new_domain])
                        selected_domains.append(new_domain)
                        rest_domains.remove(new_domain)

                # First ensure at least one sample from each selected domain
                for domainID in selected_domains:
                    sample_id = np.random.randint(len(available_samples[domainID]))
                    batch.append(available_samples[domainID].pop(sample_id))

                # Fill the rest of the batch randomly from selected domains
                while len(batch) < self.batch_size and any(available_samples[d] for d in selected_domains):
                    # Filter domains that still have samples
                    valid_domains = [d for d in selected_domains if available_samples[d]]
                    if not valid_domains:
                        break

                    domain = np.random.choice(valid_domains)
                    sample_id = np.random.randint(len(available_samples[domain]))
                    batch.append(available_samples[domain].pop(sample_id))

                # Clean up empty domains
                available_samples = {k: v for k, v in available_samples.items() if v}

                np.random.shuffle(batch)
                if len(batch) == self.batch_size or (not self.drop_last and batch):
                    batches.append(batch)

            # shuffle batches
            np.random.shuffle(batches)

        else:
            batches = [
                self.dataset.metadata_df.index[i : i + self.batch_size].to_list()
                for i in range(0, len(self.dataset), self.batch_size)
            ]
            if self.drop_last and batches and len(batches[-1]) < self.batch_size:
                batches.pop()

        for batch in batches:
            yield batch

    def __len__(self):
        total_samples = len(self.dataset)
        if self.drop_last:
            if self.domain_diversity == -1:
                return (self.dataset.metadata_df.value_counts("domainID") // self.batch_size).sum()
            else:
                return total_samples // self.batch_size
        else:
            if self.domain_diversity == -1:
                return (
                    (self.dataset.metadata_df.value_counts("domainID") + self.batch_size - 1) // self.batch_size
                ).sum()
            else:
                return (total_samples + self.batch_size - 1) // self.batch_size


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    change list of single item dict to Tensor
    """
    # get the max len for this batch
    max_len = max(x["embedding"].shape[0] for x in batch)

    # collect batch
    embeddings = []
    scores = []
    positions = []
    wts = []
    alts = []
    sample_idxs = []

    for sample in batch:
        # Padding embedding to max length
        # x shape: [batch, max_L, 1280]
        emb = sample["embedding"]
        pad_len = max_len - emb.shape[0]
        if pad_len > 0:
            emb = torch.nn.functional.pad(emb, (0, 0, 0, pad_len))
        embeddings.append(emb)

        scores.append(sample["score"])
        positions.append(sample["pos"])
        wts.append(sample["wt"])
        alts.append(sample["alt"])
        sample_idxs.append(sample["sample_idx"])

    return (
        torch.stack(embeddings),
        torch.tensor(np.array(scores), dtype=torch.float32),
        positions,
        wts,
        alts,
        sample_idxs,
    )


def collate_fn_embedding_level(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    change list of single item dict to Tensor
    """
    # collect batch
    embeddings = []
    scores = []
    positions = []
    wts = []
    alts = []
    sample_idxs = []

    for sample in batch:
        # Now we return x:[batch, 1280], as we directly take the embedding from the corresponding position
        emb = sample["embedding"][sample["pos"]]
        embeddings.append(emb)

        scores.append(sample["score"])
        positions.append(sample["pos"])
        wts.append(sample["wt"])
        alts.append(sample["alt"])
        sample_idxs.append(sample["sample_idx"])

    return (
        torch.stack(embeddings),
        torch.tensor(np.array(scores), dtype=torch.float32),
        positions,
        wts,
        alts,
        sample_idxs,
    )


class DataPipeline:
    def __init__(self, config):

        self.cache_embeddings = config.get("cache_embedding", False)
        self.precomputed_embedding_path = config.get("embedding_path", None)
        self.chunk_seq = config.get("chunk_seq", False)
        self.label_name = config.get("label", ["score"])
        # we don't allow no batch_size, domain_diversity provided
        self.batch_size = config["batch_size"]
        self.domain_diversity = config["domain_diversity"]

    def get_dataloader(self, metadata, mode="train"):
        if isinstance(metadata, str):
            metadata = pd.read_csv(metadata, index_col=0)
        elif not isinstance(metadata, pd.DataFrame):
            raise RuntimeError("Need to input metadata dataframe or the path to that.")

        start = time.time()

        dataset = ProteinData(
            metadata_df=metadata,
            cache_embeddings=self.cache_embeddings,
            precomputed_embedding_path=self.precomputed_embedding_path,
            chunk_seq=self.chunk_seq,
            label_name=self.label_name
        )

        if mode == "train" and self.domain_diversity != -1:
            shuffle = True
            drop_last = True
        elif mode == "train" and self.domain_diversity == -1:
            shuffle = True
            drop_last = False
        elif mode == "valid":
            shuffle = True
            drop_last = False
        elif mode == "test":
            shuffle = False
            drop_last = False

        sampler = ProteinBatchSampler(
            dataset=dataset,
            batch_size=self.batch_size,
            domain_diversity=self.domain_diversity,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        # dataloader = DataLoader(dataset=dataset, batch_sampler=sampler, collate_fn=collate_fn)
        dataloader = DataLoader(dataset=dataset, batch_sampler=sampler, collate_fn=collate_fn_embedding_level)

        end = time.time()

        logger.info(f"{mode} dataloader constructed in {end-start:.2f} sec")

        return dataloader
