import torch as tt
import os

from transformers import BertTokenizer, BertModel
from typing import List, Union, Tuple
from math import ceil
from tqdm import tqdm


class BertEmbedder:
    def __init__(self, model_name: str):
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.batch_size = os.getenv("BATCH_SIZE", default=512) # 64

    def _process_data(self, data: Union[str, List[str]]) -> tt.Tensor:
        tokenized = self.tokenizer(data, return_tensors="pt", padding=True)
        with tt.no_grad():
            output = self.model(**tokenized)
        h = output.last_hidden_state
        return h, tokenized

    def embed_sentence(self, sent: str) -> List[float]:
        h, _ = self._process_data(sent)
        h_mean = h[0].mean(axis=0).numpy().tolist()
        return h_mean

    def batch_embed_sentences(
        self,
        sents: List[str],
        verbose=False
    ) -> List[List[float]]:
        output = []
        n_batches = ceil(len(sents)/self.batch_size)

        if verbose:
            print("Embedding whole sentences")
            batch_ids = tqdm(
                range(n_batches),
                total=n_batches
            )
        else:
            batch_ids = range(n_batches)

        for i in batch_ids:
            batch = sents[i*self.batch_size:(i+1)*self.batch_size]
            h, _ = self._process_data(batch)
            h_mean = h.mean(axis=1).numpy().tolist()
            output += h_mean
        return output

    def embed_mask_token(self, sent: str) -> List[float]:
        h, tokenized = self._process_data(sent)
        h, tokenized = h[0], tokenized["input_ids"][0]
        mask_index = tokenized.numpy().tolist().index(
            self.tokenizer.mask_token_id
        )
        mask_embedding = h[mask_index].numpy().tolist()
        return mask_embedding

    def batch_embed_mask_tokens(
        self,
        sents: List[List[str]],
        verbose=False
    ) -> List[List[float]]:
        output = []
        n_batches = ceil(len(sents)/self.batch_size)

        if verbose:
            print("Embedding mask tokens only")
            batch_ids = tqdm(
                range(n_batches),
                total=n_batches
            )
        else:
            batch_ids = range(n_batches)

        for i in batch_ids:
            batch = sents[i*self.batch_size:(i+1)*self.batch_size]
            h, tokenized = self._process_data(batch)
            tokenized = tokenized["input_ids"]
            mask_index = tt.nonzero(
                tokenized == self.tokenizer.mask_token_id
            )
            mask_embeddings = h[
                mask_index[:, 0], mask_index[:, 1]
            ].numpy().tolist()
            output += mask_embeddings
        return output

    def embed_tokens(self, sent: str) -> List[Tuple[str, float]]:
        h, tokenized = self._process_data(sent)
        h, tokenized = h[0], tokenized["input_ids"][0]
        tokens = self.tokenizer.convert_ids_to_tokens(tokenized)

        output = []

        for token, vector in zip(tokens, h):
            output.append((token, vector.numpy().tolist()))

        return output
