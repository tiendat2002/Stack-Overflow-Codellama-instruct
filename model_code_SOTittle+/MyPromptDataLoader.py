from torch.utils.data.sampler import RandomSampler
from torch.utils.data import Dataset
from typing import *
from openprompt.data_utils import InputExample, InputFeatures
from tqdm.std import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt.plms.utils import TokenizerWrapper
from openprompt.prompt_base import Template, Verbalizer
from openprompt.utils import signature
from torch.utils.data import DataLoader


class MyPromptDataLoader(object):
    r"""
    PromptDataLoader wraps the original dataset. The input data is firstly wrapped with the
    prompt's template, and then is tokenized by a wrapperd-tokenizer.

    Args:
        dataset (:obj:`Dataset` or :obj:`List`): Either a DatasetObject or a list containing the input examples.
        template (:obj:`Template`): A derived class of :obj:`Template`
        tokenizer (:obj:`PretrainedTokenizer`): The pretrained tokenizer.
        tokenizer_wrapper_class (:cls:`TokenizerWrapper`): The class of tokenizer wrapper.
        max_seq_length (:obj:`int`, optional): The max sequence length of the input ids. It's used to truncate sentences.
        batch_size (:obj:`int`, optional): The batch_size of data loader
        teacher_forcing (:obj:`bool`, optional): Whether to fill the mask with target text. Set to true in training generation model.
        decoder_max_length (:obj:`int`, optional): the decoder maximum length of an encoder-decoder model.
        predict_eos_token (:obj:`bool`, optional): Whether to predict the <eos> token. Suggest to set to true in generation.
        truncate_method (:obj:`bool`, optional): the truncate method to use. select from `head`, `tail`, `balanced`.
        kwargs  :Other kwargs that might be passed into a tokenizer wrapper.
    """

    def __init__(self,
                 tensor_dataset,
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = False,
                 teacher_forcing: Optional[bool] = False,
                 drop_last: Optional[bool] = False,
                 ):

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.teacher_forcing = teacher_forcing

        if self.shuffle:
            sampler = RandomSampler(tensor_dataset)
        else:
            sampler = None

        self.dataloader = DataLoader(
            tensor_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=InputFeatures.collate_fct,
            drop_last=drop_last,
        )

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self, ):
        return self.dataloader.__iter__()