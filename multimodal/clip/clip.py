import os
import torch
from torch import Tensor
from torchvision import transforms as T
from typing import List

from utils import download, load_jit_model
from model import CLIP, _convert_to_fp16
from tokenizer import Tokenizer


_MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
}

_tokenizer = Tokenizer()


def available_models() -> List[str]:
    return list(_MODELS.keys())


def _transform(size):
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])


def load(name: str, jit: bool = False, model_path: str = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert name in _MODELS, f"Model {name} not found; available models = {available_models()}"
    if model_path is None:
        model_path = download(_MODELS[name], os.path.expanduser("~/.cache/clip"))

    model = torch.jit.load(model_path, map_location='cpu').eval()

    if not jit:
        state_dict = model.state_dict()
        model = CLIP(224, int(name.rsplit('/')[-1]))
        model.apply(_convert_to_fp16)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model = model.to(device)
        del state_dict

        if str(device) == 'cpu': 
            model.float()
        return model, _transform(model.visual.input_resolution)

    model = load_jit_model(model, device)
    return model, _transform(model.input_resolution.item())


def tokenize(texts, context_length=77, truncate=False) -> Tensor:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
    

if __name__ == '__main__':
    model_path = 'C:\\Users\\sithu\\Documents\\Projects\\multimodal\\checkpoints\\ViT-B-32.pt'
    model, transforms = load('ViT-B/32', True, model_path)
    # tokens = tokenize('a photo of a dog')
    # print(tokens.shape)
