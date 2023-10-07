import abc
import torch
import cv2
import re
import pprint
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display
from typing import Dict, Union, Tuple, List
import warnings
import spacy

from diffusers.models.cross_attention import CrossAttention
from transformers import CLIPTokenizer


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02, display_image=True,
                cmap=None, save_path=None):
    """
    Displays a list of images in a grid using Matplotlib with optional color mapping.
    """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape[:2] + (3,), dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255

    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]
            
    if cmap is not None:
        image_ = cv2.cvtColor(image_, cv2.COLOR_RGB2GRAY)
    if display_image:
        plt.imshow(image_, cmap=cmap)
        plt.axis('off')
        plt.show()
    if save_path is not None:
        plt.imsave(save_path, image_, cmap=cmap)


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][i].detach()
        self.step_store = self.get_empty_store()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}

    def __init__(self, save_global_store=False):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super(AttentionStore, self).__init__()
        self.save_global_store = save_global_store
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}
        self.curr_step_index = 0


class AUGCrossAttnProcessor:

    def __init__(self, controller, place_in_unet):
        super().__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # one line change
        self.controller(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        # hidden_states = xformers.ops.memory_efficient_attention(
        #     query, key, value, attn_bias=attention_mask, op=self.attention_op
        # )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 <= idx < len(stable.tokenizer(prompt)['input_ids'])}
    pprint.pprint(token_idx_to_word)
    token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                          "alter (e.g., 2,5): ")
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices

def get_ranges_to_alter(stable, prompt: str) -> List[Tuple[int, int]]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 <= idx < len(stable.tokenizer(prompt)['input_ids'])}
    pprint.pprint(token_idx_to_word)

    token_ranges = input(
        "Please enter token ranges to alter as '(start1, end1), (start2, end2)': ")

    # 使用正则表达式提取括号内的数字
    pattern = r'\((\d+),\s*(\d+)\)'
    matches = re.findall(pattern, token_ranges)

    selected_ranges = []
    for match in matches:
        start, end = map(int, match)
        selected_ranges.append((start, end))

    selected_tokens = [token_idx_to_word[i] for start,
                       end in selected_ranges for i in range(start, end + 1)]
    print(f"Altering tokens: {selected_tokens}")

    return selected_ranges


def show_image_relevance(image_relevance, image: Image.Image, relevnace_res=16):
    # create heatmap from mask on image

    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(
        1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    # because float16 precision interpolation is not supported on cpu
    image_relevance = image_relevance.cuda()
    image_relevance = torch.nn.functional.interpolate(
        image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu()  # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / \
        (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(
        relevnace_res ** 2, relevnace_res ** 2)
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

def extract_noun_phrases(prompt: str, parser: spacy.language.Language) -> tuple[list, list]:
    doc = parser(prompt)
    subtrees = []
    modifiers = ["amod", "nmod", "compound", "npadvmod", "advmod", "acomp"]
    nouns = []

    for w in doc:
        if w.pos_ not in ["NOUN", "PROPN"] or w.dep_ in modifiers:
            continue
        subtree = []
        stack = []
        for child in w.children:
            if child.dep_ in modifiers:
                subtree.append(child)
                stack.extend(child.children)

        while stack:
            node = stack.pop()
            if node.dep_ in modifiers or node.dep_ == "conj":
                subtree.append(node)
                stack.extend(node.children)
        if subtree:
            subtree.append(w)
            subtrees.append(subtree)

        if w.pos_ in ["NOUN", "PROPN"]:
            nouns.append(w)

    return nouns, subtrees


def get_indices(tokenizer: CLIPTokenizer, prompt: str) -> Dict[str, int]:
    ids = tokenizer(prompt).input_ids
    token_idx_to_word = {idx: tokenizer.decode(t)
                         for idx, t in enumerate(ids)
                         if 0 <= idx < len(tokenizer(prompt)['input_ids'])}
    return token_idx_to_word


def get_word_nouns_indices(output_sublist: tuple[list, list], token_idx_to_word: Dict[str, int]) -> list:
    words_phrase_indices = []
    words_phrase = [word for word in output_sublist]
    words_phrase = [token.text if isinstance(
        token, spacy.tokens.token.Token) else token for token in words_phrase]

    for k, v in token_idx_to_word.items():
        if v in words_phrase:
            words_phrase_indices.append(k)

    return words_phrase_indices


def get_word_phrase_indices(output_sublist: tuple[list, list], token_idx_to_word: Dict[str, int]) -> list:
    words_phrase_indices = []
    words_phrase = [word for word in output_sublist]
    words_phrase = [token.text if isinstance(
        token, spacy.tokens.token.Token) else token for token in words_phrase]

    sublist_len = len(words_phrase)
    keys = list(token_idx_to_word.keys())
    values = list(token_idx_to_word.values())
    for i in range(len(values) - sublist_len + 1):
        if values[i:i+sublist_len] == words_phrase:
            return keys[i:i+sublist_len]
    return None


def match_indices(tokenizer: CLIPTokenizer, prompt: str, paired_indices: tuple[list, list]) -> tuple[list, list]:
    token_idx_to_word = get_indices(tokenizer, prompt)
    prompt_s = prompt.split()
    if len(prompt_s) != len(token_idx_to_word) - 2:
        warnings.warn("Please change the prompt to match the tokenizer", Warning)
        words_nouns_indices = get_word_nouns_indices(paired_indices[0], token_idx_to_word)
        return words_nouns_indices, []

    words_nouns_indices = get_word_nouns_indices(paired_indices[0], token_idx_to_word)

    words_phrase_indices = []
    for sublist in paired_indices[1]:
        word_indices = get_word_phrase_indices(sublist, token_idx_to_word)
        words_phrase_indices.append(word_indices)

    return words_nouns_indices, words_phrase_indices