import itertools
from logging import Logger
from typing import Callable, List, Optional, Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
import torch
import spacy
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.models.cross_attention import CrossAttention

from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor
from diffusers import  AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers

from ptp_utils import (AUGCrossAttnProcessor, AttentionStore, get_indices_to_alter,
show_image_relevance, extract_noun_phrases, match_indices)
import ptp_utils
from PIL import Image
import torch.nn.functional as F
import cv2

class GatherAndBindPipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(self,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 unet: UNet2DConditionModel,
                 scheduler: KarrasDiffusionSchedulers,
                 safety_checker: StableDiffusionSafetyChecker,
                 feature_extractor: CLIPFeatureExtractor,
                 requires_safety_checker: bool = True,
                 ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor,
                         requires_safety_checker)

        self.parser = spacy.load("en_core_web_trf")
    
    def auto_get_indices_to_alter(self, tokenizer: CLIPTokenizer, 
                                  prompt: List[str], parser: spacy.language.Language) -> tuple[list, list]:
        prompt = prompt[0]
        output_sublist = extract_noun_phrases(prompt, parser)
        words_nouns_indices, words_phrase_indices = match_indices(tokenizer, prompt, output_sublist)
        return words_nouns_indices, words_phrase_indices
    
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                Logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return text_inputs, prompt_embeds
    
    def _symmetric_kl(self, flatten1: torch.Tensor, flatten2: torch.Tensor,
                      l2_no_grad: bool = False) -> torch.Tensor:
        """ Symmetric KL divergence. """

        p = torch.distributions.Categorical(probs=flatten1)
        if l2_no_grad:
            with torch.no_grad():
                q = torch.distributions.Categorical(probs=flatten2)
        else:
            q = torch.distributions.Categorical(probs=flatten2)

        kl_divergence_pq = torch.distributions.kl_divergence(p, q)
        kl_divergence_qp = torch.distributions.kl_divergence(q, p)
        avg_kl_divergence = (kl_divergence_pq + kl_divergence_qp) / 2

        return avg_kl_divergence

    def _symmetric_jsd(self, flatten1: torch.Tensor, flatten2: torch.Tensor,
                       l2_no_grad: bool = False) -> torch.Tensor:
        """ Jensen-Shannon divergence. """

        p = torch.distributions.Categorical(probs=flatten1)
        if l2_no_grad:
            with torch.no_grad():
                q = torch.distributions.Categorical(probs=flatten2)
        else:
            q = torch.distributions.Categorical(probs=flatten2)

        M = (flatten1 + flatten2) / 2
        m = torch.distributions.Categorical(probs=M)

        kl_divergence_pm = torch.distributions.kl_divergence(p, m)
        kl_divergence_qm = torch.distributions.kl_divergence(q, m)
        jsd = (kl_divergence_pm + kl_divergence_qm) / 2

        return jsd

    def _compute_kl(self, attention_maps: List[torch.Tensor],
                    loss_mode: str = "kl", l2_no_grad: bool = False) -> torch.Tensor:
        # Convert map into a single distribution: 16x16 -> 256
        for i, attention_map in enumerate(attention_maps):
            if len(attention_map.shape) > 1:
                attention_maps[i] = attention_map.reshape(-1)

        attention_overlaps = []
        last_element = attention_maps[-1]

        for attention_map in attention_maps[:-1]:
            if loss_mode == "jsd":
                avg_kl_divergence = self._symmetric_jsd(attention_map, last_element, l2_no_grad)
            elif loss_mode == "kl":
                avg_kl_divergence = self._symmetric_kl(attention_map, last_element, l2_no_grad)
            elif loss_mode == "None":
                return attention_overlaps
            attention_overlaps.append(avg_kl_divergence)

        return attention_overlaps
    
    
    def _compute_entropies_for_indices(self, attention_maps: torch.Tensor, 
                                           indices_to_alter: List[int],
                                           indices_to_alter_adj: List[int],
                                           loss_mode: str = "jsd",
                                           l2_no_grad: bool = False,) -> List[torch.Tensor]:
        """
        Computes the sum of attention values for the specified indices.
        """
        attention_for_text = attention_maps[:, :, 1:-1]
        attention_for_text *= 100
        sums = attention_for_text.sum(dim=(-2, -3), keepdim=True)
        attention_for_text_normalized = attention_for_text / (sums + 1e-9)  # 1e-9 is added for numerical stability

        # Shift indices since we removed the first token
        indices_to_alter = [index - 1 for index in indices_to_alter]
        
        # Compute sum of attention values for specified tokens
        attention_entropies = []
        for i in indices_to_alter:
            image = attention_for_text_normalized[:, :, i]
             # Compute entropy for each token's attention map
            entropies = -torch.sum(image * torch.log(image + 1e-9), dim=(-1, -2))
            attention_entropies.append(entropies)

        if indices_to_alter_adj is not None:
            attention_overlaps = []
            for indices in indices_to_alter_adj:
                indices_list = [index - 1 for index in indices]
                attention_map_list = [attention_for_text_normalized[:, :, i] for i in indices_list]
                attention_overlaps.append(self._compute_kl(attention_map_list, loss_mode=loss_mode,
                                                           l2_no_grad=l2_no_grad))

            return attention_entropies, attention_overlaps
        else:
            return attention_entropies, None
    
    
    def _aggregate_and_get_entropies_per_token(self, attention_store: AttentionStore,
                                                   prompts: Union[str, List[str]],
                                                   indices_to_alter: List[int],
                                                   indices_to_alter_adj: List[int],
                                                   attention_res: int = 16,
                                                   loss_mode: str = "jsd",
                                                   l2_no_grad: bool = False):
        """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
        attention_maps = self.aggregate_attention(
            prompts=prompts,
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0)
        
        attention_entropies, attention_overlaps = self._compute_entropies_for_indices(
            attention_maps=attention_maps,
            indices_to_alter=indices_to_alter,
            indices_to_alter_adj=indices_to_alter_adj,
            loss_mode=loss_mode,
            l2_no_grad=l2_no_grad)
        
        return attention_entropies, attention_overlaps

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """ Update the latent according to the computed loss. """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond
        return latents
    
    @staticmethod
    def _compute_loss(entropies: List[torch.Tensor], overlaps: List[torch.Tensor],
                      loss_weight: float = 1.0, return_losses: bool = False) -> torch.Tensor:
        """
        Computes the average entropy loss from a list of entropies or a tensor.
        """
        if isinstance(entropies, Tuple):
            entropies = entropies[0]
        losses_entropies = [max(0, curr_max) for curr_max in entropies]
        max_entropies = torch.max(torch.stack(losses_entropies))
        loss_entropies = torch.clamp(max_entropies, min=0)

        loss = loss_entropies
        all_empty = all(len(sublist) == 0 for sublist in overlaps)
        if not all_empty:
            losses_overlaps = [max(0, curr_max[0]) for curr_max in overlaps]
            if len(losses_overlaps) > 0:
                loss_overlaps = torch.max(torch.stack(losses_overlaps))
                loss += loss_weight * loss_overlaps

        if return_losses and not all_empty:
            return loss, losses_entropies, losses_overlaps
        elif return_losses:
            return loss, losses_entropies, None
        else:
            return loss
        

    def _perform_iterative_refinement_step(self,
                                           latents: torch.Tensor,
                                           prompts: Union[str, List[str]],
                                           indices_to_alter: List[int],
                                           indices_to_alter_adj: List[int],
                                           loss: torch.Tensor,
                                           threshold: float,
                                           text_embeddings: torch.Tensor,
                                           text_input,
                                           attention_store: AttentionStore,
                                           step_size: float,
                                           t: int,
                                           loss_weight: float = 1.0,
                                           attention_res: int = 16,
                                           max_refinement_steps: int = 20,
                                           loss_mode: str = "kl",
                                           l2_no_grad: bool = False):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code according to our loss objective until the given threshold is reached for all tokens.
        """
        iteration = 0
        target_loss = max(0, threshold)
        while loss > target_loss:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
            self.unet.zero_grad()

            # Get max activation value for each subject token
            attention_entropies, attention_overlaps = self._aggregate_and_get_entropies_per_token(
                prompts=prompts,
                attention_store=attention_store,
                indices_to_alter=indices_to_alter,
                indices_to_alter_adj=indices_to_alter_adj,
                attention_res=attention_res,
                loss_mode=loss_mode,
                l2_no_grad=l2_no_grad,
                )

            loss, losses_entropies, losses_overlaps = self._compute_loss(attention_entropies, attention_overlaps,
                                              loss_weight, return_losses=True)

            if loss != 0:
                latents = self._update_latent(latents, loss, step_size)

            with torch.no_grad():
                noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0)).sample
                noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample

            low_token_ent = np.argmax([l.item() if type(l) != int else l for l in losses_entropies])
            max_entropy = round(attention_entropies[low_token_ent].item(), 4)
            low_word_ent = self.tokenizer.decode(text_input.input_ids[0][indices_to_alter[low_token_ent]])

            if losses_overlaps is not None:
                values = [l.item() if isinstance(l, torch.Tensor) else l for l in losses_overlaps]
                low_token_over = np.argmax(values)
                max_overlap = round(attention_overlaps[low_token_over][0].item(), 4)
                indices = indices_to_alter_adj[low_token_over]
                words = [text_input.input_ids[0][i] for i in indices if i < len(text_input.input_ids[0])]
                low_word_over = self.tokenizer.decode(words)
            else:
                max_overlap = 0
                low_word_over = None

            print(f'\t Try {iteration}. [{low_word_ent}] has a max entropy of {max_entropy}',
                  f' [{low_word_over}] has a max overlap of {max_overlap}')

            if iteration >= max_refinement_steps:
                print(f'\t Exceeded max number of iterations ({max_refinement_steps})! '
                      f'Finished with a max attention of {loss}')
                break

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
        self.unet.zero_grad()

        # Get max activation value for each subject token
        attention_entropies = self._aggregate_and_get_entropies_per_token(
                prompts=prompts,
                attention_store=attention_store,
                indices_to_alter=indices_to_alter,
                indices_to_alter_adj=indices_to_alter_adj,
                attention_res=attention_res,
                loss_mode=loss_mode,
                l2_no_grad=l2_no_grad,
                )
        loss = self._compute_loss(attention_entropies, attention_overlaps, loss_weight)
        print(f"\t Finished with loss of: {loss}")
        return loss, latents, attention_entropies


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        controller: AttentionStore = None,
        increment_token_indices: List[int] = None,
        attention_res: int = 16,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        scale_factor: int = 20,
        scale_range: Tuple[float, float] = (1., 0.5),
        if_refinement_step: bool = False,
        thresholds: Optional[dict] = {0: 4., 10: 2., 20: 1.},
        max_iter_to_alter: Optional[int] = 25,
        run_standard_sd: bool = False,
        loss_weight: float = 1.0,
        loss_mode: str = "kl",
        if_show_attention_map: bool = False,
        l2_no_grad: bool = False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        self.register_attention_control(controller)  # add attention controller

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_inputs, text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))

        words_nouns_indices, words_phrase_indices = self.auto_get_indices_to_alter(
            tokenizer=self.tokenizer,
            prompt=prompt,
            parser=self.parser)
        
        entropy_results = {}
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                with torch.enable_grad():
                    
                    latents = latents.clone().detach().requires_grad_(True)

                    noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
                    self.unet.zero_grad()
                    
                    if increment_token_indices is not None:
                        indices_to_alter_all = [element for sublist in [
                            words_nouns_indices] + words_phrase_indices for element in sublist]
                        indices_to_alter_all += increment_token_indices
                    else:
                        indices_to_alter_all = [element for sublist in [
                            words_nouns_indices] + words_phrase_indices for element in sublist]
                        
                    if i % 2 == 0 and if_show_attention_map:
                        self.show_cross_attention(attention_store=controller,
                                                prompts=prompt,
                                                res=attention_res,
                                                from_where=("up", "down", "mid"),
                                                indices_to_alter=indices_to_alter_all,
                                                display_image=True,
                                                cmap='viridis',
                                                step=i,
                                                save_path=None)


                    # Get max activation value for each subject token
                    attention_entropies, attention_overlaps = self._aggregate_and_get_entropies_per_token(
                        prompts=prompt,
                        attention_store=controller,
                        indices_to_alter=words_nouns_indices,
                        indices_to_alter_adj=words_phrase_indices,
                        attention_res=attention_res,
                        loss_mode=loss_mode,
                        l2_no_grad=l2_no_grad)
                    
                    # Compute entropies
                    word_ent = [self.tokenizer.decode(text_inputs.input_ids[0][i]) for i in words_nouns_indices]
                    if i % 30 == 0:
                        entropy_results[f'{word_ent[0]}_t{i}'] = round(attention_entropies[0].item(), 4)
                        entropy_results[f'{word_ent[1]}_t{i}'] = round(attention_entropies[1].item(), 4)
                    
                    if not run_standard_sd:
                        # compute the loss
                        loss = self._compute_loss(attention_entropies, attention_overlaps, loss_weight)

                        if i in thresholds.keys() and loss > thresholds[i] and if_refinement_step:
                            del noise_pred_text
                            torch.cuda.empty_cache()
                            loss, latents, attention_entropies = self._perform_iterative_refinement_step(
                                latents=latents,
                                prompts=prompt,
                                indices_to_alter=words_nouns_indices,
                                indices_to_alter_adj=words_phrase_indices,
                                loss=loss,
                                threshold=thresholds[i],
                                loss_weight=loss_weight,
                                text_embeddings=text_embeddings,
                                text_input=text_inputs,
                                attention_store=controller,
                                step_size=scale_factor * np.sqrt(scale_range[i]),
                                t=t,
                                attention_res=attention_res,
                                loss_mode=loss_mode,
                                l2_no_grad=l2_no_grad)

                            # if if_show_attention_map:
                            #     print('----------perform_iterative_refinement_step----------')
                            #     self.show_cross_attention(attention_store=controller,
                            #                     prompts=prompt,
                            #                     res=attention_res,
                            #                     from_where=("up", "down", "mid"),
                            #                     indices_to_alter=indices_to_alter_all,
                            #                     display_image=True,
                            #                     cmap='viridis')
                            
                        # Perform gradient update
                        if i < max_iter_to_alter:
                            loss = self._compute_loss(attention_entropies, attention_overlaps, loss_weight)
                            if loss > 0:
                                latents = self._update_latent(latents=latents, loss=loss,
                                                            step_size=scale_factor * np.sqrt(scale_range[i]))
                            print(f'Iteration {i} | Loss: {loss:0.4f}')

                            
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                    
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * \
                        (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # step callback
                latents = controller.step_callback(latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(
            image, device, text_embeddings.dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), indices_to_alter_all, entropy_results

    def register_attention_control(self, controller):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith(
                "attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[
                    block_id]
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
                place_in_unet = "down"
            else:
                continue
            cross_att_count += 1
            attn_procs[name] = AUGCrossAttnProcessor(
                controller=controller, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        controller.num_att_layers = cross_att_count

    def aggregate_attention(self, prompts, attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
        out = []
        attention_maps = attention_store.get_average_attention()
        num_pixels = res ** 2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(
                        len(prompts), -1, res, res, item.shape[-1])[select]
                    out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out.cpu()

    def show_cross_attention_heatmap(self, prompts, attention_store: AttentionStore, res: int,
                                    indices_to_alter: List[int], from_where: List[str], select: int = 0,
                                    orig_image=None, save_path: str = None):
        tokens = self.tokenizer.encode(prompts[select])
        decoder = self.tokenizer.decode
        attention_maps = self.aggregate_attention(
            prompts, attention_store, res, from_where, True, select).detach().cpu()
        images = []
        for i in range(len(tokens)):
            image = attention_maps[:, :, i]
            if i in indices_to_alter:
                image = show_image_relevance(image, orig_image)
                image = image.astype(np.uint8)
                image = np.array(Image.fromarray(
                    image).resize((res ** 2, res ** 2)))
                image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
                images.append(image)
        ptp_utils.view_images(np.stack(images, axis=0), save_path=save_path)

    def show_cross_attention(self, prompts, attention_store: AttentionStore,
                             indices_to_alter: List[int], res: int, from_where: List[str],
                             display_image: bool = True,  save_path: str = None, 
                             cmap: str = 'viridis', select: int = 0, step: int=0):
        tokens = self.tokenizer.encode(prompts[select])
        decoder = self.tokenizer.decode
        attention_maps = self.aggregate_attention(
            prompts, attention_store, res, from_where, True, select).detach().cpu()
        images = []
        for i in range(len(tokens)):
            image = attention_maps[:, :, i]
            if i in indices_to_alter:
                image = 255 * image / image.max()
                image = image.unsqueeze(-1).expand(*image.shape, 3)
                image = image.numpy().astype(np.uint8)
                image = np.array(Image.fromarray(image).resize((res ** 2, res ** 2)))
                image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
                images.append(image)
                # image_ = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                # plt.imsave(f"{save_path}/{i}_{step}.png", image_, cmap=cmap)
        ptp_utils.view_images(np.stack(images, axis=0), display_image=display_image,
                              cmap=cmap, save_path=save_path)


    def show_self_attention_comp(self, prompts, attention_store: AttentionStore, res: int, from_where: List[str],
                                 max_com=10, select: int = 0):
        attention_maps = self.aggregate_attention(
            prompts, attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
        u, s, vh = np.linalg.svd(
            attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
        images = []
        for i in range(max_com):
            image = vh[i].reshape(res, res)
            image = image - image.min()
            image = 255 * image / image.max()
            image = np.repeat(np.expand_dims(image, axis=2),
                              3, axis=2).astype(np.uint8)
            image = Image.fromarray(image).resize((256, 256))
            image = np.array(image)
            images.append(image)
        ptp_utils.view_images(np.concatenate(images, axis=1))
