import json
import inspect
import warnings
import functools
from contextlib import contextmanager
from typing import Optional, Union, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.attention.flex_attention import BlockMask, create_block_mask as create_block_causal_mask_flex

from transformers import Qwen2Model
from transformers.utils.generic import ExplicitEnum
from transformers.cache_utils import StaticCache, SlidingWindowCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.generation.streamers import BaseStreamer
from transformers.generation.logits_process import TopPLogitsWarper, TemperatureLogitsWarper
from transformers import (
    Qwen2ForCausalLM,
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerationConfig,
    PreTrainedModel
)
from transformers.generation.utils import (
    GenerateDecoderOnlyOutput,
    GenerateNonBeamOutput,
    GenerateOutput
)

def debug_mask(mask, head_idx=0):
    if mask.ndim == 4:
        if head_idx >= mask.shape[1]:
            raise RuntimeError(f'invalid head_idx {head_idx}')
        masks_to_print = mask[:, head_idx, :, :]
        print(f'--- head {head_idx} ---')
    else:
        raise RuntimeError(f'expected 3 or 4 dimensions, not {mask.ndim}')
    for i, mask in enumerate((masks_to_print == 0.).int().cpu()):
        print(f'\nbatch {i}:')
        for row in mask:
            print(' '.join(map(str, row.tolist())))
    print("--------------------")

def debug(d):
    return json.dumps({k: str(v) for k, v in d.items()}, indent=2)

def sample_tokens(probs, generator=None):
    next_token = torch.multinomial(probs, num_samples=1, generator=generator)
    return next_token

def sample_tokens_parallel(probs, generator=None):
    next_token = sample_tokens(probs.view(-1, probs.shape[-1]), generator)
    return next_token.view(probs.shape[0], probs.shape[1])

def parse_interleaved(tokenizer, inputs, output_ids, choreography_k):
    return [
        tokenizer.batch_decode(
            row[inputs['input_ids'].shape[1]:].reshape(-1, choreography_k).T,
            skip_special_tokens=True
        )
        for row in output_ids
    ]

class GenerationMode(ExplicitEnum):
    CONTRASTIVE_SEARCH = "contrastive_search"
    GREEDY_SEARCH = "greedy_search"
    SAMPLE = "sample"
    CHOREOGRAPHED_SAMPLE = "choreographed_sample"
    ASSISTED_GENERATION = "assisted_generation"
    DOLA_GENERATION = "dola_generation"
    BEAM_SEARCH = "beam_search"
    BEAM_SAMPLE = "beam_sample"
    CONSTRAINED_BEAM_SEARCH = "constrained_beam_search"
    GROUP_BEAM_SEARCH = "group_beam_search"

def make_flex_block_causal_mask(
    attention_mask_2d: torch.Tensor,
    attention_chunk_size: Optional[int] = None,
    query_length=None,
    key_length=None,
    offsets=None,
) -> BlockMask:
    batch_size, total_seq_len = attention_mask_2d.shape
    if not key_length:
        key_length = total_seq_len
    if not query_length:
        query_length = total_seq_len
    attention_mask_2d = torch.nn.functional.pad(attention_mask_2d, value=0, pad=(0, key_length))
    device = attention_mask_2d.device
    document_ids = attention_mask_2d.clone()

    if attention_chunk_size is not None:
        document_ids = (document_ids.fill_(1).cumsum(-1) - 1) // (attention_chunk_size)

    def causal_mask_mod(batch_idx, head_idx, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = document_ids[batch_idx, q_idx] == document_ids[batch_idx, kv_idx]
        padding_mask = attention_mask_2d[batch_idx, q_idx] > 0
        final_mask = causal_mask & padding_mask & document_mask
        return final_mask

    if offsets is not None:
        q_offset = offsets[0]
        kv_offset = offsets[1]

        def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
            offset_q = q_idx + q_offset
            offset_kv = kv_idx + kv_offset
            return causal_mask_mod(batch_idx, head_idx, offset_q, offset_kv)
    else:
        mask_mod = causal_mask_mod

    return create_block_causal_mask_flex(
        mask_mod=mask_mod,
        B=batch_size,
        H=None, # attention head
        Q_LEN=query_length,
        KV_LEN=key_length,
        device=device,
    )

class ChoreographedModel(Qwen2Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _update_causal_mask(
        self,
        attention_mask,
        input_tensor,
        cache_position,
        past_key_values,
        output_attentions=False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
            device=self.device,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

@contextmanager
def attn_impl_context(obj, impl: str):
    originals = [
        layer.self_attn.config._attn_implementation
        for layer in obj.model.layers
    ]
    for layer in obj.model.layers:
        layer.self_attn.config._attn_implementation = impl
    try:
        yield
    finally:
        for layer, orig in zip(obj.model.layers, originals):
            layer.self_attn.config._attn_implementation = orig

def force_attn_impl(impl: str):
    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            with attn_impl_context(self, impl):
                return method(self, *args, **kwargs)
        return wrapper
    return decorator

class ChoreographedCausalLM(Qwen2ForCausalLM):

    @classmethod
    def from_pretrained(cls, *args, choreography_k=1, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        model.choreography_k = choreography_k
        return model

    def __init__(self, config):
        super().__init__(config)
        self.model = ChoreographedModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _choreographed_sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        k: int = 1,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        device = self.device
        eos_token_id = generation_config._eos_token_tensor
        pad_token_id = generation_config._pad_token_tensor

        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        logits_processor = LogitsProcessorList([
            l for l in logits_processor
            if isinstance(l, (TopPLogitsWarper, TemperatureLogitsWarper))
        ])

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        batch_size, prompt_len = input_ids.shape
        this_peer_finished = False
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
        unfinished_sequences = torch.ones(batch_size, k, dtype=torch.bool, device=input_ids.device)

        dtype = self.dtype
        min_dtype = torch.finfo(dtype).min
        base_mask = torch.full((batch_size, prompt_len), fill_value=0, dtype=dtype, device=device)
        base_mask[input_ids == pad_token_id] = min_dtype
        base_mask = base_mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, k, -1) # (batch_size, prompt_len) -> (batch_size, 1, k, prompt_len)
        position_ids = torch.cumsum(input_ids != pad_token_id, dim=1)[:, -1:].expand(-1, k)

        diag = torch.full((k, k), fill_value=min_dtype, dtype=dtype, device=device)
        diag.fill_diagonal_(0)
        diag = diag.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        # initial attention mask just for prefilling
        model_kwargs['attention_mask'] = torch.where(input_ids == pad_token_id, 0, 1)

        cur_len = prompt_len
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            # TODO -- not handling compilation
            outputs = self(**model_inputs, return_dict=True)

            model_kwargs['past_key_values'] = outputs['past_key_values']
            model_kwargs['cache_position'] = torch.arange(model_kwargs['cache_position'][-1] + 1, model_kwargs['cache_position'][-1] + k + 1, dtype=torch.long, device=device)
            model_kwargs['position_ids'] = position_ids if is_prefill else model_kwargs['position_ids'] + 1
            model_kwargs['attention_mask'] = torch.cat((base_mask, diag.repeat(1, 1, 1, cur_len - prompt_len + 1)), dim=-1)

            if synced_gpus and this_peer_finished:
                continue

            if is_prefill:
                next_token_logits = outputs.logits[:, -1:, :].to(copy=True, dtype=torch.float32, device=input_ids.device).expand(batch_size, k, -1)
                next_token_scores = logits_processor(input_ids, next_token_logits)
                is_prefill = False
            else:
                next_token_logits = outputs.logits[:, -k:, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
                next_token_scores = logits_processor(input_ids, next_token_logits)

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            if do_sample:
                probs = F.softmax(next_token_scores, dim=-1)
                next_tokens = sample_tokens_parallel(probs)
            else:
                raise NotImplementedError()

            next_tokens[~unfinished_sequences] = pad_token_id

            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            if cur_len == stopping_criteria.max_length:
                this_peer_finished = True
            else:
                unfinished_sequences = unfinished_sequences & ~(torch.isin(next_tokens, eos_token_id))
                this_peer_finished = unfinished_sequences.max() == 0

            cur_len += 1
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return input_ids

    @torch.no_grad()
    @force_attn_impl('sdpa') # lol!
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)  # only used for assisted generation

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )
        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            synced_gpus = (is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)) and dist.get_world_size() > 1

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                warnings.warn(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            generation_config.use_cache = True

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )
        elif kwargs_has_attention_mask:
            # TODO (joao): generalize this check with other types of inputs
            if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        # If the model supports `logits_to_keep` in forward(), set it to 1 to avoid computing the whole
        # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
        # dynamically overrides this value as it can need more than the last token logits
        if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
            model_kwargs["logits_to_keep"] = 1

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. Prepare the cache.
        # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
        # - different models have a different cache name expected by the model (default = "past_key_values")
        # - `max_length`, prepared above, is used to determine the maximum cache length
        max_cache_length = generation_config.max_length - 1
        if (
            inputs_tensor.shape[1] != input_ids_length
            and model_input_name == "inputs_embeds"
            and not self.config.is_encoder_decoder
        ):
            max_cache_length += inputs_tensor.shape[1]
        self._prepare_cache_for_generation(
            generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
        )

        # 8. determine generation mode
        generation_mode = generation_config.get_generation_mode(assistant_model)

        # choreography!
        if self.choreography_k and self.choreography_k > 1:
            generation_mode = GenerationMode.CHOREOGRAPHED_SAMPLE

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 9. prepare logits processors and stopping criteria
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )

        # Set model_kwargs `use_cache` so we can use it later in forward runs
        model_kwargs["use_cache"] = generation_config.use_cache

        # 10. go into different generation modes
        if generation_mode == GenerationMode.ASSISTED_GENERATION:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing assisted generate, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if batch_size > 1:
                raise ValueError("assisted generate is only supported for batch_size = 1")
            if not model_kwargs["use_cache"]:
                raise ValueError("assisted generate requires `use_cache=True`")
            if generation_config.cache_implementation in ["static", "hybrid", "sliding_window"]:
                raise ValueError("assisted generate is not supported with Static cache classes`")
            if self._is_stateful:
                # In assisted generation we need the ability to confirm whether the model would pick certain tokens,
                # which is not possible with stateful models (they can't reset to a previous subset of generated text)
                raise ValueError(
                    f"assisted generation is not supported with stateful models, such as {self.__class__.__name__}"
                )

            # 11. Get the candidate generator, given the parameterization
            candidate_generator = self._get_candidate_generator(
                generation_config=generation_config,
                input_ids=input_ids,
                inputs_tensor=inputs_tensor,
                assistant_model=assistant_model,
                logits_processor=logits_processor,
                target_tokenizer=tokenizer,
                assistant_tokenizer=assistant_tokenizer,
                model_kwargs=model_kwargs,
            )

            # 12. run assisted generate
            result = self._assisted_decoding(
                input_ids,
                candidate_generator=candidate_generator,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        elif generation_mode == GenerationMode.DOLA_GENERATION:
            if self._is_stateful:
                # DoLa decoding was not designed for stateful models, and would require some changes
                raise ValueError(
                    f"dola decoding is not supported with stateful models, such as {self.__class__.__name__}"
                )
            result = self._dola_decoding(
                input_ids,
                dola_layers=generation_config.dola_layers,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
            if not model_kwargs["use_cache"]:
                raise ValueError("Contrastive search requires `use_cache=True`")
            if self._is_stateful:
                # Just like assisted generation, we need to be able to rollback to a previous state (see comment above)
                raise ValueError(
                    f"contrastive search is not supported with stateful models, such as {self.__class__.__name__}"
                )

            result = self._contrastive_search(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # 11. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
            result = self._sample(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode in (GenerationMode.CHOREOGRAPHED_SAMPLE):
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            model_kwargs['logits_to_keep'] = self.choreography_k # important!

            result = self._choreographed_sample(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                k=self.choreography_k,
                **model_kwargs,
            )

        elif generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH):
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 12. run beam sample
            result = self._beam_search(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                num_beam_groups=generation_config.num_beam_groups,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self._group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints

            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
                        f"of positive integers, but is {generation_config.force_words_ids}."
                    )

                if (
                    not isinstance(generation_config.force_words_ids, list)
                    or len(generation_config.force_words_ids) == 0
                ):
                    typeerror()

                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                            for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 11. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self._constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        # Convert to legacy cache format if requested
        if (
            generation_config.return_legacy_cache is True
            and hasattr(result, "past_key_values")
            and getattr(result.past_key_values, "to_legacy_cache") is not None
        ):
            result.past_key_values = result.past_key_values.to_legacy_cache()
        return result
