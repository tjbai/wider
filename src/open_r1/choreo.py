import os
import json
from typing import Optional, Union

import torch
import torch.nn.functional as F
from transformers import (
    Qwen2ForCausalLM,
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerationConfig,
    DynamicCache,
)
from transformers.generation.utils import (
    GenerateDecoderOnlyOutput,
    GenerateNonBeamOutput,
)
from transformers.generation.streamers import BaseStreamer

def debug(d):
    return json.dumps({k: str(v) for k, v in d.items()}, indent=2)

def sample_tokens(probs, generator=None):
    next_token = torch.multinomial(probs, num_samples=1, generator=generator)
    return next_token

def sample_tokens_parallel(probs, generator=None):
    next_token = sample_tokens(probs.view(-1, probs.shape[-1]), generator)
    return next_token.view(probs.shape[0], probs.shape[1])

def _sample(
    self: Qwen2ForCausalLM,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    k: int = 1,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    eos_token_id = 151643
    pad_token_id = 151643

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
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    model_forward = self.__call__
    # compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
    # if compile_forward:
    #     os.environ["TOKENIZERS_PARALLELISM"] = "0"
    #     model_forward = self.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
        is_prefill = False
    else:
        is_prefill = True

    # track 2d finished sequences now
    batch_size, prompt_len = input_ids.shape
    this_peer_finished = False
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
    unfinished_sequences = torch.ones(batch_size, k, dtype=torch.bool, device=input_ids.device)

    dtype = self.dtype
    min_dtype = torch.finfo(dtype).min
    base_mask = torch.full(
        (k, prompt_len),
        fill_value=0, dtype=dtype, device='cpu',
    )
    diag = torch.full((k, k), fill_value=min_dtype, dtype=dtype, device='cpu')
    diag.fill_diagonal_(0)

    cur_len = prompt_len
    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        if is_prefill:
            outputs = self(**model_inputs, return_dict=True)
        else:
            outputs = model_forward(**model_inputs, return_dict=True)

        model_kwargs['past_key_values'] = outputs['past_key_values']
        model_kwargs['cache_position'] = torch.arange(model_kwargs['cache_position'][-1] + 1, model_kwargs['cache_position'][-1] + k + 1, dtype=torch.long, device='cpu')

        # TODO -- this needs to account for prompt padding (just precompute cumsum from initial attention mask)
        model_kwargs['position_ids'] = torch.full((batch_size, k), fill_value=prompt_len + cur_len, device='cpu')

        # TODO -- make this more efficient
        model_kwargs['attention_mask'] = torch.cat((
            base_mask,
            diag.repeat(1, cur_len + 1)
        ), dim=1).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, k, -1)

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
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # NOTE -- updated
        if do_sample:
            probs = F.softmax(next_token_scores, dim=-1)
            next_tokens = sample_tokens_parallel(probs)
        else:
            raise NotImplementedError('todo')

        next_tokens[~unfinished_sequences] = pad_token_id

        # NOTE -- updated
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        # NOTE -- updated
        if cur_len == stopping_criteria.max_length:
            this_peer_finished = True
        else:
            # TODO -- need to test
            unfinished_sequences = unfinished_sequences & ~(next_tokens == eos_token_id)
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

def _sample_default(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    # init values
    # pad_token_id = generation_config._pad_token_tensor
    pad_token_id = 0
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    model_forward = self.__call__
    '''
    compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
    if compile_forward:
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        model_forward = self.get_compiled_call(generation_config.compile_config)
    '''

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
        is_prefill = False
    else:
        is_prefill = True

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        print('model_inputs', debug(model_inputs))
        print('model_kwargs', debug(model_kwargs))

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        if is_prefill:
            outputs = self(**model_inputs, return_dict=True)
            is_prefill = False
        else:
            outputs = model_forward(**model_inputs, return_dict=True)

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        print('updated model_kwargs', debug(model_kwargs))
        if synced_gpus and this_peer_finished:
            continue

        # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = F.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs
        print('###')

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
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

# Qwen2ForCausalLM._sample = _sample
