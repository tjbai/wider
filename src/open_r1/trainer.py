import warnings
from typing import Union, Callable, Any, List, Dict

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, or_masks

from trl.trainer import GRPOTrainer
from trl.trainer.grpo_trainer import nanstd
from trl.models.utils import unwrap_model_for_generation
from trl.data_utils import maybe_apply_chat_template, is_conversational
from trl.extras.profiling import profiling_context, profiling_decorator
from accelerate.utils import gather, gather_object

class ChoreographedTrainer(GRPOTrainer):

    def __init__(self, *args, choreography_k=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.choreography_k = choreography_k
        assert self.use_liger_loss, 'please use liger loss'

    def _parse_interleaved(self, completion_ids: torch.Tensor) -> List[List[str]]:
        return [
            self.processing_class.batch_decode(
                row.reshape(-1, self.choreography_k).T,
                skip_special_tokens=True
            )
            for row in completion_ids
        ]

    def _get_rewards_per_func(self, inputs, prompts, completions):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):
                    raise NotImplementedError()
                else:
                    output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        return rewards_per_func

    @profiling_decorator
    def _prepare_inputs(
        self, accumulated_local_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.gradient_accumulation_steps * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                accumulated_local_batch = self._generate_and_score_completions(accumulated_local_batch)
                self._buffered_inputs = split_tensor_dict(
                    accumulated_local_batch, self.args.gradient_accumulation_steps
                )
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            inputs = self._generate_and_score_completions(accumulated_local_batch)
        return inputs

    def _generate_and_score_completions(self, inputs) -> Dict:
        print('dispatching generate and score')
        print(inputs)

        device = self.accelerator.device
        mode = 'train' if self.model.training else 'eval'

        prompts = [x['prompt'] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)['prompt'] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors='pt', padding=True, padding_side='left', add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs['input_ids'], prompt_inputs['attention_mask']

        print('doesnt get here')

        if self.max_prompt_len is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.use_vllm:
            raise NotImplementedError('vLLM not supported')

        with unwrap_model_for_generation(
            self.model_wrapped,
            self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
        ) as unwrapped_model:
            print('starting generate')
            if (was_ckpt := unwrapped_model.is_gradient_checkpointing):
                unwrapped_model.gradient_checkpointing_disable()
                unwrapped_model.config.use_cache = True

            prompt_completion_ids = unwrapped_model.generate(
                prompt_ids,
                attention_mask=prompt_mask,
                generation_config=self.generation_config,
                use_cache=True
            )

            if was_ckpt:
                unwrapped_model.gradient_checkpointing_enable()

        print('done generating')
        _, prompt_len = prompt_ids.shape
        prompt_ids = prompt_completion_ids[:, :prompt_len]
        completion_ids = prompt_completion_ids[:, prompt_len:]

        # needs to be interleaved perfectly!
        batch_size, comp_len = completion_ids.shape
        assert comp_len % self.choreography_k == 0

        thread_ids = completion_ids\
            .view(batch_size, -1, self.choreography_k)\
            .transpose(-1, -2)\
            .reshape(batch_size * self.choreography_k, -1) # (B*K, N)

        is_eos = thread_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        thread_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        completion_mask = thread_mask\
            .view(batch_size, self.choreography_k, -1)\
            .transpose(-1, -2)\
            .reshape(batch_size, -1)

        def prefix(b, _, q_idx, kv_idx):
            if q_idx >= prompt_len:
                return kv_idx < prompt_len & prompt_mask[b, kv_idx] > 0
            return q_idx >= kv_idx & prompt_mask[b, kv_idx] > 0

        def interleaved(b, _, q_idx, kv_idx):
            return q_idx >= prompt_len & ((q_idx - prompt_len) % self.choreography_k) == ((kv_idx - prompt_len) % self.choreography_k)

        choreographed_mask = create_block_mask(
            mask_mod=or_masks(prefix, interleaved),
            B=batch_size, H=None,
            Q_LEN=prompt_len+comp_len,
            KV_LEN=prompt_len+comp_len,
        )

        # so the mask comprises 2 parts... for now we will broadcast across the head dimension but NOT the batch dimension

        with torch.no_grad():
            print('computing log probs and shit')
            logps_kwargs = {
                'input_ids': prompt_completion_ids,
                'logits_to_keep': comp_len,
                'batch_size': self.args.per_device_train_batch_size if mode == 'train' else self.args.per_device_eval_batch_size,
                'attention_mask': choreographed_mask,
            }

            if self.num_iterations == 1:
                old_per_token_logps = None
            else:
                old_per_token_logps = self._get_per_token_logps(self.model, **logps_kwargs)

            logps_kwargs['attention_mask'] = torch.cat([prompt_mask, completion_mask], dim=1)
            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, **logps_kwargs)
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(self.model, **logps_kwargs)

        completions_text = self._parse_interleaved(completion_ids)
        print(f'completions:\n{completions_text}')
        if is_conversational(inputs[0]):
            completions = []
            for prompt, batch in zip(prompts, completions_text):
                prefill = prompt.pop()['content'] if prompt[-1]['role'] == 'assistant' else ''
                completions.append([
                    [{'role': 'assistant', 'content': prefill + completion}]
                    for completion in batch
                ])
        else:
            completions = completions_text

        rewards_per_func = self._get_rewards_per_func(inputs, prompts, completions)
        rewards_per_func = gather(rewards_per_func)

        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # group rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # advantage estimates
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # grab the data for local process
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        if mode == 'train':
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]['num_tokens'] = [self.state.num_input_tokens_seen]

        agg_completion_mask = self.accelerator.gather_for_metrics(completion_mask.sum(1))
        self._metrics[mode]['completions/mean_length'].append(agg_completion_mask.float().mean().item())
        self._metrics[mode]['completions/min_length'].append(agg_completion_mask.float().min().item())
        self._metrics[mode]['completions/max_length'].append(agg_completion_mask.float().max().item())

        # these might be funky/useless if we don't add a 'join' step
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(agg_completion_mask)
        self._metrics[mode]['completions/clipped_ratio'].append(clipped_completions_ratio)
        if len(term_completion_mask) == 0:
            # edge case where no completed sequences are found
            term_completion_mask = torch.zeros(1, device=device)
        self._metrics[mode]['completions/mean_terminated_length'].append(term_completion_mask.float().mean().item())
        self._metrics[mode]['completions/min_terminated_length'].append(term_completion_mask.float().min().item())
        self._metrics[mode]['completions/max_terminated_length'].append(term_completion_mask.float().max().item())

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f'rewards/{reward_func_name}/mean'].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f'rewards/{reward_func_name}/std'].append(std_rewards)
        self._metrics[mode]['reward'].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]['reward_std'].append(std_grouped_rewards.mean().item())

        self._textual_logs['prompt'].extend(gather_object(prompts_text))
        self._textual_logs['completion'].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs['rewards'][name].extend(rewards_per_func[:, i].tolist())

        return {
            'prompt_ids': prompt_ids,
            'prompt_mask': prompt_mask,
            'completion_ids': completion_ids,
            'completion_mask': completion_mask,
            'advantages': advantages,
            'old_per_token_logps': old_per_token_logps,
            'ref_per_token_logps': ref_per_token_logps,
            'choreographed_mask': choreographed_mask,
        }

    def compute_liger_loss(self, model, inputs):
        last_hidden_state = self._get_last_hidden_state(
            model,
            input_ids=torch.cat([inputs['prompt_ids'], inputs['completion_ids']], dim=1),
            attention_mask=inputs['choregraphed_mask'],
            logits_to_keep=inputs['completion_ids'].shape(1)
        )

        unwrapped_model = self.accelerator.unwrap_model(model)
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=inputs['completion_ids'],
            attention_mask=inputs['completion_mask'],
            advantages=inputs['advantages'],
            bias=unwrapped_model.lm_head.bias,
            ref_per_token_logps=inputs['ref_per_token_logps'],
            old_per_token_logps=inputs['old_per_token_logps'],
        )

        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]
        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss
