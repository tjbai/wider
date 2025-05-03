from typing import Optional
from datetime import timedelta

import fire
import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.utils import EnvConfig
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.utils.imports import is_accelerate_available

def main(
    model_name_or_path: str,
    tasks: str,
    output_dir: str,
    revision: str = 'main',
    batch_size: int = 1,
    max_length: int = 32768,
    max_samples: Optional[int] = None,
    push_to_hub: bool = False,
    hub_organization: Optional[str] = None,
    custom_model_class: Optional[str] = None,
):
    accelerator = None
    if is_accelerate_available():
        from accelerate import Accelerator, InitProcessGroupKwargs
        accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])

    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        save_details=True,
        push_to_hub=push_to_hub,
        hub_results_org=hub_organization,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        env_config=EnvConfig(cache_dir='~/.cache/huggingface'),
        override_batch_size=batch_size,
        max_samples=max_samples,
    )

    model_kwargs = {
        'pretrained': model_name_or_path,
        'revision': revision,
        'dtype': 'auto',
        'trust_remote_code': False,
        'use_chat_template': False,
        'batch_size': batch_size,
        'device': 'cuda',
    }
    if custom_model_class:
        model_kwargs['model_class_name'] = custom_model_class

    model_config = TransformersModelConfig(**model_kwargs)

    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()

if __name__ == '__main__':
    fire.Fire(main)
