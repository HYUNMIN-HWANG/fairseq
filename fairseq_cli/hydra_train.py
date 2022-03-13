#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from fairseq import distributed_utils, metrics
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.initialize import add_defaults, hydra_init
from fairseq.dataclass.utils import omegaconf_no_object_check
from fairseq.utils import reset_logging
from fairseq_cli.train import main as pre_main

logger = logging.getLogger("fairseq_cli.hydra_train")


@hydra.main(config_path=os.path.join("..", "fairseq", "config"), config_name="config")
def hydra_main(cfg: FairseqConfig) -> float:
    _hydra_main(cfg)


def _hydra_main(cfg: FairseqConfig, **kwargs) -> float:
    import ipdb; ipdb.set_trace()
    add_defaults(cfg)

    if cfg.common.reset_logging:
        reset_logging()  # Hydra hijacks logging, fix that
    else:
        # check if directly called or called through hydra_main
        if HydraConfig.initialized():
            with open_dict(cfg):
                # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
                cfg.job_logging_cfg = OmegaConf.to_container(
                    HydraConfig.get().job_logging, resolve=True
                )

    with omegaconf_no_object_check():
        cfg = OmegaConf.create(
            OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
        )
    OmegaConf.set_struct(cfg, True)

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, pre_main, **kwargs)
        else:
            distributed_utils.call_main(cfg, pre_main, **kwargs)
    except BaseException as e:
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! " + str(e))

    # get best val and return - useful for sweepers
    try:
        best_val = metrics.get_smoothed_value(
            "valid", cfg.checkpoint.best_checkpoint_metric
        )
    except:
        best_val = None

    if best_val is None:
        best_val = float("inf")

    return best_val

"""
(31- add_defaults(cfg) 전)
{'_name': None, 
 'common': {'_name': None, 'no_progress_bar': False, 'log_interval': 200, 'log_format': json, 'log_file': None, 'tensorboard_logdir': 'tb', 'wandb_project': None, 'azureml_logging': False, 'seed': 1, 'cpu': False, 'tpu': False, 'bf16': False, 'memory_efficient_bf16': False, 'fp16': True, 'memory_efficient_fp16': False, 'fp16_no_flatten_grads': False, 'fp16_init_scale': 128, 'fp16_scale_window': None, 'fp16_scale_tolerance': 0.0, 'on_cpu_convert_precision': False, 'min_loss_scale': 0.0001, 'threshold_loss_scale': None, 'amp': False, 'amp_batch_retries': 2, 'amp_init_scale': 128, 'amp_scale_window': None, 'user_dir': 'examples/data2vec', 'empty_cache_freq': 0, 'all_gather_list_size': 16384, 'model_parallel_size': 1, 'quantization_config_path': None, 'profile': False, 'reset_logging': False, 'suppress_crashes': False, 'use_plasma_view': False, 'plasma_path': '/tmp/plasma'}, 
 'common_eval': {'_name': None, 'path': None, 'post_process': None, 'quiet': False, 'model_overrides': '{}', 'results_path': None}, 
 'distributed_training': {'_name': None, 'distributed_world_size': 4, 'distributed_num_procs': 4, 'distributed_rank': 0, 'distributed_backend': 'nccl', 'distributed_init_method': None, 'distributed_port': -1, 'device_id': 0, 'distributed_no_spawn': False, 'ddp_backend': legacy_ddp, 'ddp_comm_hook': none, 'bucket_cap_mb': 25, 'fix_batches_to_gpus': False, 'find_unused_parameters': False, 'gradient_as_bucket_view': False, 'fast_stat_sync': False, 'heartbeat_timeout': -1, 'broadcast_buffers': False, 'slowmo_momentum': None, 'slowmo_base_algorithm': 'localsgd', 'localsgd_frequency': 3, 'nprocs_per_node': 4, 'pipeline_model_parallel': False, 'pipeline_balance': None, 'pipeline_devices': None, 'pipeline_chunks': 0, 'pipeline_encoder_balance': None, 'pipeline_encoder_devices': None, 'pipeline_decoder_balance': None, 'pipeline_decoder_devices': None, 'pipeline_checkpoint': never, 'zero_sharding': none, 'fp16': '${common.fp16}', 'memory_efficient_fp16': '${common.memory_efficient_fp16}', 'tpu': '${common.tpu}', 'no_reshard_after_forward': False, 'fp32_reduce_scatter': False, 'cpu_offload': False, 'use_sharded_state': False, 'not_fsdp_flatten_parameters': False}, 
 'dataset': {'_name': None, 'num_workers': 6, 'skip_invalid_size_inputs_valid_test': True, 'max_tokens': 3800000, 'batch_size': None, 'required_batch_size_multiple': 1, 'required_seq_len_multiple': 1, 'dataset_impl': None, 'data_buffer_size': 10, 'train_subset': 'train', 'valid_subset': 'valid', 'combine_valid_subsets': None, 'ignore_unused_valid_subsets': False, 'validate_interval': 5, 'validate_interval_updates': 0, 'validate_after_updates': 0, 'fixed_validation_seed': None, 'disable_validation': True, 'max_tokens_valid': '${dataset.max_tokens}', 'batch_size_valid': '${dataset.batch_size}', 'max_valid_steps': None, 'curriculum': 0, 'gen_subset': 'test', 'num_shards': 1, 'shard_id': 0, 'grouped_shuffling': False, 'update_epoch_batch_itr': '${dataset.grouped_shuffling}', 'update_ordered_indices_seed': False}, 
 'optimization': {'_name': None, 'max_epoch': 0, 'max_update': 400000, 'stop_time_hours': 0.0, 'clip_norm': 0.0, 'sentence_avg': False, 'update_freq': [4], 'lr': [0.0005], 'stop_min_lr': -1.0, 'use_bmuf': False, 'skip_remainder_batch': False}, 
 'checkpoint': {'_name': None, 'save_dir': 'checkpoints', 'restore_file': 'checkpoint_last.pt', 'continue_once': None, 'finetune_from_model': None, 'reset_dataloader': False, 'reset_lr_scheduler': False, 'reset_meters': False, 'reset_optimizer': False, 'optimizer_overrides': '{}', 'save_interval': 5, 'save_interval_updates': 25000, 'keep_interval_updates': 1, 'keep_interval_updates_pattern': -1, 'keep_last_epochs': -1, 'keep_best_checkpoints': -1, 'no_save': False, 'no_epoch_checkpoints': True, 'no_last_checkpoints': False, 'no_save_optimizer_state': False, 'best_checkpoint_metric': 'loss', 'maximize_best_checkpoint_metric': False, 'patience': -1, 'checkpoint_suffix': '', 'checkpoint_shard_count': 1, 'load_checkpoint_on_all_dp_ranks': False, 'write_checkpoints_asynchronously': False, 'model_parallel_size': '${common.model_parallel_size}'}, 
 'bmuf': {'_name': None, 'block_lr': 1.0, 'block_momentum': 0.875, 'global_sync_iter': 50, 'warmup_iterations': 500, 'use_nbm': False, 'average_sync': False, 'distributed_world_size': '${distributed_training.distributed_world_size}'}, 
 'generation': {'_name': None, 'beam': 5, 'nbest': 1, 'max_len_a': 0.0, 'max_len_b': 200, 'min_len': 1, 'match_source_len': False, 'unnormalized': False, 'no_early_stop': False, 'no_beamable_mm': False, 'lenpen': 1.0, 'unkpen': 0.0, 'replace_unk': None, 'sacrebleu': False, 'score_reference': False, 'prefix_size': 0, 'no_repeat_ngram_size': 0, 'sampling': False, 'sampling_topk': -1, 'sampling_topp': -1.0, 'constraints': None, 'temperature': 1.0, 'diverse_beam_groups': -1, 'diverse_beam_strength': 0.5, 'diversity_rate': -1.0, 'print_alignment': None, 'print_step': False, 'lm_path': None, 'lm_weight': 0.0, 'iter_decode_eos_penalty': 0.0, 'iter_decode_max_iter': 10, 'iter_decode_force_max_iter': False, 'iter_decode_with_beam': 1, 'iter_decode_with_external_reranker': False, 'retain_iter_history': False, 'retain_dropout': False, 'retain_dropout_modules': None, 'decoding_format': None, 'no_seed_provided': False}, 
 'eval_lm': {'_name': None, 'output_word_probs': False, 'output_word_stats': False, 'context_window': 0, 'softmax_batch': 9223372036854775807}, 
 'interactive': {'_name': None, 'buffer_size': 0, 'input': '-'}, 
 'model': {'_name': 'data2vec_audio', 'extractor_mode': 'layer_norm', 'encoder_layerdrop': 0.05, 'dropout_input': 0.0, 'dropout_features': 0.0, 'feature_grad_mult': 1.0, 'encoder_embed_dim': 768, 'mask_prob': 0.65, 'mask_length': 10, 'loss_beta': 0, 'loss_scale': None, 'instance_norm_target_layer': True, 'average_top_k_layers': 8, 'pos_conv_depth': 5, 'conv_pos': 95, 'ema_decay': 0.999, 'ema_end_decay': 0.9999, 'ema_anneal_end_step': 30000, 'ema_transformer_only': True, 'ema_layers_only': True, 'require_same_masks': True, 'mask_dropout': 0}, 
 'task': {'_name': 'audio_pretraining', 'data': '/path/to/manifests', 'max_sample_size': 320000, 'min_sample_size': 32000, 'normalize': True}, 
 'criterion': {'_name': 'model', 'log_keys': ['ema_decay', 'target_var', 'pred_var']}, 
 'optimizer': {'_name': 'adam', 'adam_betas': '(0.9,0.98)', 'adam_eps': 1e-06, 'weight_decay': 0.01}, 
 'lr_scheduler': {'_name': 'tri_stage', 'phase_ratio': [0.03, 0.9, 0.07]}, 
 'scoring': None, 
 'bpe': None, 
 'tokenizer': None, 
 'ema': {'_name': None, 'store_ema': False, 'ema_decay': 0.9999, 'ema_start_update': 0, 'ema_seed_model': None, 'ema_update_freq': 1, 'ema_fp32': False}}
"""

"""
(51 OmegaConf.set_struct(cfg, True) 후)
{'_name': None, 
 'common': {'_name': None, 'no_progress_bar': False, 'log_interval': 200, 'log_format': 'json', 'log_file': None, 'tensorboard_logdir': 'tb', 'wandb_project': None, 'azureml_logging': False, 'seed': 1, 'cpu': False, 'tpu': False, 'bf16': False, 'memory_efficient_bf16': False, 'fp16': True, 'memory_efficient_fp16': False, 'fp16_no_flatten_grads': False, 'fp16_init_scale': 128, 'fp16_scale_window': None, 'fp16_scale_tolerance': 0.0, 'on_cpu_convert_precision': False, 'min_loss_scale': 0.0001, 'threshold_loss_scale': None, 'amp': False, 'amp_batch_retries': 2, 'amp_init_scale': 128, 'amp_scale_window': None, 'user_dir': 'examples/data2vec', 'empty_cache_freq': 0, 'all_gather_list_size': 16384, 'model_parallel_size': 1, 'quantization_config_path': None, 'profile': False, 'reset_logging': False, 'suppress_crashes': False, 'use_plasma_view': False, 'plasma_path': '/tmp/plasma'}, 
 'common_eval': {'_name': None, 'path': None, 'post_process': None, 'quiet': False, 'model_overrides': '{}', 'results_path': None}, 
 'distributed_training': {'_name': None, 'distributed_world_size': 4, 'distributed_num_procs': 4, 'distributed_rank': 0, 'distributed_backend': 'nccl', 'distributed_init_method': None, 'distributed_port': -1, 'device_id': 0, 'distributed_no_spawn': False, 'ddp_backend': 'legacy_ddp', 'ddp_comm_hook': 'none', 'bucket_cap_mb': 25, 'fix_batches_to_gpus': False, 'find_unused_parameters': False, 'gradient_as_bucket_view': False, 'fast_stat_sync': False, 'heartbeat_timeout': -1, 'broadcast_buffers': False, 'slowmo_momentum': None, 'slowmo_base_algorithm': 'localsgd', 'localsgd_frequency': 3, 'nprocs_per_node': 4, 'pipeline_model_parallel': False, 'pipeline_balance': None, 'pipeline_devices': None, 'pipeline_chunks': 0, 'pipeline_encoder_balance': None, 'pipeline_encoder_devices': None, 'pipeline_decoder_balance': None, 'pipeline_decoder_devices': None, 'pipeline_checkpoint': 'never', 'zero_sharding': 'none', 'fp16': True, 'memory_efficient_fp16': False, 'tpu': False, 'no_reshard_after_forward': False, 'fp32_reduce_scatter': False, 'cpu_offload': False, 'use_sharded_state': False, 'not_fsdp_flatten_parameters': False}, 
 'dataset': {'_name': None, 'num_workers': 6, 'skip_invalid_size_inputs_valid_test': True, 'max_tokens': 3800000, 'batch_size': None, 'required_batch_size_multiple': 1, 'required_seq_len_multiple': 1, 'dataset_impl': None, 'data_buffer_size': 10, 'train_subset': 'train', 'valid_subset': 'valid', 'combine_valid_subsets': None, 'ignore_unused_valid_subsets': False, 'validate_interval': 5, 'validate_interval_updates': 0, 'validate_after_updates': 0, 'fixed_validation_seed': None, 'disable_validation': True, 'max_tokens_valid': 3800000, 'batch_size_valid': None, 'max_valid_steps': None, 'curriculum': 0, 'gen_subset': 'test', 'num_shards': 1, 'shard_id': 0, 'grouped_shuffling': False, 'update_epoch_batch_itr': False, 'update_ordered_indices_seed': False}, 
 'optimization': {'_name': None, 'max_epoch': 0, 'max_update': 400000, 'stop_time_hours': 0.0, 'clip_norm': 0.0, 'sentence_avg': False, 'update_freq': [4], 'lr': [0.0005], 'stop_min_lr': -1.0, 'use_bmuf': False, 'skip_remainder_batch': False}, 
 'checkpoint': {'_name': None, 'save_dir': 'checkpoints', 'restore_file': 'checkpoint_last.pt', 'continue_once': None, 'finetune_from_model': None, 'reset_dataloader': False, 'reset_lr_scheduler': False, 'reset_meters': False, 'reset_optimizer': False, 'optimizer_overrides': '{}', 'save_interval': 5, 'save_interval_updates': 25000, 'keep_interval_updates': 1, 'keep_interval_updates_pattern': -1, 'keep_last_epochs': -1, 'keep_best_checkpoints': -1, 'no_save': False, 'no_epoch_checkpoints': True, 'no_last_checkpoints': False, 'no_save_optimizer_state': False, 'best_checkpoint_metric': 'loss', 'maximize_best_checkpoint_metric': False, 'patience': -1, 'checkpoint_suffix': '', 'checkpoint_shard_count': 1, 'load_checkpoint_on_all_dp_ranks': False, 'write_checkpoints_asynchronously': False, 'model_parallel_size': 1}, 
 'bmuf': {'_name': None, 'block_lr': 1.0, 'block_momentum': 0.875, 'global_sync_iter': 50, 'warmup_iterations': 500, 'use_nbm': False, 'average_sync': False, 'distributed_world_size': 4},
 'generation': {'_name': None, 'beam': 5, 'nbest': 1, 'max_len_a': 0.0, 'max_len_b': 200, 'min_len': 1, 'match_source_len': False, 'unnormalized': False, 'no_early_stop': False, 'no_beamable_mm': False, 'lenpen': 1.0, 'unkpen': 0.0, 'replace_unk': None, 'sacrebleu': False, 'score_reference': False, 'prefix_size': 0, 'no_repeat_ngram_size': 0, 'sampling': False, 'sampling_topk': -1, 'sampling_topp': -1.0, 'constraints': None, 'temperature': 1.0, 'diverse_beam_groups': -1, 'diverse_beam_strength': 0.5, 'diversity_rate': -1.0, 'print_alignment': None, 'print_step': False, 'lm_path': None, 'lm_weight': 0.0, 'iter_decode_eos_penalty': 0.0, 'iter_decode_max_iter': 10, 'iter_decode_force_max_iter': False, 'iter_decode_with_beam': 1, 'iter_decode_with_external_reranker': False, 'retain_iter_history': False, 'retain_dropout': False, 'retain_dropout_modules': None, 'decoding_format': None, 'no_seed_provided': False}, 
 'eval_lm': {'_name': None, 'output_word_probs': False, 'output_word_stats': False, 'context_window': 0, 'softmax_batch': 9223372036854775807}, 
 'interactive': {'_name': None, 'buffer_size': 0, 'input': '-'}, 
 'model': {'_name': 'data2vec_audio', 'extractor_mode': 'layer_norm', 'encoder_layerdrop': 0.05, 'dropout_input': 0.0, 'dropout_features': 0.0, 'feature_grad_mult': 1.0, 'encoder_embed_dim': 768, 'mask_prob': 0.65, 'mask_length': 10, 'loss_beta': 0, 'loss_scale': None, 'instance_norm_target_layer': True, 'average_top_k_layers': 8, 'pos_conv_depth': 5, 'conv_pos': 95, 'ema_decay': 0.999, 'ema_end_decay': 0.9999, 'ema_anneal_end_step': 30000, 'ema_transformer_only': True, 'ema_layers_only': True, 'require_same_masks': True, 'mask_dropout': 0}, 
 'task': {'_name': 'audio_pretraining', 'data': '/path/to/manifests', 'labels': None, 'binarized_dataset': False, 'sample_rate': 16000, 'normalize': True, 'enable_padding': False, 'max_sample_size': 320000, 'min_sample_size': 32000, 'num_batch_buckets': 0, 'precompute_mask_indices': False, 'inferred_w2v_config': None, 'tpu': False, 'text_compression_level': 'none'}, 
 'criterion': {'_name': 'model', 'loss_weights': {}, 'log_keys': ['ema_decay', 'target_var', 'pred_var']}, 
 'optimizer': {'_name': 'adam', 'adam_betas': '(0.9,0.98)', 'adam_eps': 1e-06, 'weight_decay': 0.01, 'use_old_adam': False, 'fp16_adam_stats': False, 'tpu': False, 'lr': [0.0005]}, 
 'lr_scheduler': {'_name': 'tri_stage', 'warmup_steps': 0, 'hold_steps': 0, 'decay_steps': 0, 'phase_ratio': [0.03, 0.9, 0.07], 'init_lr_scale': 0.01, 'final_lr_scale': 0.01, 'max_update': 400000, 'lr': [0.0005]}, 
 'scoring': None, 
 'bpe': None, 
 'tokenizer': None, 
 'ema': {'_name': None, 'store_ema': False, 'ema_decay': 0.9999, 'ema_start_update': 0, 'ema_seed_model': None, 'ema_update_freq': 1, 'ema_fp32': False}, 
 'job_logging_cfg': {'version': 1, 'formatters': {'simple': {'format': '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'}}, 
 'handlers': {'console': {'class': 'logging.StreamHandler', 'formatter': 'simple', 'stream': 'ext://sys.stdout'}, 'file': {'class': 'logging.FileHandler', 'formatter': 'simple', 'filename': 'hydra_train.log'}}, 
 'root': {'level': 'INFO', 'handlers': ['console', 'file']}, 
 'disable_existing_loggers': False}}
"""

"""
{'_name': None, 
 'common': {'_name': None, 'no_progress_bar': False, 'log_interval': 200, 'log_format': 'json', 'log_file': None, 'tensorboard_logdir': 'tb', 'wandb_project': None, 'azureml_logging': False, 'seed': 1, 'cpu': False, 'tpu': False, 'bf16': False, 'memory_efficient_bf16': False, 'fp16': True, 'memory_efficient_fp16': False, 'fp16_no_flatten_grads': False, 'fp16_init_scale': 128, 'fp16_scale_window': None, 'fp16_scale_tolerance': 0.0, 'on_cpu_convert_precision': False, 'min_loss_scale': 0.0001, 'threshold_loss_scale': None, 'amp': False, 'amp_batch_retries': 2, 'amp_init_scale': 128, 'amp_scale_window': None, 'user_dir': 'examples/data2vec', 'empty_cache_freq': 0, 'all_gather_list_size': 16384, 'model_parallel_size': 1, 'quantization_config_path': None, 'profile': False, 'reset_logging': False, 'suppress_crashes': False, 'use_plasma_view': False, 'plasma_path': '/tmp/plasma'}, 
 'common_eval': {'_name': None, 'path': None, 'post_process': None, 'quiet': False, 'model_overrides': '{}', 'results_path': None}, 
 'distributed_training': {'_name': None, 'distributed_world_size': 4, 'distributed_num_procs': 4, 'distributed_rank': 0, 'distributed_backend': 'nccl', 'distributed_init_method': 'tcp://localhost:12682', 'distributed_port': -1, 'device_id': 0, 'distributed_no_spawn': False, 'ddp_backend': 'legacy_ddp', 'ddp_comm_hook': 'none', 'bucket_cap_mb': 25, 'fix_batches_to_gpus': False, 'find_unused_parameters': False, 'gradient_as_bucket_view': False, 'fast_stat_sync': False, 'heartbeat_timeout': -1, 'broadcast_buffers': False, 'slowmo_momentum': None, 'slowmo_base_algorithm': 'localsgd', 'localsgd_frequency': 3, 'nprocs_per_node': 4, 'pipeline_model_parallel': False, 'pipeline_balance': None, 'pipeline_devices': None, 'pipeline_chunks': 0, 'pipeline_encoder_balance': None, 'pipeline_encoder_devices': None, 'pipeline_decoder_balance': None, 'pipeline_decoder_devices': None, 'pipeline_checkpoint': 'never', 'zero_sharding': 'none', 'fp16': True, 'memory_efficient_fp16': False, 'tpu': False, 'no_reshard_after_forward': False, 'fp32_reduce_scatter': False, 'cpu_offload': False, 'use_sharded_state': False, 'not_fsdp_flatten_parameters': False}, 
 'dataset': {'_name': None, 'num_workers': 6, 'skip_invalid_size_inputs_valid_test': True, 'max_tokens': 3800000, 'batch_size': None, 'required_batch_size_multiple': 1, 'required_seq_len_multiple': 1, 'dataset_impl': None, 'data_buffer_size': 10, 'train_subset': 'train', 'valid_subset': 'valid', 'combine_valid_subsets': None, 'ignore_unused_valid_subsets': False, 'validate_interval': 5, 'validate_interval_updates': 0, 'validate_after_updates': 0, 'fixed_validation_seed': None, 'disable_validation': True, 'max_tokens_valid': 3800000, 'batch_size_valid': None, 'max_valid_steps': None, 'curriculum': 0, 'gen_subset': 'test', 'num_shards': 1, 'shard_id': 0, 'grouped_shuffling': False, 'update_epoch_batch_itr': False, 'update_ordered_indices_seed': False}, 
 'optimization': {'_name': None, 'max_epoch': 0, 'max_update': 400000, 'stop_time_hours': 0.0, 'clip_norm': 0.0, 'sentence_avg': False, 'update_freq': [4], 'lr': [0.0005], 'stop_min_lr': -1.0, 'use_bmuf': False, 'skip_remainder_batch': False}, 
 'checkpoint': {'_name': None, 'save_dir': 'checkpoints', 'restore_file': 'checkpoint_last.pt', 'continue_once': None, 'finetune_from_model': None, 'reset_dataloader': False, 'reset_lr_scheduler': False, 'reset_meters': False, 'reset_optimizer': False, 'optimizer_overrides': '{}', 'save_interval': 5, 'save_interval_updates': 25000, 'keep_interval_updates': 1, 'keep_interval_updates_pattern': -1, 'keep_last_epochs': -1, 'keep_best_checkpoints': -1, 'no_save': False, 'no_epoch_checkpoints': True, 'no_last_checkpoints': False, 'no_save_optimizer_state': False, 'best_checkpoint_metric': 'loss', 'maximize_best_checkpoint_metric': False, 'patience': -1, 'checkpoint_suffix': '', 'checkpoint_shard_count': 1, 'load_checkpoint_on_all_dp_ranks': False, 'write_checkpoints_asynchronously': False, 'model_parallel_size': 1}, 
 'bmuf': {'_name': None, 'block_lr': 1.0, 'block_momentum': 0.875, 'global_sync_iter': 50, 'warmup_iterations': 500, 'use_nbm': False, 'average_sync': False, 'distributed_world_size': 4}, 
 'generation': {'_name': None, 'beam': 5, 'nbest': 1, 'max_len_a': 0.0, 'max_len_b': 200, 'min_len': 1, 'match_source_len': False, 'unnormalized': False, 'no_early_stop': False, 'no_beamable_mm': False, 'lenpen': 1.0, 'unkpen': 0.0, 'replace_unk': None, 'sacrebleu': False, 'score_reference': False, 'prefix_size': 0, 'no_repeat_ngram_size': 0, 'sampling': False, 'sampling_topk': -1, 'sampling_topp': -1.0, 'constraints': None, 'temperature': 1.0, 'diverse_beam_groups': -1, 'diverse_beam_strength': 0.5, 'diversity_rate': -1.0, 'print_alignment': None, 'print_step': False, 'lm_path': None, 'lm_weight': 0.0, 'iter_decode_eos_penalty': 0.0, 'iter_decode_max_iter': 10, 'iter_decode_force_max_iter': False, 'iter_decode_with_beam': 1, 'iter_decode_with_external_reranker': False, 'retain_iter_history': False, 'retain_dropout': False, 'retain_dropout_modules': None, 'decoding_format': None, 'no_seed_provided': False}, 
 'eval_lm': {'_name': None, 'output_word_probs': False, 'output_word_stats': False, 'context_window': 0, 'softmax_batch': 9223372036854775807}, 
 'interactive': {'_name': None, 'buffer_size': 0, 'input': '-'}, 
 'model': {'_name': 'data2vec_audio', 'extractor_mode': layer_norm, 'encoder_layers': 12, 'encoder_embed_dim': 768, 'encoder_ffn_embed_dim': 3072, 'encoder_attention_heads': 12, 'activation_fn': gelu, 'layer_type': transformer, 'dropout': 0.1, 'attention_dropout': 0.1, 'activation_dropout': 0.0, 'encoder_layerdrop': 0.05, 'dropout_input': 0.0, 'dropout_features': 0.0, 'final_dim': 0, 'layer_norm_first': False, 'conv_feature_layers': '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]', 'conv_bias': False, 'logit_temp': 0.1, 'quantize_targets': False, 'quantize_input': False, 'same_quantizer': False, 'target_glu': False, 'feature_grad_mult': 1.0, 'quantizer_depth': 1, 'quantizer_factor': 3, 'latent_vars': 320, 'latent_groups': 2, 'latent_dim': 0, 'mask_length': 10, 'mask_prob': 0.65, 'mask_selection': static, 'mask_other': 0.0, 'no_mask_overlap': False, 'mask_min_space': 1, 'require_same_masks': True, 'mask_dropout': 0.0, 'mask_channel_length': 10, 'mask_channel_prob': 0.0, 'mask_channel_before': False, 'mask_channel_selection': static, 'mask_channel_other': 0.0, 'no_mask_channel_overlap': False, 'mask_channel_min_space': 1, 'num_negatives': 100, 'negatives_from_everywhere': False, 'cross_sample_negatives': 0, 'codebook_negatives': 0, 'conv_pos': 95, 'conv_pos_groups': 16, 'pos_conv_depth': 5, 'latent_temp': [2.0, 0.5, 0.999995], 'max_positions': 100000, 'checkpoint_activations': False, 'required_seq_len_multiple': 2, 'crop_seq_to_multiple': 1, 'depthwise_conv_kernel_size': 31, 'attn_type': '', 'pos_enc_type': 'abs', 'fp16': False, 'loss_beta': 0.0, 'loss_scale': None, 'average_top_k_layers': 8, 'layer_norm_target_layer': False, 'instance_norm_target_layer': True, 'instance_norm_targets': False, 'layer_norm_targets': False, 'batch_norm_target_layer': False, 'group_norm_target_layer': False, 'ema_decay': 0.999, 'ema_end_decay': 0.9999, 'ema_anneal_end_step': 30000, 'ema_transformer_only': True, 'ema_layers_only': True, 'max_update': '${optimization.max_update}', 'min_target_var': 0.1, 'min_pred_var': 0.01}, 
 'task': {'_name': 'audio_pretraining', 'data': '/path/to/manifests', 'labels': None, 'binarized_dataset': False, 'sample_rate': 16000, 'normalize': True, 'enable_padding': False, 'max_sample_size': 320000, 'min_sample_size': 32000, 'num_batch_buckets': 0, 'precompute_mask_indices': False, 'inferred_w2v_config': None, 'tpu': False, 'text_compression_level': none}, 
 'criterion': {'_name': 'model', 'loss_weights': {}, 'log_keys': ['ema_decay', 'target_var', 'pred_var']}, 'optimizer': {'_name': 'adam', 'adam_betas': '(0.9,0.98)', 'adam_eps': 1e-06, 'weight_decay': 0.01, 'use_old_adam': False, 'fp16_adam_stats': False, 'tpu': False, 'lr': [0.0005]}, 
 'lr_scheduler': {'_name': 'tri_stage', 'warmup_steps': 0, 'hold_steps': 0, 'decay_steps': 0, 'phase_ratio': [0.03, 0.9, 0.07], 'init_lr_scale': 0.01, 'final_lr_scale': 0.01, 'max_update': 400000.0, 'lr': [0.0005]}, 
 'scoring': None, 
 'bpe': None, 
 'tokenizer': None, 
 'ema': {'_name': None, 'store_ema': False, 'ema_decay': 0.9999, 'ema_start_update': 0, 'ema_seed_model': None, 'ema_update_freq': 1, 'ema_fp32': False}, 
 'job_logging_cfg': {'version': 1, 'formatters': {'simple': {'format': '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'}}, 
 'handlers': {'console': {'class': 'logging.StreamHandler', 'formatter': 'simple', 'stream': 'ext://sys.stdout'}, 
 'file': {'class': 'logging.FileHandler', 'formatter': 'simple', 'filename': 'hydra_train.log'}}, 
 'root': {'level': 'INFO', 'handlers': ['console', 'file']}, 'disable_existing_loggers': False}}
"""

"""
[2022-03-13 06:33:53,549][fairseq_cli.train][INFO] - Data2VecAudioModel(
  (feature_extractor): ConvFeatureExtractionModel(
    (conv_layers): ModuleList(
      (0): Sequential(
        (0): Conv1d(1, 512, kernel_size=(10,), stride=(5,), bias=False)
        (1): Dropout(p=0.0, inplace=False)
        (2): Sequential(
          (0): TransposeLast()
          (1): Fp32LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (2): TransposeLast()
        )
        (3): GELU()
      )
      (1): Sequential(
        (0): Conv1d(512, 512, kernel_size=(3,), stride=(2,), bias=False)
        (1): Dropout(p=0.0, inplace=False)
        (2): Sequential(
          (0): TransposeLast()
          (1): Fp32LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (2): TransposeLast()
        )
        (3): GELU()
      )
      (2): Sequential(
        (0): Conv1d(512, 512, kernel_size=(3,), stride=(2,), bias=False)
        (1): Dropout(p=0.0, inplace=False)
        (2): Sequential(
          (0): TransposeLast()
          (1): Fp32LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (2): TransposeLast()
        )
        (3): GELU()
      )
      (3): Sequential(
        (0): Conv1d(512, 512, kernel_size=(3,), stride=(2,), bias=False)
        (1): Dropout(p=0.0, inplace=False)
        (2): Sequential(
          (0): TransposeLast()
          (1): Fp32LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (2): TransposeLast()
        )
        (3): GELU()
      )
      (4): Sequential(
        (0): Conv1d(512, 512, kernel_size=(3,), stride=(2,), bias=False)
        (1): Dropout(p=0.0, inplace=False)
        (2): Sequential(
          (0): TransposeLast()
          (1): Fp32LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (2): TransposeLast()
        )
        (3): GELU()
      )
      (5): Sequential(
        (0): Conv1d(512, 512, kernel_size=(2,), stride=(2,), bias=False)
        (1): Dropout(p=0.0, inplace=False)
        (2): Sequential(
          (0): TransposeLast()
          (1): Fp32LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (2): TransposeLast()
        )
        (3): GELU()
      )
      (6): Sequential(
        (0): Conv1d(512, 512, kernel_size=(2,), stride=(2,), bias=False)
        (1): Dropout(p=0.0, inplace=False)
        (2): Sequential(
          (0): TransposeLast()
          (1): Fp32LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (2): TransposeLast()
        )
        (3): GELU()
      )
    )
  )
  (post_extract_proj): Linear(in_features=512, out_features=768, bias=True)
  (dropout_input): Dropout(p=0.0, inplace=False)
  (dropout_features): Dropout(p=0.0, inplace=False)
  (encoder): TransformerEncoder(
    (pos_conv): Sequential(
      (0): Sequential(
        (0): Conv1d(768, 768, kernel_size=(19,), stride=(1,), padding=(9,), groups=16)
        (1): SamePad()
        (2): TransposeLast()
        (3): LayerNorm((768,), eps=1e-05, elementwise_affine=False)
        (4): TransposeLast()
        (5): GELU()
      )
      (1): Sequential(
        (0): Conv1d(768, 768, kernel_size=(19,), stride=(1,), padding=(9,), groups=16)
        (1): SamePad()
        (2): TransposeLast()
        (3): LayerNorm((768,), eps=1e-05, elementwise_affine=False)
        (4): TransposeLast()
        (5): GELU()
      )
      (2): Sequential(
        (0): Conv1d(768, 768, kernel_size=(19,), stride=(1,), padding=(9,), groups=16)
        (1): SamePad()
        (2): TransposeLast()
        (3): LayerNorm((768,), eps=1e-05, elementwise_affine=False)
        (4): TransposeLast()
        (5): GELU()
      )
      (3): Sequential(
        (0): Conv1d(768, 768, kernel_size=(19,), stride=(1,), padding=(9,), groups=16)
        (1): SamePad()
        (2): TransposeLast()
        (3): LayerNorm((768,), eps=1e-05, elementwise_affine=False)
        (4): TransposeLast()
        (5): GELU()
      )
      (4): Sequential(
        (0): Conv1d(768, 768, kernel_size=(19,), stride=(1,), padding=(9,), groups=16)
        (1): SamePad()
        (2): TransposeLast()
        (3): LayerNorm((768,), eps=1e-05, elementwise_affine=False)
        (4): TransposeLast()
        (5): GELU()
      )
    )
    (layers): ModuleList(
      (0): TransformerSentenceEncoderLayer(
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.0, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (1): TransformerSentenceEncoderLayer(
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.0, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (2): TransformerSentenceEncoderLayer(
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.0, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (3): TransformerSentenceEncoderLayer(
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.0, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (4): TransformerSentenceEncoderLayer(
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.0, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (5): TransformerSentenceEncoderLayer(
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.0, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (6): TransformerSentenceEncoderLayer(
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.0, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (7): TransformerSentenceEncoderLayer(
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.0, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (8): TransformerSentenceEncoderLayer(
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.0, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (9): TransformerSentenceEncoderLayer(
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.0, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (10): TransformerSentenceEncoderLayer(
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.0, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (11): TransformerSentenceEncoderLayer(
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.0, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
    )
    (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  (final_proj): Linear(in_features=768, out_features=768, bias=True)
)
[2022-03-13 06:33:53,553][fairseq_cli.train][INFO] - task: AudioPretrainingTask
[2022-03-13 06:33:53,553][fairseq_cli.train][INFO] - model: Data2VecAudioModel
[2022-03-13 06:33:53,553][fairseq_cli.train][INFO] - criterion: ModelCriterion
[2022-03-13 06:33:53,555][fairseq_cli.train][INFO] - num. shared model params: 93,754,880 (num. trained: 93,754,880)
[2022-03-13 06:33:53,556][fairseq_cli.train][INFO] - num. expert model params: 0 (num. trained: 0)

"""

def cli_main():
    try:
        from hydra._internal.utils import get_args

        cfg_name = get_args().config_name or "config"   # cfg_name : base_librispeech
        import ipdb; ipdb.set_trace()
    except:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "config"
    
    hydra_init(cfg_name)
    hydra_main()


if __name__ == "__main__":
    cli_main()
