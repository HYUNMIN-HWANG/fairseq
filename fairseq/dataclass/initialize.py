# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import logging
from hydra.core.config_store import ConfigStore
from fairseq.dataclass.configs import FairseqConfig
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


def hydra_init(cfg_name="config") -> None:
    # import ipdb; ipdb.set_trace()
    cs = ConfigStore.instance()
    cs.store(name=f"{cfg_name}", node=FairseqConfig)

    for k in FairseqConfig.__dataclass_fields__:
        v = FairseqConfig.__dataclass_fields__[k].default # default 값으로 저장함
        try:
            cs.store(name=k, node=v)
        except BaseException:
            logger.error(f"{k} - {v}")
            raise


def add_defaults(cfg: DictConfig) -> None:
    # import ipdb; ipdb.set_trace()
    """This function adds default values that are stored in dataclasses that hydra doesn't know about"""

    from fairseq.registry import REGISTRIES
    from fairseq.tasks import TASK_DATACLASS_REGISTRY
    from fairseq.models import ARCH_MODEL_NAME_REGISTRY, MODEL_DATACLASS_REGISTRY
    from fairseq.dataclass.utils import merge_with_parent
    from typing import Any

    OmegaConf.set_struct(cfg, False)

    for k, v in FairseqConfig.__dataclass_fields__.items():
        field_cfg = cfg.get(k)
        if field_cfg is not None and v.type == Any: 
            dc = None

            if isinstance(field_cfg, str):
                field_cfg = DictConfig({"_name": field_cfg})
                field_cfg.__dict__["_parent"] = field_cfg.__dict__["_parent"]

            name = getattr(field_cfg, "_name", None)

            if k == "task":
                dc = TASK_DATACLASS_REGISTRY.get(name)
            elif k == "model":
                name = ARCH_MODEL_NAME_REGISTRY.get(name, name)
                dc = MODEL_DATACLASS_REGISTRY.get(name)
            elif k in REGISTRIES:
                dc = REGISTRIES[k]["dataclass_registry"].get(name)

            if dc is not None:
                cfg[k] = merge_with_parent(dc, field_cfg)


    # k : model
    ## field_cfg -> {'_name': 'data2vec_audio', 'extractor_mode': 'layer_norm', 'encoder_layerdrop': 0.05, 'dropout_input': 0.0, 'dropout_features': 0.0, 'feature_grad_mult': 1.0, 'encoder_embed_dim': 768, 'mask_prob': 0.65, 'mask_length': 10, 'loss_beta': 0, 'loss_scale': None, 'instance_norm_target_layer': True, 'average_top_k_layers': 8, 'pos_conv_depth': 5, 'conv_pos': 95, 'ema_decay': 0.999, 'ema_end_decay': 0.9999, 'ema_anneal_end_step': 30000, 'ema_transformer_only': True, 'ema_layers_only': True, 'require_same_masks': True, 'mask_dropout': 0}
    
    # k : task
    ##  field_cfg -> {'_name': 'audio_pretraining', 'data': '/path/to/manifests', 'max_sample_size': 320000, 'min_sample_size': 32000, 'normalize': True}
    ### merge_with_parent -> merged_cfg -> {'_name': 'audio_pretraining', 'data': '/path/to/manifests', 'labels': None, 'binarized_dataset': False, 'sample_rate': 16000, 'normalize': True, 'enable_padding': False, 'max_sample_size': 320000, 'min_sample_size': 32000, 'num_batch_buckets': 0, 'precompute_mask_indices': False, 'inferred_w2v_config': None, 'tpu': '${common.tpu}', 'text_compression_level': none}
    
    # k : criterion
    ## field_cfg -> {'_name': 'model', 'log_keys': ['ema_decay', 'target_var', 'pred_var']}
    ### merge_with_parent -> merged_cfg ->{'_name': 'model', 'loss_weights': {}, 'log_keys': ['ema_decay', 'target_var', 'pred_var']}

    # k : optimizer
    ## field_cfg -> {'_name': 'adam', 'adam_betas': '(0.9,0.98)', 'adam_eps': 1e-06, 'weight_decay': 0.01}
    ### merge_with_parent -> merged_cfg -> {'_name': 'adam', 'adam_betas': '(0.9,0.98)', 'adam_eps': 1e-06, 'weight_decay': 0.01, 'use_old_adam': False, 'fp16_adam_stats': False, 'tpu': '${common.tpu}', 'lr': '${optimization.lr}'}

    # k : lr_scheduler
    ## field_cfg -> {'_name': 'tri_stage', 'phase_ratio': [0.03, 0.9, 0.07]}
    ### merge_with_parent -> merged_cfg ->{'_name': 'tri_stage', 'warmup_steps': 0, 'hold_steps': 0, 'decay_steps': 0, 'phase_ratio': [0.03, 0.9, 0.07], 'init_lr_scale': 0.01, 'final_lr_scale': 0.01, 'max_update': '${optimization.max_update}', 'lr': '${optimization.lr}'}
    