#!/usr/bin/env bash
# from .infinity_patched import Infinity as InfinityPatched  记得注释
set -x
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
nproc_per_node=${NPROC_PER_NODE:-2}

nnodes=${NNODES:-1}
node_rank=${NODE_RANK:-0}
master_addr=${MASTER_ADDR:-127.0.0.1}
master_port=${MASTER_PORT:-12346}

echo "[Using GPUs: $CUDA_VISIBLE_DEVICES, count: ${nproc_per_node}]"
echo "[nproc_per_node: ${nproc_per_node}]"
echo "[nnodes: ${nnodes}]"
echo "[node_rank: ${node_rank}]"
echo "[master_addr: ${master_addr}]"
echo "[master_port: ${master_port}]"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

BED=${BED:-checkpoints}
LOCAL_OUT=${LOCAL_OUT:-local_output}
mkdir -p "$BED"
mkdir -p "$LOCAL_OUT"

export COMPILE_GAN=${COMPILE_GAN:-0}
export USE_TIMELINE_SDK=${USE_TIMELINE_SDK:-1}
export CUDA_TIMER_STREAM_KAFKA_CLUSTER=${CUDA_TIMER_STREAM_KAFKA_CLUSTER:-bmq_data_va}
export CUDA_TIMER_STREAM_KAFKA_TOPIC=${CUDA_TIMER_STREAM_KAFKA_TOPIC:-megatron_cuda_timer_tracing_original_v2}

NISP_WEIGHT=${NISP_WEIGHT:-0.03}
NISP_MODE=${NISP_MODE:-code} # shallow code
NISP_SCALE_MIN=${NISP_SCALE_MIN:-3}
NISP_SCALE_MAX=${NISP_SCALE_MAX:-11}
NISP_HEAD_TYPE=${NISP_HEAD_TYPE:-mlp}
NISP_HEAD_HIDDEN_DIM=${NISP_HEAD_HIDDEN_DIM:-512}
MODEL=${MODEL:-2bc8}
if [ -z "${VAE_TYPE:-}" ]; then
  if [ "${MODEL}" = "layer12c4" ]; then
    VAE_TYPE=16
  else
    VAE_TYPE=32
  fi
fi
if [ -z "${NISP_TARGET_LAYER:-}" ]; then
  if [ "${MODEL}" = "layer12c4" ]; then
    NISP_TARGET_LAYER=3
  else
    NISP_TARGET_LAYER=4
  fi
fi

exp_name=${EXP_NAME:-stage2_1024_${MODEL}_${VAE_TYPE}vae_nisp_${NISP_MODE}_L${NISP_TARGET_LAYER}_scale${NISP_SCALE_MIN}-${NISP_SCALE_MAX}_${NISP_HEAD_TYPE}${NISP_HEAD_HIDDEN_DIM}_w${NISP_WEIGHT}}
bed_path=${BED}/${exp_name}/
local_out_path=${LOCAL_OUT}/${exp_name}

proc_data_path=${PROC_DATA_PATH:-/datasets/pixelprose/embedding_mmap}
proc_res_list=${PROC_RES_LIST:-1024}
if [ "${VAE_TYPE}" = "16" ]; then
  VAE_CKPT=${VAE_CKPT:-/workspace/CKPT/Infinity/infinity_vae_d16.pth}
  RUSH_RESUME=${RUSH_RESUME:-/workspace/CKPT/Infinity/infinity_125M_256x256.pth}
else
  VAE_CKPT=${VAE_CKPT:-/workspace/CKPT/Infinity/infinity_vae_d32reg.pth}
  RUSH_RESUME=${RUSH_RESUME:-/workspace/CKPT/Infinity/infinity_2b_reg.pth}
fi

torchrun \
--nproc_per_node=${nproc_per_node} \
--nnodes=${nnodes} \
--node_rank=${node_rank} \
--master_addr=${master_addr} \
--master_port=${master_port} \
train_stage2_var_entropy_raw.py \
--ep=100 \
--opt=adamw \
--cum=3 \
--sche=lin0 \
--fp16=2 \
--ada=0.9_0.97 \
--tini=-1 \
--tclip=5 \
--flash=0 \
--alng=5e-06 \
--saln=1 \
--cos=1 \
--enable_checkpointing=full-block \
--local_out_path ${local_out_path} \
--task_type='t2i' \
--bed=${bed_path} \
--exp_name=${exp_name} \
--tblr=6e-5 \
--pn 1M \
--model=${MODEL} \
--lbs=${LBS:-10} \
--workers=1 \
--Ct5=2048 \
--vae_type ${VAE_TYPE} \
--vae_ckpt=${VAE_CKPT} \
--rush_resume=${RUSH_RESUME} \
--wp 0.00000001 \
--wpe=1 \
--dynamic_resolution_across_gpus 1 \
--reweight_loss_by_scale 1 \
--add_lvl_embeding_only_first_block 1 \
--rope2d_each_sa_layer 1 \
--rope2d_normalized_by_hw 2 \
--use_fsdp_model_ema 0 \
--always_training_scales 100 \
--use_bit_label 1 \
--zero=2 \
--save_model_iters_freq ${SAVE_MODEL_ITERS_FREQ:-5000} \
--log_freq=${LOG_FREQ:-50} \
--checkpoint_type='torch' \
--prefetch_factor=16 \
--noise_apply_strength 0.3 \
--noise_apply_layers 13 \
--apply_spatial_patchify 0 \
--use_flex_attn=True \
--pad=128 \
\
--enable_nisp 1 \
--nisp_mode ${NISP_MODE} \
--nisp_weight ${NISP_WEIGHT} \
--nisp_scale_min ${NISP_SCALE_MIN} \
--nisp_scale_max ${NISP_SCALE_MAX} \
--nisp_loss_type ${NISP_LOSS_TYPE:-cosine} \
--nisp_head_type ${NISP_HEAD_TYPE} \
--nisp_head_hidden_dim ${NISP_HEAD_HIDDEN_DIM} \
--nisp_head_dropout ${NISP_HEAD_DROPOUT:-0.0} \
--nisp_target_layer ${NISP_TARGET_LAYER} \
\
--online_t5=0 \
--use_streaming_dataset=0 \
--dataset_backend='proc_memmap' \
--proc_data_path=${proc_data_path} \
--proc_res_list=${proc_res_list} \
--proc_memmap_cache_size=2 \
\
--enable_student_entropy=0 \
--student_start_step=50000 \
--student_hidden_dim=256 \
--student_depth=4 \
--student_dropout=0.0 \
--student_lr=2e-4 \
--student_wd=1e-4 \
--student_grad_clip=1.0 \
--student_start_scale=1 \
--student_kd_ratio=1.0 \
--student_gt_ratio=1.0
