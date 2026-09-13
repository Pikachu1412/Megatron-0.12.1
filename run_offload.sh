#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONWARNINGS=ignore
export PYTHONPATH="/home/shensy/project/MindSpeed-0.12.1":$PYTHONPATH
export NCCL_DEBUG=WARN
# NPU env lacks triton, disable torch.compile/dynamo
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

#==============================================================================
# 激活值卸载最优配置（MBS=2 必须 offload 场景实测）
#
# 策略：LAST_LAYER_NO_OFFLOAD —— 最后一层 activation 保留 GPU，前 N-1 层卸载；
#       反向计算第 N 层时预取第 N-1 层，给 H2D 传输预留整层 backward 时间窗口。
#       实测 3273 ms / iter, 20.3 TFLOPs/GPU（基线纯 offload 3920 ms / 17.0 TFLOPs，
#       提升约 +19.7%）。
#
# 互斥约束：LAST_LAYER_NO_OFFLOAD=1 时 H2D_PREFETCH / CROSS_LAYER_PREFETCH 必须为 0。
#==============================================================================
export MEGATRON_LAST_LAYER_NO_OFFLOAD=1
export MEGATRON_H2D_PREFETCH=0
export MEGATRON_CROSS_LAYER_PREFETCH=0
GPUS_PER_NODE=4
WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6000
MODEL_TYPE="qwen3_30b_a3b"
# MODEL_TYPE="qwen2_5_32b"
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=4
VIRTUAL_PIPELINE_PARALLEL_SIZE=1  # VPP: 虚拟流水线并行，需要能整除 num_layers/PP
CONTEXT_PARALLEL_SIZE=1
EXPERT_TENSOR_PARALLEL_SIZE=1
EXPERT_MODEL_PARALLEL_SIZE=1
# EP dispatcher 类型: hybridep / deepep / alltoall
EP_DISPATCHER_TYPE="alltoall"
BATCH_SEQ_ARGS=(
    # --no-check-for-nan-in-loss-and-grad
    --micro-batch-size ${MBS:-1}
    --global-batch-size ${GBS:-16}
    --seq-length ${SEQ_LEN:-2048}
    --train-iters ${TRAIN_ITERS:-5}
)
# 是否启用 FP8 训练
USE_FP8=false
# FP8 配置选项: e4m3 / hybrid
FP8_FORMAT="hybrid"
# FP8 Recipe: delayed / tensorwise / blockwise / mxfp8
FP8_RECIPE="tensorwise"
# 是否将首尾层保持为 BF16（提高训练稳定性）
FP8_FIRST_LAST_BF16=true
# 首尾保持 BF16 的层数
FP8_NUM_LAYERS_BF16=2
# 是否只对 MoE 层启用 FP8（Dense 层保持 BF16）
FP8_MOE_ONLY=false

# 激活值卸载配置（两种方式互斥，只能选择一种）
# 卸载类型: none / fine_grained / cpu_offloading
#   - none: 不启用激活值卸载
#   - fine_grained: 细粒度激活值卸载（按模块级别卸载到 CPU）
#   - cpu_offloading: CPU 卸载（按层数卸载到 CPU，使用 Transformer Engine）
ACTIVATION_OFFLOAD_TYPE="${ACTIVATION_OFFLOAD_TYPE:-fine_grained}"
# 细粒度卸载的模块列表（仅当 ACTIVATION_OFFLOAD_TYPE="fine_grained" 时生效）
# 可选模块: attn_norm qkv_linear core_attn attn_proj mlp_norm expert_fc1 moe_act
OFFLOAD_MODULES="attn_norm qkv_linear core_attn attn_proj mlp_norm expert_fc1 moe_act"
# CPU 卸载的 Transformer 层数（仅当 ACTIVATION_OFFLOAD_TYPE="cpu_offloading" 时生效）
CPU_OFFLOADING_NUM_LAYERS=8

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

PROFILE_ARGS=(
    # --profile
    # --use-pytorch-profiler
    # --profile-step-start 5
    # --profile-step-end 7
    # --profile-ranks 0 1 2 3
)
export WANDB_API_KEY="c7145b22dabbc9bfd66697ffdeb74939cd797c84"
export WANDB_MODE=online
WANDB_ARGS=(
    # --wandb-project Offload-910b-a3b
    # --wandb-exp-name offload-no
)
#==============================================================================
# 模型选择: qwen3_30b_a3b / qwen2_5_32b
#==============================================================================
SAVE_PATH="./save"
# 模型 Checkpoint 路径
MCORE_MODEL_PATH_QWEN3_30B_A3B="/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3-30B-A3B-Base-to-mcore"
MCORE_MODEL_PATH_QWEN2_5_32B="/mnt/tidal-alsh01/dataset/redone/data/lly/Pai-Megatron-Patch/WORKDIR/ckpt/ckpt-32b-dllm-v1-to-mcore-tp8pp1"

# Tokenizer 路径
TOKENIZER_PATH_QWEN3_30B_A3B="/home/shensy/models/Qwen3-30B-A3B"
TOKENIZER_PATH_QWEN2_5_32B="/data0/models/Qwen3-30B-A3B"

# Qwen3-30B-A3B (MoE 模型)
MODEL_ARCH_QWEN3_30B_A3B=(
    --num-layers 16
    --hidden-size 2048
    --ffn-hidden-size 6144
    --num-attention-heads 32
    --num-query-groups 4
    --kv-channels 128
    --max-position-embeddings 40960
    # --padded-vocab-size 151936
    --swiglu
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --disable-bias-linear
    --qk-layernorm
    --group-query-attention
    --untie-embeddings-and-output-weights
    # MoE 专家配置
    --num-experts 128
    --moe-ffn-hidden-size 768
    --moe-router-topk 8
    --moe-layer-freq '([1]*16)'
    --moe-grouped-gemm
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 0.001
)

# Qwen2.5-32B (Dense 模型)
MODEL_ARCH_QWEN2_5_32B=(
    --num-layers 64
    --hidden-size 5120
    --ffn-hidden-size 27648
    --num-attention-heads 40
    --num-query-groups 8
    --kv-channels 128
    --max-position-embeddings 131072
    --padded-vocab-size 152064
    --swiglu
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --disable-bias-linear
    --group-query-attention
    --untie-embeddings-and-output-weights
)

# 根据 MODEL_TYPE 选择模型架构、Checkpoint 路径和 Tokenizer 路径
case $MODEL_TYPE in
    "qwen3_30b_a3b")
        MODEL_ARCH_ARGS=("${MODEL_ARCH_QWEN3_30B_A3B[@]}")
        export MCORE_MODEL_PATH=$MCORE_MODEL_PATH_QWEN3_30B_A3B
        TOKENIZER_PATH=$TOKENIZER_PATH_QWEN3_30B_A3B
        ;;
    "qwen2_5_32b")
        MODEL_ARCH_ARGS=("${MODEL_ARCH_QWEN2_5_32B[@]}")
        export MCORE_MODEL_PATH=$MCORE_MODEL_PATH_QWEN2_5_32B
        TOKENIZER_PATH=$TOKENIZER_PATH_QWEN2_5_32B
        ;;
    *)
        echo "Unknown MODEL_TYPE: $MODEL_TYPE"
        exit 1
        ;;
esac

POSITION_ENCODING_ARGS=(
    --use-rotary-position-embeddings
    --position-embedding-type rope
    --rotary-base 1000000
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size $TENSOR_PARALLEL_SIZE
    --pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE
    --context-parallel-size $CONTEXT_PARALLEL_SIZE
    --expert-tensor-parallel-size $EXPERT_TENSOR_PARALLEL_SIZE
    --expert-model-parallel-size $EXPERT_MODEL_PARALLEL_SIZE
    --sequence-parallel
    --use-distributed-optimizer
)
# 仅当 VPP > 1 时启用虚拟流水线并行（interleaved schedule）
if [ "$VIRTUAL_PIPELINE_PARALLEL_SIZE" -gt 1 ]; then
    PARALLEL_ARGS+=(--num-virtual-stages-per-pipeline-rank $VIRTUAL_PIPELINE_PARALLEL_SIZE)
fi

case $EP_DISPATCHER_TYPE in
    "hybridep")
        EP_OPTI_ARGS=(
            --moe-router-dtype fp32
            --moe-token-dispatcher-type flex
            --moe-flex-dispatcher-backend hybridep
        )
        if [ $EXPERT_MODEL_PARALLEL_SIZE -le 8 ]; then
            export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$EXPERT_MODEL_PARALLEL_SIZE
        else
            export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8
        fi
        echo "HybridEP enabled: NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"
        ;;
    "deepep")
        EP_OPTI_ARGS=(
            --moe-router-dtype fp32
            --moe-token-dispatcher-type flex
            --moe-flex-dispatcher-backend deepep
        )
        echo "DeepEP enabled"
        ;;
    "alltoall")
        EP_OPTI_ARGS=(
            --moe-token-dispatcher-type alltoall
        )
        echo "AllToAll dispatcher enabled"
        ;;
    *)
        echo "Unknown EP_DISPATCHER_TYPE: $EP_DISPATCHER_TYPE"
        exit 1
        ;;
esac

TRAINING_ARGS=(
    --lr 1e-6
    --min-lr 1e-6
    --lr-decay-style cosine
    --lr-warmup-iters 1
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --init-method-std 0.008
    --attention-dropout 0.0
    --hidden-dropout 0.0
)
DATA_ARGS=(
    --mock-data
    --num-workers 0
    --no-create-attention-mask-in-dataloader
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_PATH
)
CHECKPOINT_ARGS=(
    # --save $SAVE_PATH
    # --load $MCORE_MODEL_PATH
    --no-save-optim
    --no-load-optim
)
LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --eval-interval 10000
    --eval-iters 0
    --tensorboard-queue-size 1
    --tensorboard-dir ./tb
    --log-timers-to-tensorboard
    --log-validation-ppl-to-tensorboard
)

# FP8 训练参数配置
if [ "$USE_FP8" = true ]; then
    FP8_ARGS=(
        --fp8-format $FP8_FORMAT
        --fp8-recipe $FP8_RECIPE
        --fp8-amax-history-len 1024
        --fp8-amax-compute-algo max
    )
    if [ "$FP8_FIRST_LAST_BF16" = true ]; then
        FP8_ARGS+=(
            --first-last-layers-bf16
            --num-layers-at-start-in-bf16 $FP8_NUM_LAYERS_BF16
            --num-layers-at-end-in-bf16 $FP8_NUM_LAYERS_BF16
        )
    fi
    if [ "$FP8_MOE_ONLY" = true ]; then
        FP8_ARGS+=(--fp8-moe-only)
        echo "FP8 training enabled (MoE only): format=$FP8_FORMAT, recipe=$FP8_RECIPE"
    else
        echo "FP8 training enabled: format=$FP8_FORMAT, recipe=$FP8_RECIPE"
    fi
else
    FP8_ARGS=()
    echo "FP8 training disabled, using BF16"
fi

OPTIMIZATION_ARGS=(
    --bf16
    --use-mcore-models
    --transformer-impl transformer_engine
    --attention-backend flash
    --use-flash-attn
    --no-gradient-accumulation-fusion
)

# 激活值卸载参数配置（两种方式互斥）
case $ACTIVATION_OFFLOAD_TYPE in
    "fine_grained")
        ACTIVATION_OFFLOAD_ARGS=(
            --fine-grained-activation-offloading
            --offload-modules $OFFLOAD_MODULES
        )
        echo "Fine-grained activation offloading enabled: modules=$OFFLOAD_MODULES"
        ;;
    "cpu_offloading")
        ACTIVATION_OFFLOAD_ARGS=(
            --cpu-offloading-num-layers $CPU_OFFLOADING_NUM_LAYERS
        )
        echo "CPU activation offloading enabled: num_layers=$CPU_OFFLOADING_NUM_LAYERS"
        ;;
    "none")
        ACTIVATION_OFFLOAD_ARGS=()
        echo "Activation offloading disabled"
        ;;
    *)
        echo "Unknown ACTIVATION_OFFLOAD_TYPE: $ACTIVATION_OFFLOAD_TYPE"
        echo "Valid options: none, fine_grained, cpu_offloading"
        exit 1
        ;;
esac

RECOMPUTE_ARGS=(
    # --recompute-granularity full
    # --recompute-method block
    # --recompute-num-layers 28
)

set -x
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    "${PROFILE_ARGS[@]}" \
    "${MODEL_ARCH_ARGS[@]}" \
    "${POSITION_ENCODING_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${EP_OPTI_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${BATCH_SEQ_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "${OPTIMIZATION_ARGS[@]}" \
    "${FP8_ARGS[@]}" \
    "${ACTIVATION_OFFLOAD_ARGS[@]}" \
    "${RECOMPUTE_ARGS[@]}" \
    "${WANDB_ARGS[@]}" 2>&1 | tee ${LOG_FILE:-0.log}
set +x
