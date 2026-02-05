#include <vector>
#include <cuda_fp16.h>
#include <algorithm>  // 仅用于CPU端min计算（元信息准备，符合要求）

#include "../tester/utils.h"

// -------------------------- 通用 CUDA 错误检查宏（调试必备） --------------------------
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at file %s line %d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// ======================================================================================
// 第一题：CUDA 实现 matrix trace（矩阵迹）
// 核心逻辑：CPU 端做内存准备/元信息计算，GPU 核函数并行计算对角线元素和
// 支持类型：int / float
// ======================================================================================
/**
 * @brief GPU 核函数：计算矩阵迹（对角线元素求和）
 * @tparam T 数据类型（int/float）
 * @param d_input 设备端扁平化矩阵（行优先存储，size=rows*cols）
 * @param d_sum 设备端单个值地址：存储对角线元素和
 * @param rows 矩阵行数
 * @param cols 矩阵列数
 * @param diag_len 对角线长度（min(rows, cols)，CPU 端计算后传入，减少GPU计算）
 */
template <typename T>
__global__ void traceKernel(const T* d_input, T* d_sum, size_t rows, size_t cols, size_t diag_len) {
    // 每个线程处理一个对角线元素，全局线程索引对应对角线索引i
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < diag_len) {
        // 行优先存储：第i行第i列元素的索引 = i * cols + i
        T val = d_input[i * cols + i];
        // 原子加：保证多线程求和无竞争（diag_len可能大于线程数，原子操作安全）
        atomicAdd(d_sum, val);
    }
}

/**
 * @brief 主机端接口：trace 函数（对外暴露，符合作业函数签名）
 * @tparam T 数据类型（int/float）
 * @param h_input 主机端扁平化矩阵
 * @param rows 矩阵行数
 * @param cols 矩阵列数
 * @return 矩阵迹（对角线元素和）
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    // 边界条件：空矩阵/零行/零列，直接返回0
    if (h_input.empty() || rows == 0 || cols == 0) {
        return T(0);
    }

    // -------------------------- CPU 端元信息计算（符合要求） --------------------------
    size_t diag_len = std::min(rows, cols);  // 对角线长度，CPU端计算后传入GPU
    size_t data_size = rows * cols * sizeof(T);
    T h_trace = T(0);  // 主机端存储最终结果

    // -------------------------- CPU 端设备内存申请 --------------------------
    T *d_input = nullptr, *d_sum = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, data_size));  // 设备端矩阵内存
    // 申请单个值的设备内存（存储对角线和），并初始化为0
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_sum, sizeof(T)));
    CHECK_CUDA_ERROR(cudaMemset(d_sum, 0, sizeof(T)));

    // -------------------------- 主机 → 设备 数据拷贝 --------------------------
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), data_size, cudaMemcpyHostToDevice));

    // -------------------------- 配置GPU核函数启动参数 --------------------------
    const int blockSize = 256;  // CUDA 最佳实践：blockSize设为32的倍数（256/512/1024）
    // gridSize向上取整：保证线程数≥diag_len，覆盖所有对角线元素
    int gridSize = (diag_len + blockSize - 1) / blockSize;

    // -------------------------- 启动GPU核函数计算 --------------------------
    traceKernel<T><<<gridSize, blockSize>>>(d_input, d_sum, rows, cols, diag_len);
    // 检查核函数启动错误（核函数本身错误需用cudaDeviceSynchronize检测）
    CHECK_CUDA_ERROR(cudaGetLastError());
    // 等待GPU计算完成，保证后续数据拷贝的完整性
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // -------------------------- 设备 → 主机 结果拷贝 --------------------------
    CHECK_CUDA_ERROR(cudaMemcpy(&h_trace, d_sum, sizeof(T), cudaMemcpyDeviceToHost));

    // -------------------------- 释放设备内存（避免内存泄漏） --------------------------
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_sum));

    return h_trace;
}

// ======================================================================================
// 第二题：CUDA 实现 Flash Attention（闪注意力）
// 核心逻辑：CPU 端做内存准备/维度校验，GPU 核函数完成注意力全流程计算
// 支持特性：Grouped Query Attention（GQA）、因果掩码（causal masking）
// 支持类型：float / half（FP16）
// 核心步骤：Q*K^T缩放 → 因果掩码 → Softmax → 乘V得到输出
// ======================================================================================
/**
 * @brief GPU 核函数：计算Flash Attention（单头维度，按query head并行）
 * @tparam T 数据类型（float/half）
 * @param d_q 设备端Query张量 [batch, tgt_seq, q_head, head_dim]
 * @param d_k 设备端Key张量 [batch, src_seq, kv_head, head_dim]
 * @param d_v 设备端Value张量 [batch, src_seq, kv_head, head_dim]
 * @param d_o 设备端Output张量 [batch, tgt_seq, q_head, head_dim]
 * @param batch_size 批次大小
 * @param tgt_seq_len 目标序列长度
 * @param src_seq_len 源序列长度
 * @param q_heads Query头数
 * @param kv_heads KV头数
 * @param head_dim 每个头的维度
 * @param heads_per_kv 每个KV头对应的Query头数（q_heads/kv_heads，CPU端校验）
 * @param is_causal 是否启用因果掩码（下三角掩码，t < s 时分数置为-inf）
 */
template <typename T>
__global__ void flashAttentionKernel(
    const T* d_q, const T* d_k, const T* d_v, T* d_o,
    int batch_size, int tgt_seq_len, int src_seq_len,
    int q_heads, int kv_heads, int head_dim,
    int heads_per_kv, bool is_causal
) {
    // 线程维度映射：每个线程块处理一个Query头，每个线程处理一个(tgt_seq, head_dim)组合
    // 全局线程索引：q_head_idx（Query头索引）→ 0~q_heads-1
    int q_head_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_head_idx >= q_heads) return;

    // 步骤1：计算当前Query头对应的KV头索引（GQA核心：Query头均分至KV头）
    int kv_head_idx = q_head_idx / heads_per_kv;

    // 预计算维度步长（减少GPU端重复计算，提升效率）
    const int q_step = q_heads * head_dim;       // Q中单个tgt_seq的步长
    const int k_step = kv_heads * head_dim;      // K中单个src_seq的步长
    const int v_step = kv_heads * head_dim;      // V中单个src_seq的步长
    const int o_step = q_heads * head_dim;       // O中单个tgt_seq的步长
    const int q_batch_step = tgt_seq_len * q_step;  // Q中单个batch的步长
    const int k_batch_step = src_seq_len * k_step;  // K中单个batch的步长
    const int v_batch_step = src_seq_len * v_step;  // V中单个batch的步长
    const int o_batch_step = tgt_seq_len * o_step;  // O中单个batch的步长

    // 步骤2：遍历批次和目标序列（每个Query头独立处理）
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < tgt_seq_len; t++) {
            // 定位当前(b, t, q_head_idx)的Query向量起始地址：d_q[batch_offset + t*q_step + q_head_idx*head_dim]
            const T* q_vec = d_q + b * q_batch_step + t * q_step + q_head_idx * head_dim;
            // 定位当前(b, kv_head_idx)的Key矩阵起始地址：d_k[batch_offset + kv_head_idx*head_dim]
            const T* k_mat = d_k + b * k_batch_step + kv_head_idx * head_dim;
            // 定位当前(b, kv_head_idx)的Value矩阵起始地址：d_v[batch_offset + kv_head_idx*head_dim]
            const T* v_mat = d_v + b * v_batch_step + kv_head_idx * head_dim;
            // 定位当前(b, t, q_head_idx)的Output向量起始地址：d_o[batch_offset + t*o_step + q_head_idx*head_dim]
            T* o_vec = d_o + b * o_batch_step + t * o_step + q_head_idx * head_dim;

            // 步骤3：计算 Q_t · K^T 注意力分数（长度=src_seq_len）
            T attn_scores[1024];  // 共享内存存储分数（假设src_seq_len≤1024，可根据需求调整）
            T scale = T(1.0f / sqrtf((float)head_dim));  // 缩放因子：1/√head_dim
            for (int s = 0; s < src_seq_len; s++) {
                // 因果掩码：t < s 时，分数置为极小值（近似-inf），避免看到未来token
                if (is_causal && s > t) {
                    attn_scores[s] = T(-1e9f);
                    continue;
                }
                // 计算Q_t与K_s的点积：Q_t · K_s^T
                T dot_prod = T(0);
                for (int d = 0; d < head_dim; d++) {
                    dot_prod += q_vec[d] * k_mat[s * k_step + d];
                }
                attn_scores[s] = dot_prod * scale;  // 缩放注意力分数
            }

            // 步骤4：数值稳定的Softmax（GPU端手动实现，无库函数）
            T max_val = attn_scores[0];
            for (int s = 1; s < src_seq_len; s++) {
                if (attn_scores[s] > max_val) max_val = attn_scores[s];
            }
            // 计算exp(x - max_val)，避免数值溢出
            T sum_exp = T(0);
            T attn_weights[1024];
            for (int s = 0; s < src_seq_len; s++) {
                attn_weights[s] = __expf((float)(attn_scores[s] - max_val));  // CUDA内置exp（基础指令，非库函数）
                sum_exp += attn_weights[s];
            }
            // 归一化得到注意力权重
            for (int s = 0; s < src_seq_len; s++) {
                attn_weights[s] /= sum_exp;
            }

            // 步骤5：计算 Output = attn_weights · V （注意力权重乘Value矩阵）
            for (int d = 0; d < head_dim; d++) {
                T out_val = T(0);
                for (int s = 0; s < src_seq_len; s++) {
                    out_val += attn_weights[s] * v_mat[s * v_step + d];
                }
                o_vec[d] = out_val;  // 写入输出向量
            }
        }
    }
}

/**
 * @brief 主机端接口：flashAttention 函数（对外暴露，符合作业函数签名）
 * @tparam T 数据类型（float/half）
 * @param h_q 主机端Query张量 [batch, tgt_seq, q_head, head_dim]
 * @param h_k 主机端Key张量 [batch, src_seq, kv_head, head_dim]
 * @param h_v 主机端Value张量 [batch, src_seq, kv_head, head_dim]
 * @param h_o 主机端Output张量（输出，函数内resize并赋值）
 * @param batch_size/ target_seq_len/ src_seq_len/ query_heads/ kv_heads/ head_dim 维度参数
 * @param is_causal 是否启用因果掩码
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    // -------------------------- CPU 端维度校验（元信息准备） --------------------------
    if (h_q.empty() || h_k.empty() || h_v.empty()) {
        fprintf(stderr, "Error: Input tensors are empty!\n");
        return;
    }
    if (query_heads % kv_heads != 0) {
        fprintf(stderr, "Error: query_heads must be divisible by kv_heads!\n");
        return;
    }
    int heads_per_kv = query_heads / kv_heads;  // 每个KV头对应的Query头数

    // -------------------------- CPU 端内存大小计算 & 输出张量resize --------------------------
    const int o_size = batch_size * target_seq_len * query_heads * head_dim;
    h_o.resize(o_size, T(0));  // 输出张量初始化大小并置0
    const size_t q_size = h_q.size() * sizeof(T);
    const size_t k_size = h_k.size() * sizeof(T);
    const size_t v_size = h_v.size() * sizeof(T);
    const size_t o_size_bytes = o_size * sizeof(T);

    // -------------------------- CPU 端设备内存申请 --------------------------
    T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_q, q_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_k, k_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_v, v_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_o, o_size_bytes));

    // -------------------------- 主机 → 设备 数据拷贝 --------------------------
    CHECK_CUDA_ERROR(cudaMemcpy(d_q, h_q.data(), q_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_k, h_k.data(), k_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_v, h_v.data(), v_size, cudaMemcpyHostToDevice));
    // 输出张量初始化为0（可选，主机端已resize置0，双重保障）
    CHECK_CUDA_ERROR(cudaMemset(d_o, 0, o_size_bytes));

    // -------------------------- 配置GPU核函数启动参数 --------------------------
    const int blockSize = 32;  // 按Query头并行，blockSize设32（32的倍数）
    int gridSize = (query_heads + blockSize - 1) / blockSize;  // 向上取整覆盖所有Query头

    // -------------------------- 启动GPU核函数计算Flash Attention --------------------------
    flashAttentionKernel<T><<<gridSize, blockSize>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        heads_per_kv, is_causal
    );
    CHECK_CUDA_ERROR(cudaGetLastError());  // 检查核函数启动错误
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());  // 等待GPU计算完成

    // -------------------------- 设备 → 主机 结果拷贝 --------------------------
    CHECK_CUDA_ERROR(cudaMemcpy(h_o.data(), d_o, o_size_bytes, cudaMemcpyDeviceToHost));

    // -------------------------- 释放设备内存（避免内存泄漏） --------------------------
    CHECK_CUDA_ERROR(cudaFree(d_q));
    CHECK_CUDA_ERROR(cudaFree(d_k));
    CHECK_CUDA_ERROR(cudaFree(d_v));
    CHECK_CUDA_ERROR(cudaFree(d_o));
}

// *********************************************************************
// 显式模板实例化（REQUIRED FOR LINKING WITH TESTER.O）
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);