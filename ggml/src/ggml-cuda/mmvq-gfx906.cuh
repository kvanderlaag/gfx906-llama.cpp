#pragma once

// Warp-cooperative GEMV kernels for gfx906 (Vega 20 / MI50).
//
// A half-warp (32 lanes) cooperates on a single row: each lane processes one 32-value
// block, so the weight reads are contiguous and fully coalesced. This helps small
// matrices, where the generic per-thread K loop leaves the memory system underutilized.
// Only used for ncols_dst == 1 without fusion and ncols_x <= 1024, see mmvq.cu.

#if defined(GGML_USE_HIP)

__launch_bounds__(64, 1)
static __global__ void gfx906_mul_mat_vec_q4_0_warp_coop(
        const void * __restrict__ vx, const void * __restrict__ vy,
        const int32_t * __restrict__ ids, float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y,
        const uint32_t stride_row_x, const uint3 channel_ratio,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y,
        const uint32_t stride_sample_dst, const uint32_t nrows_x) {

    const int lane_id   = threadIdx.x;
    const int half_lane = lane_id % 32;
    const int row       = blockIdx.x*2 + lane_id / 32;

    if (row >= (int) nrows_x) {
        return;
    }

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t channel_x   = ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y   = ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row*stride_row_x;

    const block_q4_0 * x = (const block_q4_0 *) vx + kbx_offset;
    const block_q8_1 * y = (const block_q8_1 *) vy + sample_y*stride_sample_y + channel_y*stride_channel_y;

    const int blocks_per_row = ncols_x / QK4_0;

    float sumf = 0.0f;

    for (int ib = half_lane; ib < blocks_per_row; ib += 32) {
        const block_q4_0 * bq4 = x + ib;
        const block_q8_1 * bq8 = y + ib;

        const int v0 = get_int_b2(bq4->qs, 0);
        const int v1 = get_int_b2(bq4->qs, 1);
        const int v2 = get_int_b2(bq4->qs, 2);
        const int v3 = get_int_b2(bq4->qs, 3);

        const int u0 = get_int_b4(bq8->qs, 0);
        const int u1 = get_int_b4(bq8->qs, 1);
        const int u2 = get_int_b4(bq8->qs, 2);
        const int u3 = get_int_b4(bq8->qs, 3);
        const int u4 = get_int_b4(bq8->qs, 4);
        const int u5 = get_int_b4(bq8->qs, 5);
        const int u6 = get_int_b4(bq8->qs, 6);
        const int u7 = get_int_b4(bq8->qs, 7);

        int sumi = 0;
        sumi = ggml_cuda_dp4a((v0 >> 0) & 0x0F0F0F0F, u0, sumi);
        sumi = ggml_cuda_dp4a((v0 >> 4) & 0x0F0F0F0F, u4, sumi);
        sumi = ggml_cuda_dp4a((v1 >> 0) & 0x0F0F0F0F, u1, sumi);
        sumi = ggml_cuda_dp4a((v1 >> 4) & 0x0F0F0F0F, u5, sumi);
        sumi = ggml_cuda_dp4a((v2 >> 0) & 0x0F0F0F0F, u2, sumi);
        sumi = ggml_cuda_dp4a((v2 >> 4) & 0x0F0F0F0F, u6, sumi);
        sumi = ggml_cuda_dp4a((v3 >> 0) & 0x0F0F0F0F, u3, sumi);
        sumi = ggml_cuda_dp4a((v3 >> 4) & 0x0F0F0F0F, u7, sumi);

        // Q4_0 is stored as unsigned nibbles, the -8 offset is folded into the sum of the Q8 values.
        const float2 ds8 = __half22float2(bq8->ds);
        sumf += __half2float(bq4->d) * (sumi*ds8.x - 8.0f*ds8.y);
    }

    sumf = warp_reduce_sum<32>(sumf);

    if (half_lane == 0) {
        dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row] = sumf;
    }
}

__launch_bounds__(64, 1)
static __global__ void gfx906_mul_mat_vec_q4_1_warp_coop(
        const void * __restrict__ vx, const void * __restrict__ vy,
        const int32_t * __restrict__ ids, float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y,
        const uint32_t stride_row_x, const uint3 channel_ratio,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y,
        const uint32_t stride_sample_dst, const uint32_t nrows_x) {

    const int lane_id   = threadIdx.x;
    const int half_lane = lane_id % 32;
    const int row       = blockIdx.x*2 + lane_id / 32;

    if (row >= (int) nrows_x) {
        return;
    }

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t channel_x   = ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y   = ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row*stride_row_x;

    const block_q4_1 * x = (const block_q4_1 *) vx + kbx_offset;
    const block_q8_1 * y = (const block_q8_1 *) vy + sample_y*stride_sample_y + channel_y*stride_channel_y;

    const int blocks_per_row = ncols_x / QK4_1;

    float sumf = 0.0f;

    for (int ib = half_lane; ib < blocks_per_row; ib += 32) {
        const block_q4_1 * bq4 = x + ib;
        const block_q8_1 * bq8 = y + ib;

        const int v0 = get_int_b4(bq4->qs, 0);
        const int v1 = get_int_b4(bq4->qs, 1);
        const int v2 = get_int_b4(bq4->qs, 2);
        const int v3 = get_int_b4(bq4->qs, 3);

        const int u0 = get_int_b4(bq8->qs, 0);
        const int u1 = get_int_b4(bq8->qs, 1);
        const int u2 = get_int_b4(bq8->qs, 2);
        const int u3 = get_int_b4(bq8->qs, 3);
        const int u4 = get_int_b4(bq8->qs, 4);
        const int u5 = get_int_b4(bq8->qs, 5);
        const int u6 = get_int_b4(bq8->qs, 6);
        const int u7 = get_int_b4(bq8->qs, 7);

        int sumi = 0;
        sumi = ggml_cuda_dp4a((v0 >> 0) & 0x0F0F0F0F, u0, sumi);
        sumi = ggml_cuda_dp4a((v0 >> 4) & 0x0F0F0F0F, u4, sumi);
        sumi = ggml_cuda_dp4a((v1 >> 0) & 0x0F0F0F0F, u1, sumi);
        sumi = ggml_cuda_dp4a((v1 >> 4) & 0x0F0F0F0F, u5, sumi);
        sumi = ggml_cuda_dp4a((v2 >> 0) & 0x0F0F0F0F, u2, sumi);
        sumi = ggml_cuda_dp4a((v2 >> 4) & 0x0F0F0F0F, u6, sumi);
        sumi = ggml_cuda_dp4a((v3 >> 0) & 0x0F0F0F0F, u3, sumi);
        sumi = ggml_cuda_dp4a((v3 >> 4) & 0x0F0F0F0F, u7, sumi);

        const float2 dm4 = __half22float2(bq4->dm);
        const float2 ds8 = __half22float2(bq8->ds);
        sumf += sumi*dm4.x*ds8.x + dm4.y*ds8.y;
    }

    sumf = warp_reduce_sum<32>(sumf);

    if (half_lane == 0) {
        dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row] = sumf;
    }
}

__launch_bounds__(64, 1)
static __global__ void gfx906_mul_mat_vec_q8_0_warp_coop(
        const void * __restrict__ vx, const void * __restrict__ vy,
        const int32_t * __restrict__ ids, float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y,
        const uint32_t stride_row_x, const uint3 channel_ratio,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y,
        const uint32_t stride_sample_dst, const uint32_t nrows_x) {

    const int lane_id   = threadIdx.x;
    const int half_lane = lane_id % 32;
    const int row       = blockIdx.x*2 + lane_id / 32;

    if (row >= (int) nrows_x) {
        return;
    }

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t channel_x   = ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y   = ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row*stride_row_x;

    const block_q8_0 * x = (const block_q8_0 *) vx + kbx_offset;
    const block_q8_1 * y = (const block_q8_1 *) vy + sample_y*stride_sample_y + channel_y*stride_channel_y;

    const int blocks_per_row = ncols_x / QK8_0;

    float sumf = 0.0f;

    for (int ib = half_lane; ib < blocks_per_row; ib += 32) {
        const block_q8_0 * bq8_0 = x + ib;
        const block_q8_1 * bq8_1 = y + ib;

        int sumi = 0;
#pragma unroll
        for (int i = 0; i < QK8_0/4; ++i) {
            sumi = ggml_cuda_dp4a(get_int_b2(bq8_0->qs, i), get_int_b4(bq8_1->qs, i), sumi);
        }

        sumf += __half2float(bq8_0->d) * __low2float(bq8_1->ds) * (float) sumi;
    }

    sumf = warp_reduce_sum<32>(sumf);

    if (half_lane == 0) {
        dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row] = sumf;
    }
}

static void gfx906_launch_mul_mat_vec_warp_coop(
        const ggml_type type_x, const void * vx, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x,
        const uint3 channel_ratio, const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t nrows_x, const uint32_t nchannels_dst, const uint32_t nsamples_dst,
        cudaStream_t stream) {

    // 2 rows per block, 64 threads per block (one half-warp per row).
    const dim3 block_dims(64, 1, 1);
    const dim3 block_nums((nrows_x + 1) / 2, nchannels_dst, nsamples_dst);

    switch (type_x) {
        case GGML_TYPE_Q4_0:
            gfx906_mul_mat_vec_q4_0_warp_coop<<<block_nums, block_dims, 0, stream>>>(
                vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, channel_ratio,
                stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio,
                stride_sample_x, stride_sample_y, stride_sample_dst, nrows_x);
            break;
        case GGML_TYPE_Q4_1:
            gfx906_mul_mat_vec_q4_1_warp_coop<<<block_nums, block_dims, 0, stream>>>(
                vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, channel_ratio,
                stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio,
                stride_sample_x, stride_sample_y, stride_sample_dst, nrows_x);
            break;
        case GGML_TYPE_Q8_0:
            gfx906_mul_mat_vec_q8_0_warp_coop<<<block_nums, block_dims, 0, stream>>>(
                vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, channel_ratio,
                stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio,
                stride_sample_x, stride_sample_y, stride_sample_dst, nrows_x);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#endif // GGML_USE_HIP
