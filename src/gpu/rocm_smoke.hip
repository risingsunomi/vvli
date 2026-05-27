// SPDX-License-Identifier: MPL-2.0

extern "C" __global__ void vvli_rocm_add(
    const float *a,
    const float *b,
    float *out,
    unsigned int n
) {
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) out[id] = a[id] + b[id];
}
