// SPDX-License-Identifier: MPL-2.0

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

typedef struct vvli_metal_smoke_result {
    char device_name[128];
    unsigned int value_count;
    float max_abs_error;
} vvli_metal_smoke_result;

static void set_device_name(vvli_metal_smoke_result *result, id<MTLDevice> device) {
    memset(result->device_name, 0, sizeof(result->device_name));
    const char *name = device.name.UTF8String;
    if (name == NULL) name = "unknown";
    snprintf(result->device_name, sizeof(result->device_name), "%s", name);
}

int vvli_metal_smoke(vvli_metal_smoke_result *result) {
    if (result == NULL) return 100;
    memset(result, 0, sizeof(*result));

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) return 1;
        set_device_name(result, device);

        static NSString *source =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void vvli_add(device const float *a [[buffer(0)]],\n"
             "                     device const float *b [[buffer(1)]],\n"
             "                     device float *out [[buffer(2)]],\n"
             "                     uint id [[thread_position_in_grid]]) {\n"
             "    out[id] = a[id] + b[id];\n"
             "}\n";

        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (library == nil) return 2;

        id<MTLFunction> function = [library newFunctionWithName:@"vvli_add"];
        if (function == nil) return 3;

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) return 4;

        const NSUInteger count = 256;
        const NSUInteger byte_count = count * sizeof(float);
        id<MTLBuffer> a = [device newBufferWithLength:byte_count options:MTLResourceStorageModeShared];
        id<MTLBuffer> b = [device newBufferWithLength:byte_count options:MTLResourceStorageModeShared];
        id<MTLBuffer> out = [device newBufferWithLength:byte_count options:MTLResourceStorageModeShared];
        if (a == nil || b == nil || out == nil) return 5;

        float *a_values = (float *)a.contents;
        float *b_values = (float *)b.contents;
        float *out_values = (float *)out.contents;
        for (NSUInteger i = 0; i < count; i++) {
            a_values[i] = (float)i * 0.25f;
            b_values[i] = 1.5f - (float)i * 0.125f;
            out_values[i] = 0.0f;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) return 6;

        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) return 7;

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:a offset:0 atIndex:0];
        [encoder setBuffer:b offset:0 atIndex:1];
        [encoder setBuffer:out offset:0 atIndex:2];

        const NSUInteger width = pipeline.threadExecutionWidth > 0 ? pipeline.threadExecutionWidth : 1;
        MTLSize threads_per_group = MTLSizeMake(width, 1, 1);
        MTLSize threads = MTLSizeMake(count, 1, 1);
        [encoder dispatchThreads:threads threadsPerThreadgroup:threads_per_group];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return 8;

        float max_abs_error = 0.0f;
        for (NSUInteger i = 0; i < count; i++) {
            const float expected = a_values[i] + b_values[i];
            const float error_abs = fabsf(out_values[i] - expected);
            if (error_abs > max_abs_error) max_abs_error = error_abs;
        }

        result->value_count = (unsigned int)count;
        result->max_abs_error = max_abs_error;
        return max_abs_error <= 0.00001f ? 0 : 9;
    }
}
