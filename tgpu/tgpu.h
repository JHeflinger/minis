#ifndef TGPU_H
#define TGPU_H

// Minimal, self contained Vulkan compute layer used by the Gaussian splat trainer.
// It owns its own VkInstance / VkDevice so the trainer can run headless, independent from the
// renderer's synchronisation, and can be tested on any Vulkan implementation (including software ones).

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stddef.h>

#define TGPU_MAX_BINDINGS 24
#define TGPU_PUSH_SIZE 128

typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize size;
} TGPUBuffer;

typedef struct {
    VkPipeline pipeline;
    VkShaderModule module;
} TGPUKernel;

typedef struct {
    VkInstance instance;
    VkPhysicalDevice gpu;
    VkDevice device;
    VkQueue queue;
    uint32_t family;
    VkCommandPool pool;
    VkCommandBuffer cmd;
    VkFence fence;
    VkDescriptorSetLayout set_layout;
    VkDescriptorPool desc_pool;
    VkDescriptorSet set;
    VkPipelineLayout layout;
    VkPhysicalDeviceMemoryProperties memprops;
    char gpu_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
    int atomic_float_add;       // device supports float atomic add on storage buffers
    uint32_t max_workgroup_invocations;
    int recording;
    TGPUBuffer staging;         // grown on demand
    void* staging_mapped;
} TGPU;

// Creates the context. `preferred` selects a physical device index (-1 = best available).
// `want_atomic_float` requests VK_EXT_shader_atomic_float if present. Returns 0 on success.
int tgpu_create(TGPU* g, int preferred, int want_atomic_float);
void tgpu_destroy(TGPU* g);

// device local buffer usable for storage, copies and indirect dispatch
int tgpu_buffer_create(TGPU* g, TGPUBuffer* b, VkDeviceSize size);
void tgpu_buffer_destroy(TGPU* g, TGPUBuffer* b);
void tgpu_bind(TGPU* g, uint32_t binding, const TGPUBuffer* b);

// loads a SPIR-V file and builds a compute pipeline over the shared layout
int tgpu_kernel_create(TGPU* g, TGPUKernel* k, const char* spv_path);
void tgpu_kernel_destroy(TGPU* g, TGPUKernel* k);

// immediate (blocking) host <-> device copies through the staging buffer
int tgpu_upload(TGPU* g, const TGPUBuffer* dst, VkDeviceSize offset, const void* data, VkDeviceSize size);
int tgpu_download(TGPU* g, const TGPUBuffer* src, VkDeviceSize offset, void* data, VkDeviceSize size);

// command recording; everything between begin and submit executes in order with a full barrier after
// every recorded operation
void tgpu_begin(TGPU* g);
void tgpu_dispatch(TGPU* g, const TGPUKernel* k, uint32_t x, uint32_t y, uint32_t z, const void* push, uint32_t push_size);
void tgpu_dispatch_indirect(TGPU* g, const TGPUKernel* k, const TGPUBuffer* args, VkDeviceSize offset, const void* push, uint32_t push_size);
void tgpu_fill(TGPU* g, const TGPUBuffer* b, VkDeviceSize offset, VkDeviceSize size, uint32_t value);
void tgpu_copy(TGPU* g, const TGPUBuffer* src, const TGPUBuffer* dst, VkDeviceSize size);
void tgpu_barrier(TGPU* g);
int tgpu_submit(TGPU* g);       // ends the command buffer, submits, waits. Returns 0 on success

#endif
