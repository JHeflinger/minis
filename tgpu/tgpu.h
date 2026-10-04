#ifndef TGPU_H
#define TGPU_H

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
    int atomic_float_add;
    uint32_t max_workgroup_invocations;
    int recording;
    TGPUBuffer staging;
    void* staging_mapped;
} TGPU;

int tgpu_create(TGPU* g, int preferred, int want_atomic_float); // set preferred -1 for best, otherwise device index

void tgpu_destroy(TGPU* g);

int tgpu_buffer_create(TGPU* g, TGPUBuffer* b, VkDeviceSize size);

void tgpu_buffer_destroy(TGPU* g, TGPUBuffer* b);

void tgpu_bind(TGPU* g, uint32_t binding, const TGPUBuffer* b);

int tgpu_kernel_create(TGPU* g, TGPUKernel* k, const char* spv_path);

void tgpu_kernel_destroy(TGPU* g, TGPUKernel* k);

int tgpu_upload(TGPU* g, const TGPUBuffer* dst, VkDeviceSize offset, const void* data, VkDeviceSize size);

int tgpu_download(TGPU* g, const TGPUBuffer* src, VkDeviceSize offset, void* data, VkDeviceSize size);

void tgpu_begin(TGPU* g);

void tgpu_dispatch(TGPU* g, const TGPUKernel* k, uint32_t x, uint32_t y, uint32_t z, const void* push, uint32_t push_size);

void tgpu_dispatch_indirect(TGPU* g, const TGPUKernel* k, const TGPUBuffer* args, VkDeviceSize offset, const void* push, uint32_t push_size);

void tgpu_fill(TGPU* g, const TGPUBuffer* b, VkDeviceSize offset, VkDeviceSize size, uint32_t value);

void tgpu_copy(TGPU* g, const TGPUBuffer* src, const TGPUBuffer* dst, VkDeviceSize size);

void tgpu_barrier(TGPU* g);

int tgpu_submit(TGPU* g);

#endif
