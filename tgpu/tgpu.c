#include "tgpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef TGPU_ASSERT
#define TGPU_ASSERT(call, msg) do { VkResult _r = (call); if (_r != VK_SUCCESS) { fprintf(stderr, "[tgpu] %s (VkResult %d)\n", msg, (int)_r); return 1; } } while (0)
#endif

static int find_memory(TGPU* g, uint32_t bits, VkMemoryPropertyFlags want, uint32_t* out) {
    for (uint32_t i = 0; i < g->memprops.memoryTypeCount; i++) {
        if ((bits & (1u << i)) && (g->memprops.memoryTypes[i].propertyFlags & want) == want) { *out = i; return 1; }
    }
    return 0;
}

static int create_raw_buffer(TGPU* g, TGPUBuffer* b, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props) {
    if (size == 0) size = 4;
    size = (size + 3) & ~(VkDeviceSize)3;
    VkBufferCreateInfo bi = { 0 };
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    TGPU_ASSERT(vkCreateBuffer(g->device, &bi, NULL, &b->buffer), "vkCreateBuffer failed");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g->device, b->buffer, &req);
    uint32_t type;
    if (!find_memory(g, req.memoryTypeBits, props, &type)) {
        fprintf(stderr, "[tgpu] no suitable memory type\n");
        vkDestroyBuffer(g->device, b->buffer, NULL);
        return 1;
    }
    VkMemoryAllocateInfo ai = { 0 };
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    VkResult r = vkAllocateMemory(g->device, &ai, NULL, &b->memory);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "[tgpu] vkAllocateMemory failed for %llu bytes (VkResult %d)\n", (unsigned long long)req.size, (int)r);
        vkDestroyBuffer(g->device, b->buffer, NULL);
        return 1;
    }
    vkBindBufferMemory(g->device, b->buffer, b->memory, 0);
    b->size = size;
    return 0;
}

int tgpu_create(TGPU* g, int preferred, int want_atomic_float) {
    memset(g, 0, sizeof(*g));
    VkApplicationInfo app = { 0 };
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "prism-trainer";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici = { 0 };
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    TGPU_ASSERT(vkCreateInstance(&ici, NULL, &g->instance), "vkCreateInstance failed");

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(g->instance, &count, NULL);
    if (count == 0) { fprintf(stderr, "[tgpu] no Vulkan devices\n"); return 1; }
    VkPhysicalDevice* devs = calloc(count, sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(g->instance, &count, devs);
    int pick = -1;
    if (preferred >= 0 && (uint32_t)preferred < count) pick = preferred;
    else {
        int best = -1;
        for (uint32_t i = 0; i < count; i++) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(devs[i], &p);
            int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 4 :
                        p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 3 :
                        p.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU ? 2 : 1;
            if (score > best) { best = score; pick = (int)i; }
        }
    }
    g->gpu = devs[pick];
    free(devs);
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(g->gpu, &props);
    strncpy(g->gpu_name, props.deviceName, sizeof(g->gpu_name) - 1);
    g->max_workgroup_invocations = props.limits.maxComputeWorkGroupInvocations;
    vkGetPhysicalDeviceMemoryProperties(g->gpu, &g->memprops);

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g->gpu, &nq, NULL);
    VkQueueFamilyProperties* qs = calloc(nq, sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(g->gpu, &nq, qs);
    int fam = -1;
    for (uint32_t i = 0; i < nq; i++) if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { fam = (int)i; break; }
    free(qs);
    if (fam < 0) { fprintf(stderr, "[tgpu] no compute queue\n"); return 1; }
    g->family = (uint32_t)fam;

    // optional float atomic add
    uint32_t next = 0;
    vkEnumerateDeviceExtensionProperties(g->gpu, NULL, &next, NULL);
    VkExtensionProperties* exts = calloc(next, sizeof(VkExtensionProperties));
    vkEnumerateDeviceExtensionProperties(g->gpu, NULL, &next, exts);
    int has_ext = 0;
    for (uint32_t i = 0; i < next; i++) if (strcmp(exts[i].extensionName, "VK_EXT_shader_atomic_float") == 0) has_ext = 1;
    free(exts);
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT af = { 0 };
    af.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    int use_atomic = 0;
    if (want_atomic_float && has_ext) {
        VkPhysicalDeviceFeatures2 f2 = { 0 };
        f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        f2.pNext = &af;
        vkGetPhysicalDeviceFeatures2(g->gpu, &f2);
        use_atomic = af.shaderBufferFloat32AtomicAdd ? 1 : 0;
    }
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT afe = { 0 };
    afe.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    afe.shaderBufferFloat32Atomics = VK_TRUE;
    afe.shaderBufferFloat32AtomicAdd = VK_TRUE;
    g->atomic_float_add = use_atomic;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = { 0 };
    qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qi.queueFamilyIndex = g->family;
    qi.queueCount = 1;
    qi.pQueuePriorities = &prio;
    const char* dev_exts[] = { "VK_EXT_shader_atomic_float" };
    VkDeviceCreateInfo di = { 0 };
    di.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    di.queueCreateInfoCount = 1;
    di.pQueueCreateInfos = &qi;
    if (use_atomic) {
        di.pNext = &afe;
        di.enabledExtensionCount = 1;
        di.ppEnabledExtensionNames = dev_exts;
    }
    TGPU_ASSERT(vkCreateDevice(g->gpu, &di, NULL, &g->device), "vkCreateDevice failed");
    vkGetDeviceQueue(g->device, g->family, 0, &g->queue);

    VkCommandPoolCreateInfo cpi = { 0 };
    cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = g->family;
    TGPU_ASSERT(vkCreateCommandPool(g->device, &cpi, NULL, &g->pool), "vkCreateCommandPool failed");
    VkCommandBufferAllocateInfo cbi = { 0 };
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool = g->pool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    TGPU_ASSERT(vkAllocateCommandBuffers(g->device, &cbi, &g->cmd), "vkAllocateCommandBuffers failed");
    VkFenceCreateInfo fi = { 0 };
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    TGPU_ASSERT(vkCreateFence(g->device, &fi, NULL, &g->fence), "vkCreateFence failed");

    VkDescriptorSetLayoutBinding binds[TGPU_MAX_BINDINGS];
    for (uint32_t i = 0; i < TGPU_MAX_BINDINGS; i++) {
        binds[i] = (VkDescriptorSetLayoutBinding){ i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL };
    }
    VkDescriptorSetLayoutCreateInfo dli = { 0 };
    dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = TGPU_MAX_BINDINGS;
    dli.pBindings = binds;
    TGPU_ASSERT(vkCreateDescriptorSetLayout(g->device, &dli, NULL, &g->set_layout), "vkCreateDescriptorSetLayout failed");
    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, TGPU_MAX_BINDINGS };
    VkDescriptorPoolCreateInfo dpi = { 0 };
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 1;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    TGPU_ASSERT(vkCreateDescriptorPool(g->device, &dpi, NULL, &g->desc_pool), "vkCreateDescriptorPool failed");
    VkDescriptorSetAllocateInfo dai = { 0 };
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = g->desc_pool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &g->set_layout;
    TGPU_ASSERT(vkAllocateDescriptorSets(g->device, &dai, &g->set), "vkAllocateDescriptorSets failed");

    VkPushConstantRange pr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, TGPU_PUSH_SIZE };
    VkPipelineLayoutCreateInfo pli = { 0 };
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &g->set_layout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pr;
    TGPU_ASSERT(vkCreatePipelineLayout(g->device, &pli, NULL, &g->layout), "vkCreatePipelineLayout failed");
    return 0;
}

void tgpu_destroy(TGPU* g) {
    if (!g->device) { if (g->instance) vkDestroyInstance(g->instance, NULL); memset(g, 0, sizeof(*g)); return; }
    vkDeviceWaitIdle(g->device);
    if (g->staging.buffer) { vkUnmapMemory(g->device, g->staging.memory); tgpu_buffer_destroy(g, &g->staging); }
    vkDestroyPipelineLayout(g->device, g->layout, NULL);
    vkDestroyDescriptorPool(g->device, g->desc_pool, NULL);
    vkDestroyDescriptorSetLayout(g->device, g->set_layout, NULL);
    vkDestroyFence(g->device, g->fence, NULL);
    vkDestroyCommandPool(g->device, g->pool, NULL);
    vkDestroyDevice(g->device, NULL);
    vkDestroyInstance(g->instance, NULL);
    memset(g, 0, sizeof(*g));
}

int tgpu_buffer_create(TGPU* g, TGPUBuffer* b, VkDeviceSize size) {
    memset(b, 0, sizeof(*b));
    return create_raw_buffer(g, b,
        size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

void tgpu_buffer_destroy(TGPU* g, TGPUBuffer* b) {
    if (b->buffer) vkDestroyBuffer(g->device, b->buffer, NULL);
    if (b->memory) vkFreeMemory(g->device, b->memory, NULL);
    memset(b, 0, sizeof(*b));
}

void tgpu_bind(TGPU* g, uint32_t binding, const TGPUBuffer* b) {
    VkDescriptorBufferInfo info = { b->buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet w = { 0 };
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = g->set;
    w.dstBinding = binding;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &info;
    vkUpdateDescriptorSets(g->device, 1, &w, 0, NULL);
}

int tgpu_kernel_create(TGPU* g, TGPUKernel* k, const char* path) {
    memset(k, 0, sizeof(*k));
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[tgpu] cannot open shader \"%s\"\n", path); return 1; }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t* code = malloc((size_t)size);
    if (fread(code, 1, (size_t)size, f) != (size_t)size) { fclose(f); free(code); return 1; }
    fclose(f);
    VkShaderModuleCreateInfo si = { 0 };
    si.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    si.codeSize = (size_t)size;
    si.pCode = code;
    VkResult r = vkCreateShaderModule(g->device, &si, NULL, &k->module);
    free(code);
    if (r != VK_SUCCESS) { fprintf(stderr, "[tgpu] shader module failed for \"%s\"\n", path); return 1; }
    VkComputePipelineCreateInfo pi = { 0 };
    pi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pi.layout = g->layout;
    pi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pi.stage.module = k->module;
    pi.stage.pName = "main";
    r = vkCreateComputePipelines(g->device, VK_NULL_HANDLE, 1, &pi, NULL, &k->pipeline);
    if (r != VK_SUCCESS) { fprintf(stderr, "[tgpu] pipeline failed for \"%s\"\n", path); return 1; }
    return 0;
}

void tgpu_kernel_destroy(TGPU* g, TGPUKernel* k) {
    if (k->pipeline) vkDestroyPipeline(g->device, k->pipeline, NULL);
    if (k->module) vkDestroyShaderModule(g->device, k->module, NULL);
    memset(k, 0, sizeof(*k));
}

static int ensure_staging(TGPU* g, VkDeviceSize size) {
    if (g->staging.buffer && g->staging.size >= size) return 0;
    if (g->staging.buffer) { vkUnmapMemory(g->device, g->staging.memory); tgpu_buffer_destroy(g, &g->staging); }
    VkDeviceSize want = size < (4u << 20) ? (4u << 20) : size;
    if (create_raw_buffer(g, &g->staging, want,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) return 1;
    TGPU_ASSERT(vkMapMemory(g->device, g->staging.memory, 0, VK_WHOLE_SIZE, 0, &g->staging_mapped), "vkMapMemory failed");
    return 0;
}

static int run_once(TGPU* g, VkBuffer src, VkBuffer dst, VkDeviceSize so, VkDeviceSize doff, VkDeviceSize size) {
    tgpu_begin(g);
    VkBufferCopy c = { so, doff, size };
    vkCmdCopyBuffer(g->cmd, src, dst, 1, &c);
    tgpu_barrier(g);
    return tgpu_submit(g);
}

int tgpu_upload(TGPU* g, const TGPUBuffer* dst, VkDeviceSize offset, const void* data, VkDeviceSize size) {
    const char* p = (const char*)data;
    VkDeviceSize chunk = 64u << 20;
    while (size > 0) {
        VkDeviceSize n = size < chunk ? size : chunk;
        if (ensure_staging(g, n)) return 1;
        memcpy(g->staging_mapped, p, (size_t)n);
        if (run_once(g, g->staging.buffer, dst->buffer, 0, offset, n)) return 1;
        p += n; offset += n; size -= n;
    }
    return 0;
}

int tgpu_download(TGPU* g, const TGPUBuffer* src, VkDeviceSize offset, void* data, VkDeviceSize size) {
    char* p = (char*)data;
    VkDeviceSize chunk = 64u << 20;
    while (size > 0) {
        VkDeviceSize n = size < chunk ? size : chunk;
        if (ensure_staging(g, n)) return 1;
        if (run_once(g, src->buffer, g->staging.buffer, offset, 0, n)) return 1;
        memcpy(p, g->staging_mapped, (size_t)n);
        p += n; offset += n; size -= n;
    }
    return 0;
}

void tgpu_begin(TGPU* g) {
    vkResetCommandBuffer(g->cmd, 0);
    VkCommandBufferBeginInfo bi = { 0 };
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(g->cmd, &bi);
    g->recording = 1;
}

void tgpu_barrier(TGPU* g) {
    VkMemoryBarrier mb = { 0 };
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                       VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    VkPipelineStageFlags stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    vkCmdPipelineBarrier(g->cmd, stages, stages | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

static void push_and_bind(TGPU* g, const TGPUKernel* k, const void* push, uint32_t push_size) {
    vkCmdBindPipeline(g->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipeline);
    vkCmdBindDescriptorSets(g->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g->layout, 0, 1, &g->set, 0, NULL);
    char block[TGPU_PUSH_SIZE] = { 0 };
    if (push && push_size > 0) memcpy(block, push, push_size > TGPU_PUSH_SIZE ? TGPU_PUSH_SIZE : push_size);
    vkCmdPushConstants(g->cmd, g->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, TGPU_PUSH_SIZE, block);
}

void tgpu_dispatch(TGPU* g, const TGPUKernel* k, uint32_t x, uint32_t y, uint32_t z, const void* push, uint32_t push_size) {
    if (x == 0 || y == 0 || z == 0) return;
    push_and_bind(g, k, push, push_size);
    vkCmdDispatch(g->cmd, x, y, z);
    tgpu_barrier(g);
}

void tgpu_dispatch_indirect(TGPU* g, const TGPUKernel* k, const TGPUBuffer* args, VkDeviceSize offset, const void* push, uint32_t push_size) {
    push_and_bind(g, k, push, push_size);
    vkCmdDispatchIndirect(g->cmd, args->buffer, offset);
    tgpu_barrier(g);
}

void tgpu_fill(TGPU* g, const TGPUBuffer* b, VkDeviceSize offset, VkDeviceSize size, uint32_t value) {
    vkCmdFillBuffer(g->cmd, b->buffer, offset, size, value);
    tgpu_barrier(g);
}

void tgpu_copy(TGPU* g, const TGPUBuffer* src, const TGPUBuffer* dst, VkDeviceSize size) {
    VkBufferCopy c = { 0, 0, size };
    vkCmdCopyBuffer(g->cmd, src->buffer, dst->buffer, 1, &c);
    tgpu_barrier(g);
}

int tgpu_submit(TGPU* g) {
    if (vkEndCommandBuffer(g->cmd) != VK_SUCCESS) { fprintf(stderr, "[tgpu] vkEndCommandBuffer failed\n"); return 1; }
    g->recording = 0;
    VkSubmitInfo si = { 0 };
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g->cmd;
    VkResult r = vkQueueSubmit(g->queue, 1, &si, g->fence);
    if (r != VK_SUCCESS) { fprintf(stderr, "[tgpu] vkQueueSubmit failed (%d)\n", (int)r); return 1; }
    r = vkWaitForFences(g->device, 1, &g->fence, VK_TRUE, UINT64_MAX);
    vkResetFences(g->device, 1, &g->fence);
    if (r != VK_SUCCESS) { fprintf(stderr, "[tgpu] fence wait failed (%d)\n", (int)r); return 1; }
    return 0;
}
