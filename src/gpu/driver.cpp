#include "driver.hpp"

#include <vector>

#define VMA_IMPLEMENTATION
#include <lib/vk_mem_alloc.h>

namespace gpu {

  Instance::Instance(const InstanceConfig &cfg) {
    VKCHECK(volkInitialize());

    VkApplicationInfo app_info {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pNext = nullptr,
      .pApplicationName = cfg.app_name.c_str(),
      .applicationVersion = cfg.app_version,
      .pEngineName = cfg.engine_name.c_str(),
      .engineVersion = cfg.engine_version,
      .apiVersion = cfg.api_version
    };
  
    std::vector<const char*> layers;
    std::vector<const char*> extensions;

    layers.reserve(cfg.layers.size());
    for (const auto &s : cfg.layers) {
      layers.push_back(s.c_str());
    }

    extensions.reserve(cfg.extensions.size());
    for (auto &s : cfg.extensions) {
      extensions.push_back(s.c_str());
    }

    VkInstanceCreateInfo info {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .pApplicationInfo = &app_info,
      .enabledLayerCount = (uint32_t)layers.size(),
      .ppEnabledLayerNames = layers.data(),
      .enabledExtensionCount = (uint32_t)extensions.size(),
      .ppEnabledExtensionNames = extensions.data()
    };

    VKCHECK(vkCreateInstance(&info, nullptr, &handle));
    
    volkLoadInstance(handle);
  }

  Instance::~Instance() {
    if (handle) {
      vkDestroyInstance(handle, nullptr);
    }
  }

  Device Instance::create_device(const DeviceConfig &cfg) {
    return {handle, cfg};
  }
  
  DebugMessenger Instance::create_debug_messenger(PFN_vkDebugUtilsMessengerCallbackEXT callback) {
    return {handle, callback};
  }

  DebugMessenger::DebugMessenger(VkInstance instance, PFN_vkDebugUtilsMessengerCallbackEXT callback) {
    VkDebugUtilsMessengerCreateInfoEXT info {
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
      .pNext = nullptr,
      .flags = 0,
      .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
      .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
      .pfnUserCallback = callback,
      .pUserData = nullptr
    };
      
    VKCHECK(vkCreateDebugUtilsMessengerEXT(instance, &info, nullptr, &handle));
  }
  
  DebugMessenger::~DebugMessenger() {
    if (base && handle) {
      vkDestroyDebugUtilsMessengerEXT(base, handle, nullptr);
    }
  }


  struct DeviceQueryInfo {
    bool complete = false;
    uint32_t queue_family_index = 0;
    VkPhysicalDeviceProperties properties;
  };

  static DeviceQueryInfo pick_physical_device(VkPhysicalDevice device, const DeviceConfig &cfg) {
    VkPhysicalDeviceProperties pproperties;
    vkGetPhysicalDeviceProperties(device, &pproperties);

    if (pproperties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      return {false, 0};
    }

    uint32_t count = 0;
    std::vector<VkQueueFamilyProperties> queues;

    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    queues.resize(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, queues.data());

    bool queue_found = false;
    uint32_t queue_family = 0;

    for (uint32_t i = 0; i < queues.size(); i++) {
      auto flags = queues[i].queueFlags;
      uint32_t required = VK_QUEUE_COMPUTE_BIT|VK_QUEUE_GRAPHICS_BIT|VK_QUEUE_TRANSFER_BIT;
      auto msk = flags & required;
        
      if (msk != required) {
        continue;
      }

      if (cfg.surface) {
        VkBool32 supported = VK_FALSE;
        VKCHECK(vkGetPhysicalDeviceSurfaceSupportKHR(device, i, cfg.surface, &supported));

        if (!supported) {
          continue;
        }
      }

      queue_found = true;
      queue_family = i;
    }

    return {queue_found, queue_family, pproperties};
  }

  Device::Device(VkInstance instance, const DeviceConfig &cfg) {
    uint32_t count = 0;
    std::vector<VkPhysicalDevice> pdevices;

    VKCHECK(vkEnumeratePhysicalDevices(instance, &count, nullptr));
    pdevices.resize(count);
    VKCHECK(vkEnumeratePhysicalDevices(instance, &count, pdevices.data()));

    DeviceQueryInfo query {};
    for (auto pdev : pdevices) {
      query = pick_physical_device(pdev, cfg);
      if (query.complete) {
        physical_device = pdev;
        properties = query.properties;
        break;
      }
    }

    queue_family_index = query.queue_family_index;    

    float priority = 1.f;
    
    VkDeviceQueueCreateInfo queues[] {
      {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueFamilyIndex = queue_family_index,
        .queueCount = 1,
        .pQueuePriorities = &priority 
      }
    };

    
    std::vector<const char*> extensions;
    extensions.reserve(cfg.extensions.size());
    for (auto &s : cfg.extensions) {
      extensions.push_back(s.c_str());
    }

    VkPhysicalDeviceFeatures features {};

    VkDeviceCreateInfo info {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = queues,
      .enabledLayerCount = 0,
      .ppEnabledLayerNames = nullptr,
      .enabledExtensionCount = (uint32_t)extensions.size(),
      .ppEnabledExtensionNames = extensions.data(),
      .pEnabledFeatures = &features
    };

    VKCHECK(vkCreateDevice(physical_device, &info, nullptr, &logical_device));
    vkGetDeviceQueue(logical_device, queue_family_index, 0, &queue);
  
    VmaVulkanFunctions vk_func {
      vkGetPhysicalDeviceProperties,
      vkGetPhysicalDeviceMemoryProperties,
      vkAllocateMemory,
      vkFreeMemory,
      vkMapMemory,
      vkUnmapMemory,
      vkFlushMappedMemoryRanges,
      vkInvalidateMappedMemoryRanges,
      vkBindBufferMemory,
      vkBindImageMemory,
      vkGetBufferMemoryRequirements,
      vkGetImageMemoryRequirements,
      vkCreateBuffer,
      vkDestroyBuffer,
      vkCreateImage,
      vkDestroyImage,
      vkCmdCopyBuffer,
    };

    VmaAllocatorCreateInfo alloc_info {};
    alloc_info.device = logical_device;
    alloc_info.instance = instance;
    alloc_info.physicalDevice = physical_device;
    alloc_info.vulkanApiVersion = VK_API_VERSION_1_0;
    alloc_info.pVulkanFunctions = &vk_func;

    VKCHECK(vmaCreateAllocator(&alloc_info, &allocator));
  }
  
  Device::Device(Device &&dev)
    : physical_device {dev.physical_device}, properties {dev.properties}, logical_device {dev.logical_device},
      allocator{dev.allocator}, queue_family_index {dev.queue_family_index},
      queue {dev.queue}
  {
    dev.logical_device = nullptr;
    dev.allocator = nullptr;
  }

  Device::~Device() {
    if (allocator) {
      vmaDestroyAllocator(allocator);
    }
    if (logical_device) {
      vkDestroyDevice(logical_device, nullptr);
    }
  }

  const Device &Device::operator=(Device &&dev) {
    std::swap(physical_device, dev.physical_device);
    std::swap(properties, dev.properties);
    std::swap(logical_device, dev.logical_device);
    std::swap(allocator, dev.allocator);
    std::swap(queue_family_index, dev.queue_family_index);
    std::swap(queue, dev.queue);
    return *this;
  }

  Swapchain Device::create_swapchain(VkSurfaceKHR surface, VkExtent2D window_extent, VkImageUsageFlags usage) {
    return {logical_device, physical_device, surface, window_extent, usage};
  }
  
  std::vector<Image> Device::get_swapchain_images(const Swapchain &swapchain) {
    uint32_t images_count = 0;
    VKCHECK(vkGetSwapchainImagesKHR(logical_device, swapchain.api_swapchain(), &images_count, nullptr));
    std::vector<VkImage> api_images;
    api_images.resize(images_count);
    VKCHECK(vkGetSwapchainImagesKHR(logical_device, swapchain.api_swapchain(), &images_count, api_images.data()));

    std::vector<Image> images;
    images.reserve(images_count);
    for (auto handle : api_images) {
      images.emplace_back(logical_device, allocator);
      images.back().create_reference(handle, swapchain.get_image_info());
    }

    return images;
  }

  struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> present_modes;
  };

  static SwapChainSupportDetails query_swapchain_info(VkPhysicalDevice device, VkSurfaceKHR surface) {
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

    if (formatCount != 0) {
      details.formats.resize(formatCount);
      vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

    if (presentModeCount != 0) {
      details.present_modes.resize(presentModeCount);
      vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.present_modes.data());
    }

    return details;
  }

  Swapchain::Swapchain(VkDevice device, VkPhysicalDevice physical_device, VkSurfaceKHR surface, VkExtent2D window, VkImageUsageFlags image_usage) {
    auto details = query_swapchain_info(physical_device, surface);

    VkSurfaceFormatKHR fmt = details.formats.at(0);
    for (auto sfmt : details.formats) {
      if (sfmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR && sfmt.format == VK_FORMAT_B8G8R8A8_SRGB) {
        fmt = sfmt;
        break;
      }
    }

    VkPresentModeKHR mode = details.present_modes.at(0);

    auto swapchain_extent = details.capabilities.currentExtent;

    if (swapchain_extent.width == UINT32_MAX) {
      swapchain_extent = window;
      swapchain_extent.width = std::clamp(swapchain_extent.width, details.capabilities.minImageExtent.width, details.capabilities.maxImageExtent.width);
      swapchain_extent.height = std::clamp(swapchain_extent.height, details.capabilities.minImageExtent.height, details.capabilities.maxImageExtent.height);
    }

    uint32_t image_count = details.capabilities.minImageCount + 1;
    if (details.capabilities.maxImageCount > 0 && image_count > details.capabilities.maxImageCount) {
      image_count = details.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR info {
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .pNext = nullptr,
      .flags = 0,
      .surface = surface,
      .minImageCount = image_count,
      .imageFormat = fmt.format,
      .imageColorSpace = fmt.colorSpace,
      .imageExtent = swapchain_extent,
      .imageArrayLayers = 1,
      .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
      .preTransform = details.capabilities.currentTransform,
      .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .presentMode = mode,
      .clipped = VK_TRUE,
      .oldSwapchain = nullptr
    };

    VKCHECK(vkCreateSwapchainKHR(device, &info, nullptr, &handle));
    
    descriptor = ImageInfo {
      fmt.format,
      VK_IMAGE_ASPECT_COLOR_BIT,
      swapchain_extent.width,
      swapchain_extent.height,
    };
  }
  
  Swapchain::~Swapchain() {
    if (base && handle) {
      vkDestroySwapchainKHR(base, handle, nullptr);
    }
  }

}