#include "init.hpp"

#include <sstream>
#include <vector>
#include <algorithm>

#define VMA_IMPLEMENTATION
#include <lib/vk_mem_alloc.h>

namespace gpu {

  Device::Device(const DeviceConfig &conf) {
    create_instance(conf);
    init_instance_extensions(conf);

    if (conf.create_swapchain) {
      surface = conf.create_surface(instance);
    
      if (!surface) {
        throw std::runtime_error {"Surface create error"};
      }
    }

    find_device(conf);
    create_device(conf);
    create_swapchain(conf);
    init_allocator();
  }

  Device::~Device() {
    vkDeviceWaitIdle(device);

    if (allocator) {
      vmaDestroyAllocator(allocator);
    }

    if (swapchain) {
      vkDestroySwapchainKHR(device, swapchain, nullptr);
    }

    if (device) {
      vkDestroyDevice(device, nullptr);
    }

    if (surface) {
      vkDestroySurfaceKHR(instance, surface, nullptr);
    }

    destroy_instance_extensions();

    if (instance) {
      vkDestroyInstance(instance, nullptr);
    }
  }

  void Device::create_instance(const DeviceConfig &config) {
    VKCHECK(volkInitialize());

    VkApplicationInfo app_info {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pNext = nullptr,
      .pApplicationName = "NONAME",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "NOENGINE",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = VK_API_VERSION_1_0
    };
  
    std::vector<const char*> layers;
    std::vector<const char*> extensions;

    layers.reserve(config.layers.size());
    for (const auto &s : config.layers) {
      layers.push_back(s.c_str());
    }

    extensions.reserve(config.instance_ext.size());
    for (auto &s : config.instance_ext) {
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

    VKCHECK(vkCreateInstance(&info, nullptr, &instance));
    
    volkLoadInstance(instance);
  }
  

  void Device::init_instance_extensions(const DeviceConfig &config) {
    if (config.create_debug_log) {
      auto iter = config.instance_ext.find(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
      if (iter == config.instance_ext.end()) {
        throw std::runtime_error {"VK_EXT_DEBUG_UTILS_EXTENSION is not set"};
      }


      VkDebugUtilsMessengerCreateInfoEXT info {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = 0,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = config.debug_log,
        .pUserData = nullptr
      };
      
      VKCHECK(vkCreateDebugUtilsMessengerEXT(instance, &info, nullptr, &debug_messenger));
    }
  }

  void Device::find_device(const DeviceConfig &config) {
    uint32_t count = 0;
    std::vector<VkPhysicalDevice> pdevices;

    VKCHECK(vkEnumeratePhysicalDevices(instance, &count, nullptr));
    pdevices.resize(count);
    VKCHECK(vkEnumeratePhysicalDevices(instance, &count, pdevices.data()));

    for (auto pdev : pdevices) {
      VkPhysicalDeviceProperties pproperties;
      vkGetPhysicalDeviceProperties(pdev, &pproperties);

      if (pproperties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        continue;
      }

      uint32_t count = 0;
      std::vector<VkQueueFamilyProperties> queues;

      vkGetPhysicalDeviceQueueFamilyProperties(pdev, &count, nullptr);
      queues.resize(count);
      vkGetPhysicalDeviceQueueFamilyProperties(pdev, &count, queues.data());

      bool queue_found = false;

      for (uint32_t i = 0; i < queues.size(); i++) {
        auto flags = queues[i].queueFlags;
        uint32_t required = VK_QUEUE_COMPUTE_BIT|VK_QUEUE_GRAPHICS_BIT|VK_QUEUE_TRANSFER_BIT;
        auto msk = flags & required;
        
        if (msk != required) {
          continue;
        }

        if (config.create_swapchain) {
          VkBool32 supported = VK_FALSE;
          VKCHECK(vkGetPhysicalDeviceSurfaceSupportKHR(pdev, i, surface, &supported));

          if (!supported) {
            continue;
          }
        }

        queue_found = true;
        queue_family = i;
      }

      if (queue_found) {
        phys_device = pdev;
        return; 
      }
    }
  }

  void Device::create_device(const DeviceConfig &config) {
    float priority = 1.f;
    
    VkDeviceQueueCreateInfo queues[] {
      {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueFamilyIndex = queue_family,
        .queueCount = 1,
        .pQueuePriorities = &priority 
      }
    };

    
    std::vector<const char*> extensions;
    extensions.reserve(config.device_ext.size());
    for (auto &s : config.device_ext) {
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

    VKCHECK(vkCreateDevice(phys_device, &info, nullptr, &device));
    vkGetDeviceQueue(device, queue_family, 0, &queue);
  }

  struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> present_modes;
  };

  static SwapChainSupportDetails query_swapchain_info(VkPhysicalDevice device, VkSurfaceKHR surface);

  void Device::create_swapchain(const DeviceConfig &config) {
    auto details = query_swapchain_info(phys_device, surface);

    VkSurfaceFormatKHR fmt = details.formats.at(0);
    for (auto sfmt : details.formats) {
      if (sfmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR && sfmt.format == VK_FORMAT_B8G8R8A8_SRGB) {
        fmt = sfmt;
        break;
      }
    }
    swapchain_format = fmt.format;

    VkPresentModeKHR mode = details.present_modes.at(0);

    swapchain_extent = details.capabilities.currentExtent;

    if (swapchain_extent.width == UINT32_MAX) {
      swapchain_extent = config.get_extent();
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
      .queueFamilyIndexCount = 1,
      .pQueueFamilyIndices = &queue_family,
      .preTransform = details.capabilities.currentTransform,
      .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .presentMode = mode,
      .clipped = VK_TRUE,
      .oldSwapchain = nullptr
    };

    VKCHECK(vkCreateSwapchainKHR(device, &info, nullptr, &swapchain));
    uint32_t s_image_count = 0;
    VKCHECK(vkGetSwapchainImagesKHR(device, swapchain, &s_image_count, nullptr));
    swapchain_images.resize(s_image_count);
    VKCHECK(vkGetSwapchainImagesKHR(device, swapchain, &s_image_count, swapchain_images.data()));
  }

  /*std::vector<gpu::Image> Device::get_backbuffers() {
    if (!swapchain) {
      return {};
    }
    std::vector<VkImage> images;
    std::vector<gpu::Image> backbuffers;
    //backbuffers.reserve();

    uint32_t s_image_count = 0;
    VKCHECK(vkGetSwapchainImagesKHR(device, swapchain, &s_image_count, nullptr));
    images.resize(s_image_count);
    VKCHECK(vkGetSwapchainImagesKHR(device, swapchain, &s_image_count, images.data()));
    backbuffers.reserve(s_image_count);

    VkImageCreateInfo image_info {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = swapchain_format,
      .extent {swapchain_extent.width, swapchain_extent.height, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr
    };

    for (uint32_t i = 0; i < s_image_count; i++) {
      gpu::Image temp {*this};
      temp.create_reference(images[i], image_info);
      backbuffers.push_back(std::move(temp));
    }

    return backbuffers;
  }*/

  void Device::destroy_instance_extensions() {
    if (!instance)
      return;

    if (debug_messenger) {
      vkDestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);
    }
  }

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

  void Device::init_allocator()
  {
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

    VmaAllocatorCreateInfo info {};
    info.device = device;
    info.instance = instance;
    info.physicalDevice = phys_device;
    info.vulkanApiVersion = VK_API_VERSION_1_0;
    info.pVulkanFunctions = &vk_func;

    VKCHECK(vmaCreateAllocator(&info, &allocator));
  }
}