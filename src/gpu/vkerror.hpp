#ifndef VKERROR_HPP_INCLUDED
#define VKERROR_HPP_INCLUDED

#include "lib/volk.h"

#define VKCHECK(expr) gpu::vk_check_error((expr), __FILE__, __LINE__, #expr) 

namespace gpu {
  void vk_check_error(VkResult result, const char *file, int line, const char *cmd);
}

#endif