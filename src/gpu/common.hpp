#ifndef VKERROR_HPP_INCLUDED
#define VKERROR_HPP_INCLUDED

#include "lib/volk.h"

#define VKCHECK(expr) gpu::vk_check_error((expr), __FILE__, __LINE__, #expr) 

namespace gpu {
  void vk_check_error(VkResult result, const char *file, int line, const char *cmd);

  template<typename HandleT, typename Deleter>
  struct HandleHolder {
    HandleHolder() {}
    HandleHolder(HandleT elem) : handle {elem} {}
    
    ~HandleHolder() { 
      if (handle) { 
        Deleter deleter {};
        deleter(handle); 
      }
    }

    HandleHolder(HandleHolder<HandleT, Deleter> &&rhs) : handle {rhs.handle}  
      { rhs.handle = nullptr; }

    HandleHolder<HandleT, Deleter> &operator=(HandleHolder<HandleT, Deleter> &&rhs) {
      //std::swap(handle, rhs.handle);
      return *this;
    }

    HandleHolder(const HandleHolder<HandleT, Deleter> &) = delete;
    HandleHolder<HandleT, Deleter> &operator=(const HandleHolder<HandleT, Deleter> &) = delete;

    operator HandleT() const { return handle; }

  protected:
    HandleT handle {nullptr};
  };
}

#endif