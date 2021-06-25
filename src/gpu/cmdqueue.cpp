#include "cmdqueue.hpp"

#include <cstring>
#include <cassert>
#include <thread>
#include <cstdio>

namespace gpu {

  ByteQueue::ByteQueue(uint32_t buff_size) {
    buffer = new uint8_t[buff_size];
    assert(buffer != nullptr);
    
    buffer_size = buff_size;
    read_pos = buffer;
    write_pos.store(buffer, std::memory_order_release);
    alloc_pos.store(buffer, std::memory_order_release);
  }

  ByteQueue::~ByteQueue() {
    delete[] buffer;
  } 


  QueueStatus ByteQueue::write(uint32_t size, const void *data) {

    if (shutdown_flag.load(std::memory_order_relaxed)) {
      return QueueStatus::Closed;
    }

    const uint8_t *end_pos = buffer + buffer_size;
    auto pos = alloc_pos.fetch_add(sizeof(uint32_t) + size, std::memory_order_relaxed);

    if (pos > end_pos) {
      return QueueStatus::Overflow;
    }

    if (pos + sizeof(uint32_t) + size > end_pos) {
      overflow_flag.store(true, std::memory_order_relaxed);
      return QueueStatus::Overflow;
    }
    
    
    std::memcpy(pos, &size, sizeof(size));
    std::memcpy(pos + sizeof(size), data, size);

    auto ptr = pos + size + sizeof(size);

    while(!write_pos.compare_exchange_weak(pos, ptr, std::memory_order_relaxed, std::memory_order_release)) {
      //std::this_thread::sleep_for(std::chrono::nanoseconds(1));
    }

    return QueueStatus::Ok;
  }

  QueueStatus ByteQueue::read(uint32_t max_size, void *dst, uint32_t &readed) {
    const uint8_t *end_pos = buffer + buffer_size;

    while (true) {
      
      auto pos = write_pos.load(std::memory_order_acquire);
      if (pos != read_pos) {
        
        if (end_pos < read_pos + sizeof(uint32_t)) {
          throw std::runtime_error {"Queue broken"};
        }

        std::memcpy(&readed, read_pos, sizeof(readed));
        read_pos += sizeof(readed);

        if (readed > max_size) {
          throw std::runtime_error {"Not enouth memory to store readed value"};
        }

        if (end_pos < read_pos + readed) {
          throw std::runtime_error {"Queue broken"};
        }

        std::memcpy(dst, read_pos, readed);
        read_pos += readed;
        if (read_pos > pos) {
          throw std::runtime_error {"Queue broken"};
        }

        return QueueStatus::Ok;
      } else {
        if (shutdown_flag.load(std::memory_order_relaxed)) {
          return QueueStatus::Closed;
        }

        if (overflow_flag.load(std::memory_order_relaxed)) {
          return QueueStatus::Overflow;
        }
      }
    }
    
    return QueueStatus::Ok;
  }

  void ByteQueue::write_done() {
    shutdown_flag.store(true, std::memory_order_release);
  }
  
  bool ByteQueue::is_closed() const {
    auto pos = write_pos.load(std::memory_order_acquire);
    if (pos == read_pos) {
      return shutdown_flag.load(std::memory_order_acquire);
    }
    return false;
  }

  void ByteQueue::wait_reset() {
    while (overflow_flag.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(10));
    }
  }

  void ByteQueue::reset() {
    read_pos = buffer;
    alloc_pos.store(buffer, std::memory_order_relaxed);
    write_pos.store(buffer, std::memory_order_relaxed);
    overflow_flag.store(false, std::memory_order_release);
  }

}