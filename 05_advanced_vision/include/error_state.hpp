#pragma once

#include <exception>
#include <mutex>

class ErrorState {
 public:
  void capture_current_exception() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!error_) {
      error_ = std::current_exception();
    }
  }

  void rethrow_if_set() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (error_) {
      std::rethrow_exception(error_);
    }
  }

 private:
  mutable std::mutex mutex_;
  std::exception_ptr error_;
};
