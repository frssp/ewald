// Minimal std::thread parallel loop shared by the native kernels.
#pragma once

#include <algorithm>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace ewald {

// Runs func(start, end, thread_index) over [0, n) in contiguous chunks.
// num_threads <= 0 uses all hardware threads.
template <class F>
inline void parallel_for(size_t n, int num_threads, F&& func)
{
    if(num_threads <= 0) num_threads = (int)std::max(1u, std::thread::hardware_concurrency());
    num_threads = (int)std::min<size_t>(num_threads, std::max<size_t>(1, n / 256));
    if(num_threads <= 1) {
        func(size_t(0), n, 0);
        return;
    }
    std::vector<std::thread> threads;
    std::exception_ptr error;
    std::mutex error_mutex;
    size_t chunk = (n + num_threads - 1) / num_threads;
    for(int t = 0; t < num_threads; t++) {
        size_t start = t * chunk, end = std::min(n, start + chunk);
        if(start >= end) break;
        threads.emplace_back([&, start, end, t]() {
            try { func(start, end, t); }
            catch(...) { std::lock_guard<std::mutex> lock(error_mutex); error = std::current_exception(); }
        });
    }
    for(auto& th : threads) th.join();
    if(error) std::rethrow_exception(error);
}

}  // namespace ewald
