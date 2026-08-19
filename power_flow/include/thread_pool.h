#ifndef POWER_FLOW_THREAD_POOL_H
#define POWER_FLOW_THREAD_POOL_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace powerflow {

// A lightweight, reusable thread pool for parallel task execution.
// Tasks are submitted as std::function and executed by worker threads.
class ThreadPool {
public:
    // Construct a pool with `num_threads` workers.
    // If num_threads == 0, hardware concurrency is used.
    explicit ThreadPool(size_t num_threads = 0)
        : stop_(false) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 4;
        }
        workers_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] { workerLoop(); });
        }
    }

    ~ThreadPool() { shutdown(); }

    // Submit a task and return a future for the result.
    template <class F, class... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type> {
        using result_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<result_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<result_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(mtx_);
            if (stop_) {
                throw std::runtime_error("ThreadPool: submit on stopped pool");
            }
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        return res;
    }

    // Number of worker threads.
    size_t worker_count() const { return workers_.size(); }

    // Gracefully stop the pool and join all workers.
    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(mtx_);
            if (stop_) return;
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
        workers_.clear();
    }

private:
    void workerLoop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::atomic<bool> stop_;
};

// Parallel for over [0, n), dispatching chunks to the given pool.
// `body(i)` is invoked for each index.
inline void parallel_for(ThreadPool& pool, size_t n, size_t chunk_size,
                         const std::function<void(size_t)>& body) {
    if (n == 0) return;
    if (chunk_size == 0) chunk_size = 1;
    size_t num_chunks = (n + chunk_size - 1) / chunk_size;
    if (num_chunks <= 1 || pool.worker_count() <= 1) {
        for (size_t i = 0; i < n; ++i) body(i);
        return;
    }
    std::vector<std::future<void>> futures;
    futures.reserve(num_chunks);
    for (size_t c = 0; c < num_chunks; ++c) {
        size_t begin = c * chunk_size;
        size_t end = std::min(begin + chunk_size, n);
        futures.emplace_back(pool.submit([begin, end, &body]() {
            for (size_t i = begin; i < end; ++i) body(i);
        }));
    }
    for (auto& f : futures) f.get();
}

} // namespace powerflow

#endif // POWER_FLOW_THREAD_POOL_H
