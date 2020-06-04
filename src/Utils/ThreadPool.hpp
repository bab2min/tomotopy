#pragma once

/*
A simple C++11 Thread Pool implementation(https://github.com/progschj/ThreadPool)
modified by bab2min to have additional parameter threadId
*/

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

namespace tomoto
{
	class ThreadPool {
	public:
		ThreadPool(size_t threads = 0, size_t maxQueued = 0);
		template<class F, class... Args>
		auto enqueue(F&& f, Args&&... args)
			->std::future<typename std::result_of<F(size_t, Args...)>::type>;
		
		template<class F, class... Args>
		auto enqueueToAll(F&& f, Args&&... args)
			->std::vector<std::future<typename std::result_of<F(size_t, Args...)>::type>>;

		~ThreadPool();
		
		size_t getNumWorkers() const { return workers.size(); }
		size_t getNumEnqued() const { return tasks.size(); }
	private:
		// need to keep track of threads so we can join them
		std::vector< std::thread > workers;
		// the task queue
		std::queue< std::function<void(size_t)> > shared_task;
		std::vector< std::queue< std::function<void(size_t)> > > tasks;
		// synchronization
		std::mutex queue_mutex;
		std::condition_variable condition, inputCnd;
		size_t maxQueued;
		bool stop;
	};


	// the constructor just launches some amount of workers
	inline ThreadPool::ThreadPool(size_t threads, size_t _maxQueued)
		: tasks(threads), maxQueued(_maxQueued), stop(false)
	{
		for (size_t i = 0; i < threads; ++i)
		{
			workers.emplace_back([this, i]
			{
				while (1)
				{
					std::function<void(size_t)> task;

					{
						std::unique_lock<std::mutex> lock(this->queue_mutex);
						this->condition.wait(lock,
							[this, i] { return this->stop || !this->shared_task.empty() || !this->tasks[i].empty(); });
						if (this->stop && this->shared_task.empty() && this->tasks[i].empty()) return;
						if (this->tasks[i].empty())
						{
							task = std::move(this->shared_task.front());
							this->shared_task.pop();
						}
						else
						{
							task = std::move(this->tasks[i].front());
							this->tasks[i].pop();
						}
						
						if (this->maxQueued) this->inputCnd.notify_all();
					}

					//std::cout << "Start #" << i << std::endl;
					task(i);
					//std::cout << "End #" << i << std::endl;
				}
			});
		}
	}

	// add new work item to the pool
	template<class F, class... Args>
	auto ThreadPool::enqueue(F&& f, Args&&... args)
		-> std::future<typename std::result_of<F(size_t, Args...)>::type>
	{
		using return_type = typename std::result_of<F(size_t, Args...)>::type;

		auto task = std::make_shared< std::packaged_task<return_type(size_t)> >(
			std::bind(std::forward<F>(f), std::placeholders::_1, std::forward<Args>(args)...));

		std::future<return_type> res = task->get_future();
		{
			std::unique_lock<std::mutex> lock(queue_mutex);

			// don't allow enqueueing after stopping the pool
			if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
			if (maxQueued && shared_task.size() >= maxQueued)
			{
				inputCnd.wait(lock, [&]() { return shared_task.size() < maxQueued; });
			}
			shared_task.emplace([task](size_t id) { (*task)(id); });
		}
		condition.notify_one();
		return res;
	}

	template<class F, class... Args>
	auto ThreadPool::enqueueToAll(F&& f, Args&&... args)
		->std::vector<std::future<typename std::result_of<F(size_t, Args...)>::type> >
	{
		using return_type = typename std::result_of<F(size_t, Args...)>::type;

		std::vector<std::future<return_type> > ret;
		std::unique_lock<std::mutex> lock(queue_mutex);
		for (size_t i = 0; i < workers.size(); ++i)
		{
			auto task = std::make_shared< std::packaged_task<return_type(size_t)> >(
				std::bind(f, std::placeholders::_1, args...));

			ret.emplace_back(task->get_future());

			{
				// don't allow enqueueing after stopping the pool
				if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
				tasks[i].emplace([task](size_t id) { (*task)(id); });
			}
		}
		condition.notify_all();
		return ret;
	}

	// the destructor joins all threads
	inline ThreadPool::~ThreadPool()
	{
		{
			std::unique_lock<std::mutex> lock(queue_mutex);
			stop = true;
		}
		condition.notify_all();
		for (std::thread &worker : workers)
			worker.join();
	}
}
