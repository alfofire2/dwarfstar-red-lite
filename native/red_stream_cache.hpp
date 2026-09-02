#pragma once

#import <Metal/Metal.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace redlite {

struct Segment { uint64_t offset = 0; uint64_t length = 0; };

struct ExpertRecord {
    int layer = 0;
    int expert = 0;
    uint64_t total_bytes = 0;
    std::vector<Segment> segments;
    uint64_t key() const {
        return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
               static_cast<uint32_t>(expert);
    }
};

struct CacheStats {
    uint64_t requests = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t bytes_read = 0;
    uint64_t read_ns = 0;
};

std::vector<ExpertRecord> load_manifest(const std::string & path);

class ExpertCache {
public:
    ExpertCache(const std::string & model_path, const std::vector<ExpertRecord> & records,
                uint64_t cache_bytes, bool nocache);
    ~ExpertCache();
    ExpertCache(const ExpertCache &) = delete;
    ExpertCache & operator=(const ExpertCache &) = delete;

    bool ensure(const ExpertRecord & rec, bool count_request = true);
    void prefetch(const ExpertRecord & rec);
    void stop_prefetch();

    id<MTLBuffer> buffer_for_slot(size_t slot) const;
    uint64_t buffer_offset_for_slot(size_t slot) const;
    uint64_t slot_bytes() const { return slot_bytes_; }
    uint64_t slots() const { return slots_; }
    CacheStats stats() const;

private:
    enum class SlotState { Empty, Loading, Ready };
    struct Slot { uint64_t key = 0; uint64_t stamp = 0; SlotState state = SlotState::Empty; };

    bool load_into_slot(const ExpertRecord & rec, size_t slot_index);
    size_t reserve_slot_locked(uint64_t key);
    void worker_loop();

    int fd_ = -1;
    id<MTLDevice> device_ = nil;
    std::vector<id<MTLBuffer>> buffers_;
    uint64_t slot_bytes_ = 0;
    uint64_t slots_ = 0;
    uint64_t slots_per_buffer_ = 0;

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::vector<Slot> slot_state_;
    std::unordered_map<uint64_t, size_t> key_to_slot_;
    uint64_t clock_ = 0;
    CacheStats stats_;

    std::deque<ExpertRecord> prefetch_queue_;
    std::unordered_map<uint64_t, bool> queued_;
    std::thread worker_;
    bool stopping_ = false;
};

} // namespace redlite
