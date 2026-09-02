#include "red_stream_cache.hpp"

#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>

namespace redlite {

static std::vector<std::string> split(const std::string & s, char delim) {
    std::vector<std::string> out; std::stringstream ss(s); std::string item;
    while (std::getline(ss, item, delim)) out.push_back(item);
    return out;
}

std::vector<ExpertRecord> load_manifest(const std::string & path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open manifest: " + path);
    std::vector<ExpertRecord> out; std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto cols = split(line, '\t');
        if (cols.size() != 4) throw std::runtime_error("invalid manifest row");
        ExpertRecord rec;
        rec.layer = std::stoi(cols[0]); rec.expert = std::stoi(cols[1]); rec.total_bytes = std::stoull(cols[2]);
        for (const auto & raw : split(cols[3], ',')) {
            auto pair = split(raw, ':');
            if (pair.size() != 2) throw std::runtime_error("invalid manifest segment");
            rec.segments.push_back({std::stoull(pair[0]), std::stoull(pair[1])});
        }
        out.push_back(std::move(rec));
    }
    return out;
}

ExpertCache::ExpertCache(const std::string & model_path, const std::vector<ExpertRecord> & records,
                         uint64_t cache_bytes, bool nocache) {
    if (records.empty()) throw std::runtime_error("manifest has no expert records");
    slot_bytes_ = 0;
    for (const auto & r : records) slot_bytes_ = std::max(slot_bytes_, r.total_bytes);
    slot_bytes_ = (slot_bytes_ + 4095u) & ~4095ull;
    slots_ = cache_bytes / slot_bytes_;
    if (slots_ == 0) throw std::runtime_error("cache is smaller than one expert record");

    fd_ = ::open(model_path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open model");
#ifdef F_NOCACHE
    if (nocache) (void) ::fcntl(fd_, F_NOCACHE, 1);
#else
    (void) nocache;
#endif

    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) throw std::runtime_error("Metal device unavailable");
    const uint64_t target_buffer_bytes = 256ull * 1024ull * 1024ull;
    slots_per_buffer_ = std::max<uint64_t>(1, target_buffer_bytes / slot_bytes_);
    uint64_t remaining = slots_;
    while (remaining) {
        const uint64_t nslots = std::min(remaining, slots_per_buffer_);
        id<MTLBuffer> b = [device_ newBufferWithLength:nslots * slot_bytes_ options:MTLResourceStorageModeShared];
        if (!b) throw std::runtime_error("failed to allocate Metal shared expert cache chunk");
        buffers_.push_back(b);
        remaining -= nslots;
    }
    slot_state_.resize(static_cast<size_t>(slots_));
    worker_ = std::thread(&ExpertCache::worker_loop, this);
}

ExpertCache::~ExpertCache() {
    stop_prefetch();
    if (fd_ >= 0) ::close(fd_);
    buffers_.clear();
    device_ = nil;
}

size_t ExpertCache::reserve_slot_locked(uint64_t key) {
    size_t chosen = slot_state_.size();
    for (size_t i = 0; i < slot_state_.size(); ++i) if (slot_state_[i].state == SlotState::Empty) { chosen = i; break; }
    if (chosen == slot_state_.size()) {
        uint64_t best = UINT64_MAX;
        for (size_t i = 0; i < slot_state_.size(); ++i) {
            if (slot_state_[i].state == SlotState::Ready && slot_state_[i].stamp < best) {
                best = slot_state_[i].stamp; chosen = i;
            }
        }
    }
    if (chosen == slot_state_.size()) return chosen;
    auto & slot = slot_state_[chosen];
    if (slot.state != SlotState::Empty) { key_to_slot_.erase(slot.key); stats_.evictions++; }
    slot.key = key; slot.stamp = ++clock_; slot.state = SlotState::Loading; key_to_slot_[key] = chosen;
    return chosen;
}

id<MTLBuffer> ExpertCache::buffer_for_slot(size_t slot) const { return buffers_.at(slot / slots_per_buffer_); }
uint64_t ExpertCache::buffer_offset_for_slot(size_t slot) const { return (slot % slots_per_buffer_) * slot_bytes_; }

bool ExpertCache::load_into_slot(const ExpertRecord & rec, size_t slot_index) {
    id<MTLBuffer> b = buffer_for_slot(slot_index);
    auto * base = static_cast<uint8_t *>([b contents]) + buffer_offset_for_slot(slot_index);
    uint64_t dst = 0, bytes = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (const auto & seg : rec.segments) {
        uint64_t done = 0;
        while (done < seg.length) {
            const ssize_t n = ::pread(fd_, base + dst + done, static_cast<size_t>(seg.length - done), static_cast<off_t>(seg.offset + done));
            if (n <= 0) return false;
            done += static_cast<uint64_t>(n);
        }
        dst += seg.length; bytes += seg.length;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    { std::lock_guard<std::mutex> lock(mu_); stats_.bytes_read += bytes; stats_.read_ns += ns; }
    return true;
}

bool ExpertCache::ensure(const ExpertRecord & rec, bool count_request) {
    const uint64_t key = rec.key(); size_t slot_index = 0;
    {
        std::unique_lock<std::mutex> lock(mu_);
        if (count_request) stats_.requests++;
        for (;;) {
            auto it = key_to_slot_.find(key);
            if (it != key_to_slot_.end()) {
                auto & slot = slot_state_[it->second];
                if (slot.state == SlotState::Ready) {
                    slot.stamp = ++clock_; if (count_request) stats_.hits++; return true;
                }
                if (slot.state == SlotState::Loading) {
                    const size_t waiting_slot = it->second;
                    cv_.wait(lock, [&] { return slot_state_[waiting_slot].state != SlotState::Loading; });
                    continue;
                }
            }
            if (count_request) stats_.misses++;
            slot_index = reserve_slot_locked(key);
            if (slot_index != slot_state_.size()) break;
            cv_.wait(lock);
        }
    }
    const bool ok = load_into_slot(rec, slot_index);
    {
        std::lock_guard<std::mutex> lock(mu_); auto & slot = slot_state_[slot_index];
        if (ok) { slot.state = SlotState::Ready; slot.stamp = ++clock_; }
        else { key_to_slot_.erase(key); slot = Slot{}; }
    }
    cv_.notify_all(); return ok;
}

void ExpertCache::prefetch(const ExpertRecord & rec) {
    std::lock_guard<std::mutex> lock(mu_); const uint64_t key = rec.key();
    if (key_to_slot_.count(key) || queued_.count(key)) return;
    queued_[key] = true; prefetch_queue_.push_back(rec); cv_.notify_all();
}

void ExpertCache::worker_loop() {
    for (;;) {
        ExpertRecord rec;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [&] { return stopping_ || !prefetch_queue_.empty(); });
            if (stopping_ && prefetch_queue_.empty()) return;
            rec = std::move(prefetch_queue_.front()); prefetch_queue_.pop_front(); queued_.erase(rec.key());
        }
        (void) ensure(rec, false);
    }
}

void ExpertCache::stop_prefetch() {
    { std::lock_guard<std::mutex> lock(mu_); if (stopping_) return; stopping_ = true; }
    cv_.notify_all(); if (worker_.joinable()) worker_.join();
}

CacheStats ExpertCache::stats() const { std::lock_guard<std::mutex> lock(mu_); return stats_; }

} // namespace redlite
