#import <Foundation/Foundation.h>
#include "red_stream_cache.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static std::string need(int & i, int argc, char ** argv) {
    if (++i >= argc) throw std::runtime_error("missing argument value");
    return argv[i];
}

int main(int argc, char ** argv) {
    @autoreleasepool {
        try {
            std::string model, manifest;
            uint64_t cache_mib = 1024, seed = 42;
            size_t requests = 5000, hotset = 1024, prefetch = 4;
            bool nocache = false;
            for (int i = 1; i < argc; ++i) {
                std::string a = argv[i];
                if (a == "--model") model = need(i, argc, argv);
                else if (a == "--manifest") manifest = need(i, argc, argv);
                else if (a == "--cache-mib") cache_mib = std::stoull(need(i, argc, argv));
                else if (a == "--requests") requests = std::stoull(need(i, argc, argv));
                else if (a == "--hotset") hotset = std::stoull(need(i, argc, argv));
                else if (a == "--prefetch") prefetch = std::stoull(need(i, argc, argv));
                else if (a == "--seed") seed = std::stoull(need(i, argc, argv));
                else if (a == "--nocache") nocache = true;
                else throw std::runtime_error("unknown argument: " + a);
            }
            if (model.empty() || manifest.empty()) throw std::runtime_error("--model and --manifest are required");

            auto records = redlite::load_manifest(manifest);
            if (records.empty()) throw std::runtime_error("manifest has no records");
            hotset = std::max<size_t>(1, std::min(hotset, records.size()));

            std::mt19937_64 rng(seed);
            std::uniform_int_distribution<size_t> hot_dist(0, hotset - 1), all_dist(0, records.size() - 1);
            std::uniform_int_distribution<int> pct(0, 99);
            std::vector<size_t> seq; seq.reserve(requests);
            for (size_t i = 0; i < requests; ++i) seq.push_back(pct(rng) < 80 ? hot_dist(rng) : all_dist(rng));

            redlite::ExpertCache cache(model, records, cache_mib * 1024ull * 1024ull, nocache);
            std::cout << "red-stream alpha probe\n"
                      << "records        : " << records.size() << "\n"
                      << "cache MiB      : " << cache_mib << "\n"
                      << "slot KiB       : " << (cache.slot_bytes() / 1024.0) << "\n"
                      << "cache slots    : " << cache.slots() << "\n"
                      << "requests       : " << requests << "\n"
                      << "hotset         : " << hotset << "\n"
                      << "prefetch depth : " << prefetch << "\n"
                      << "F_NOCACHE      : " << (nocache ? "on" : "off") << "\n";

            const auto start = std::chrono::steady_clock::now(); size_t completed = 0;
            for (size_t i = 0; i < seq.size(); ++i) {
                for (size_t j = 1; j <= prefetch && i + j < seq.size(); ++j) cache.prefetch(records[seq[i + j]]);
                if (!cache.ensure(records[seq[i]])) throw std::runtime_error("expert pread failed");
                completed++;
            }
            cache.stop_prefetch();
            const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            auto s = cache.stats();
            const double hit_rate = s.requests ? 100.0 * s.hits / s.requests : 0.0;
            const double read_s = s.read_ns / 1e9;
            const double gib = s.bytes_read / double(1024ull * 1024ull * 1024ull);
            std::cout << "completed      : " << completed << "\n"
                      << "hits/misses    : " << s.hits << "/" << s.misses << "\n"
                      << "hit rate       : " << hit_rate << "%\n"
                      << "evictions      : " << s.evictions << "\n"
                      << "SSD bytes read : " << gib << " GiB\n"
                      << "pread time     : " << read_s << " s\n"
                      << "pread GiB/s    : " << (read_s > 0 ? gib / read_s : 0.0) << "\n"
                      << "wall time      : " << wall << " s\n"
                      << "requests/s     : " << (wall > 0 ? completed / wall : 0.0) << "\n";
            return 0;
        } catch (const std::exception & e) {
            std::cerr << "red-stream: " << e.what() << "\n";
            return 2;
        }
    }
}
