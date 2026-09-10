#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <map>
#include <tuple>
#include <vector>

namespace py = pybind11;

using Sequence = std::vector<long long>;
using MergeRule = std::tuple<long long, long long, long long>;

long long next_token_id(const std::vector<Sequence>& sequences) {
    long long max_token = -1;
    for (const auto& sequence : sequences) {
        for (long long token : sequence) {
            max_token = std::max(max_token, token);
        }
    }
    return max_token + 1;
}

std::map<std::pair<long long, long long>, long long> count_pairs(const std::vector<Sequence>& sequences) {
    std::map<std::pair<long long, long long>, long long> counts;
    for (const auto& sequence : sequences) {
        for (std::size_t i = 0; i + 1 < sequence.size(); ++i) {
            counts[{sequence[i], sequence[i + 1]}] += 1;
        }
    }
    return counts;
}

Sequence apply_merge(const Sequence& sequence, const MergeRule& rule) {
    auto [left, right, new_id] = rule;
    Sequence merged;
    for (std::size_t i = 0; i < sequence.size();) {
        if (i + 1 < sequence.size() && sequence[i] == left && sequence[i + 1] == right) {
            merged.push_back(new_id);
            i += 2;
        } else {
            merged.push_back(sequence[i]);
            i += 1;
        }
    }
    return merged;
}

py::dict train_bpe(const std::vector<Sequence>& input_sequences, int num_merges, int min_frequency,
                    long long initial_vocab_size) {
    std::vector<Sequence> sequences = input_sequences;
    std::vector<MergeRule> merge_rules;
    long long next_id = std::max(next_token_id(sequences), initial_vocab_size);

    for (int merge_index = 0; merge_index < num_merges; ++merge_index) {
        auto counts = count_pairs(sequences);
        if (counts.empty()) {
            break;
        }

        auto best = counts.begin();
        for (auto it = counts.begin(); it != counts.end(); ++it) {
            if (it->second > best->second ||
                (it->second == best->second && it->first < best->first)) {
                best = it;
            }
        }
        if (best->second < min_frequency) {
            break;
        }

        MergeRule rule{best->first.first, best->first.second, next_id};
        merge_rules.push_back(rule);
        for (auto& sequence : sequences) {
            sequence = apply_merge(sequence, rule);
        }
        next_id += 1;
    }

    py::dict metadata;
    metadata["num_merges_requested"] = num_merges;
    metadata["num_merges_performed"] = static_cast<int>(merge_rules.size());
    metadata["min_frequency"] = min_frequency;

    py::dict result;
    result["merge_rules"] = merge_rules;
    result["vocab_size"] = next_id;
    result["metadata"] = metadata;
    return result;
}

Sequence encode(const Sequence& sequence, const std::vector<MergeRule>& merge_rules) {
    Sequence encoded = sequence;
    for (const auto& rule : merge_rules) {
        encoded = apply_merge(encoded, rule);
    }
    return encoded;
}

std::vector<Sequence> batch_encode(
    const std::vector<Sequence>& sequences,
    const std::vector<MergeRule>& merge_rules) {
    std::vector<Sequence> encoded;
    encoded.reserve(sequences.size());
    for (const auto& sequence : sequences) {
        encoded.push_back(encode(sequence, merge_rules));
    }
    return encoded;
}

PYBIND11_MODULE(_graph_bpe, m) {
    m.doc() = "Native GraphTokenizer BPE backend";
    m.def("train_bpe", &train_bpe, py::arg("token_sequences"), py::arg("num_merges"),
          py::arg("min_frequency"), py::arg("initial_vocab_size") = -1);
    m.def("encode", &encode, py::arg("token_sequence"), py::arg("merge_rules"));
    m.def("batch_encode", &batch_encode, py::arg("token_sequences"), py::arg("merge_rules"));
}
