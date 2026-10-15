#pragma once
// The vocabulary of a language model without its weights (llama.cpp vocab-only load of a GGUF written by `convert_hf_to_gguf.py --vocab-only`): turns task text
// into the padded token ids a policy expects. Everything that touches llama.h lives in tokenizer.cpp.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/status.h"

namespace dooms {

struct PaddedTokens {
    std::vector<int32_t> tokens;     // exactly max_length ids, right-padded with the vocabulary's padding id
    std::vector<uint8_t> mask;       // 1 for real tokens, 0 for padding
    bool truncated = false;          // the text was longer than max_length
};

class Tokenizer {
public:
    //   NotFound  no such file     DataLoss  not a GGUF vocabulary llama.cpp can load, or it has no padding token
    static Result<Tokenizer> load(const std::string & path);

    Tokenizer(Tokenizer &&) noexcept;
    Tokenizer & operator=(Tokenizer &&) noexcept;
    ~Tokenizer();

    int64_t vocab_size() const;
    int32_t pad_id() const;

    // The text exactly as the policy's pipeline sees it: a trailing newline is added if missing (the SmolVLA "new line" step), tokenized without a BOS token,
    // special-token names in the text are NOT interpreted, truncated/padded on the right to max_length.
    //   InvalidArgument  empty text, max_length outside [1, 4096]
    Result<PaddedTokens> encode_task(const std::string & text, int64_t max_length) const;

private:
    struct Impl;
    explicit Tokenizer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace dooms
