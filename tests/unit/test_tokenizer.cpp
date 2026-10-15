// Tests the vocabulary-only tokenizer: argument validation and padding rules (always, when a vocabulary exists: DOOMS_VLA_DIR/vocab.gguf), and the task tokens
// against the ones lerobot's processor produced (DOOMS_VLA_DIR/reference.gguf).
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "check.h"
#include "format/gguf.h"
#include "models/vlm/tokenizer.h"

using namespace dooms;

namespace {
std::string vla_dir() {
    const char * d = std::getenv("DOOMS_VLA_DIR");
    return d ? d : "";
}
}  // namespace

TEST(load_rejects_missing_and_wrong_files) {
    EXPECT(!Tokenizer::load("/nonexistent/vocab.gguf").is_ok());
    const std::string dir = vla_dir();
    if (dir.empty()) { std::printf("        (skipped: set DOOMS_VLA_DIR)\n"); return; }
    EXPECT(!Tokenizer::load(dir + "/host.gguf").is_ok());  // a GGUF without a vocabulary
}

TEST(encode_validates_and_pads) {
    const std::string dir = vla_dir();
    if (dir.empty()) { std::printf("        (skipped: set DOOMS_VLA_DIR)\n"); return; }
    auto tok = Tokenizer::load(dir + "/vocab.gguf");
    EXPECT(tok.is_ok());
    if (!tok.is_ok()) return;
    EXPECT(!tok.value().encode_task("", 48).is_ok());
    EXPECT(!tok.value().encode_task("go", 0).is_ok());
    EXPECT(!tok.value().encode_task("go", 5000).is_ok());
    auto a = tok.value().encode_task("go", 8);
    auto b = tok.value().encode_task("go\n", 8);  // an existing newline is not doubled
    EXPECT(a.is_ok() && b.is_ok());
    if (a.is_ok() && b.is_ok()) {
        EXPECT(a.value().tokens == b.value().tokens);
        EXPECT_EQ(a.value().tokens.size(), static_cast<size_t>(8));
        int real = 0;
        for (size_t i = 0; i < 8; i++) {
            real += a.value().mask[i];
            if (!a.value().mask[i]) EXPECT_EQ(a.value().tokens[i], tok.value().pad_id());
        }
        EXPECT(real >= 2 && real < 8);
        EXPECT(!a.value().truncated);
    }
    auto t = tok.value().encode_task("one two three four five six seven eight nine ten", 4);
    EXPECT(t.is_ok());
    if (t.is_ok()) { EXPECT(t.value().truncated); EXPECT_EQ(t.value().mask[3], static_cast<uint8_t>(1)); }
}

TEST(task_tokens_equal_the_policy_processors) {
    const std::string dir = vla_dir();
    if (dir.empty()) { std::printf("        (skipped: set DOOMS_VLA_DIR)\n"); return; }
    auto tok = Tokenizer::load(dir + "/vocab.gguf");
    auto ref = GgufFile::open(dir + "/reference.gguf");
    EXPECT(tok.is_ok() && ref.is_ok());
    if (!tok.is_ok() || !ref.is_ok()) return;
    auto ids = ref.value().find_tensor("tokens");
    auto mask = ref.value().find_tensor("token_mask");
    auto prompt = ref.value().get_string("task");
    EXPECT(ids.is_ok() && mask.is_ok() && prompt.is_ok());
    if (!ids.is_ok() || !mask.is_ok() || !prompt.is_ok()) return;
    const float * want_ids = reinterpret_cast<const float *>(ref.value().tensor_data(*ids.value()));
    const float * want_mask = reinterpret_cast<const float *>(ref.value().tensor_data(*mask.value()));
    const int64_t n = ids.value()->shape.numel();
    auto got = tok.value().encode_task(prompt.value(), n);
    EXPECT(got.is_ok());
    if (!got.is_ok()) return;
    int64_t different = 0;
    for (int64_t i = 0; i < n; i++) different += got.value().tokens[static_cast<size_t>(i)] != static_cast<int32_t>(want_ids[i]) || got.value().mask[static_cast<size_t>(i)] != (want_mask[i] != 0.f ? 1 : 0);
    std::printf("        \"%s\": %lld of %lld token/mask entries differ from the policy processor\n", prompt.value().c_str(), static_cast<long long>(different), static_cast<long long>(n));
    EXPECT_EQ(different, int64_t(0));
}

int main() { return dooms_test::run_all(); }
