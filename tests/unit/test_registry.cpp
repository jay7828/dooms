#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "check.h"
#include "dooms/dooms.h"

using namespace dooms;

namespace {

class FakeModel : public Model {
public:
    std::string name() const override { return "fake"; }
    Status load(const std::string & path) override {
        if (path.empty()) return invalid_argument("empty path");
        loaded_ = true;
        return Status();
    }
    Status run(const TensorMap & in, TensorMap & out) override {
        if (!loaded_) return invalid_argument("not loaded");
        out = in;
        return Status();
    }

private:
    bool loaded_ = false;
};

ModelFactory fake_factory() { return [] { return std::make_unique<FakeModel>(); }; }

bool contains(const std::vector<std::string> & v, const std::string & s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}

}  // namespace

TEST(registry_add_create_and_list) {
    EXPECT(Registry::get().add("test_fake", fake_factory()).is_ok());
    auto m = Registry::get().create("test_fake");
    EXPECT(m.is_ok());
    if (!m.is_ok()) return;
    EXPECT_EQ(m.value()->name(), std::string("fake"));
    EXPECT(contains(Registry::get().ids(), "test_fake"));

    // The created model works through the interface.
    TensorMap in, out;
    EXPECT(m.value()->run(in, out).code() == Code::InvalidArgument);  // before load
    EXPECT(m.value()->load("anything").is_ok());
    in["x"] = {{2}, {1.f, 2.f}};
    EXPECT(m.value()->run(in, out).is_ok());
    EXPECT_EQ(out["x"].data.size(), size_t(2));
}

TEST(registry_rejects_bad_registrations) {
    EXPECT(Registry::get().add("test_dup", fake_factory()).is_ok());
    EXPECT(Registry::get().add("test_dup", fake_factory()).code() == Code::InvalidArgument);  // never replaced silently
    EXPECT(Registry::get().add("", fake_factory()).code() == Code::InvalidArgument);
    EXPECT(Registry::get().add("test_null_factory", ModelFactory()).code() == Code::InvalidArgument);
    EXPECT(!contains(Registry::get().ids(), "test_null_factory"));
}

TEST(registry_unknown_and_broken_factories) {
    EXPECT(Registry::get().create("test_no_such_model").status().code() == Code::NotFound);
    EXPECT(Registry::get().add("test_returns_null", [] { return std::unique_ptr<Model>(); }).is_ok());
    EXPECT(Registry::get().create("test_returns_null").status().code() == Code::Internal);
}

TEST(registry_ids_are_sorted) {
    EXPECT(Registry::get().add("test_sort_b", fake_factory()).is_ok());
    EXPECT(Registry::get().add("test_sort_a", fake_factory()).is_ok());
    auto ids = Registry::get().ids();
    EXPECT(std::is_sorted(ids.begin(), ids.end()));
}

TEST(registry_is_safe_under_concurrent_registration) {
    constexpr int kThreads = 8, kPerThread = 50;
    std::vector<std::thread> threads;
    std::vector<int> failures(kThreads, 0);
    for (int t = 0; t < kThreads; t++) {
        threads.emplace_back([t, &failures] {
            for (int i = 0; i < kPerThread; i++) {
                const std::string id = "test_mt_" + std::to_string(t) + "_" + std::to_string(i);
                if (!Registry::get().add(id, fake_factory()).is_ok()) failures[t]++;
                if (!Registry::get().create(id).is_ok()) failures[t]++;
            }
        });
    }
    for (auto & th : threads) th.join();
    for (int f : failures) EXPECT_EQ(f, 0);
    int found = 0;
    for (const std::string & id : Registry::get().ids()) found += id.rfind("test_mt_", 0) == 0;
    EXPECT_EQ(found, kThreads * kPerThread);
}

TEST(builtin_registration_is_idempotent) {
    EXPECT(register_builtin_models().is_ok());
    EXPECT(register_builtin_models().is_ok());  // second call must not fail on duplicate ids
#ifdef DOOMS_WITH_LLAMA
    EXPECT(contains(Registry::get().ids(), "llama"));
#else
    EXPECT(!contains(Registry::get().ids(), "llama"));
#endif
}

#ifdef DOOMS_WITH_LLAMA

// Opt-in: set DOOMS_TEST_GGUF to a small GGUF file to test a real load.
constexpr const char * kRealModelEnvVar = "DOOMS_TEST_GGUF";

TEST(llama_bridge_missing_file) {
    EXPECT(register_builtin_models().is_ok());
    auto m = Registry::get().create("llama");
    EXPECT(m.is_ok());
    if (!m.is_ok()) return;
    EXPECT(m.value()->load("/nonexistent/dooms/definitely_missing.gguf").code() == Code::NotFound);
}

TEST(llama_bridge_rejects_a_file_that_is_not_gguf) {
    EXPECT(register_builtin_models().is_ok());
    const auto path = std::filesystem::temp_directory_path() / "dooms_test_not_a_model.gguf";
    {
        std::ofstream f(path, std::ios::binary);
        f << "this is definitely not a GGUF file";
    }
    auto m = Registry::get().create("llama");
    EXPECT(m.is_ok());
    if (m.is_ok()) EXPECT(m.value()->load(path.string()).code() == Code::DataLoss);
    std::filesystem::remove(path);
}

TEST(llama_bridge_run_is_unsupported_until_m9) {
    EXPECT(register_builtin_models().is_ok());
    auto m = Registry::get().create("llama");
    EXPECT(m.is_ok());
    TensorMap out;
    if (m.is_ok()) EXPECT(m.value()->run({}, out).code() == Code::Unsupported);
}

TEST(llama_bridge_loads_real_model_when_provided) {
    const char * path = std::getenv(kRealModelEnvVar);
    if (path == nullptr) {
        std::printf("        (skipped: set %s to a GGUF file to run)\n", kRealModelEnvVar);
        return;
    }
    EXPECT(register_builtin_models().is_ok());
    auto m = Registry::get().create("llama");
    EXPECT(m.is_ok());
    if (!m.is_ok()) return;
    EXPECT(m.value()->load(path).is_ok());
    EXPECT(m.value()->load(path).code() == Code::InvalidArgument);  // already loaded
}

#endif  // DOOMS_WITH_LLAMA

int main() { return dooms_test::run_all(); }
