#include "gemv.h"
#include "utils/handle.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#if defined(HAVE_MLU)
#include <cnrt.h>
#endif

#if defined(GEMV_ENABLE_CNBLAS)
#include <cnblas.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    int hidden = 256;
    int intermediate = 512;
    int layers = 1;
    int tokens = 1000;
    int warmup = 10;
    int repeat = 3;
    int seed = 20260925;
    int max_report_tokens = 0;
    std::string impl = "tile_sram_db";
    std::string backend = "all";
    std::string report = "build/app/decoder_decode_report.md";
    std::string csv = "build/app/decoder_decode_report.csv";
    bool verify = true;
};

struct Timing {
    double total_ms = 0.0;
    double gemv_ms = 0.0;
    double attention_ms = 0.0;
    double other_ms = 0.0;
    int tokens = 0;

    Timing &operator+=(const Timing &rhs) {
        total_ms += rhs.total_ms;
        gemv_ms += rhs.gemv_ms;
        attention_ms += rhs.attention_ms;
        other_ms += rhs.other_ms;
        tokens += rhs.tokens;
        return *this;
    }
};

struct Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<float> data;

    Matrix() = default;
    Matrix(int r, int c) : rows(r), cols(c), data(static_cast<size_t>(r) * c) {}

    float *row(int index) {
        return data.data() + static_cast<size_t>(index) * cols;
    }

    const float *row(int index) const {
        return data.data() + static_cast<size_t>(index) * cols;
    }
};

struct LayerWeights {
    Matrix q;
    Matrix k;
    Matrix v;
    Matrix o;
    Matrix up;
    Matrix gate;
    Matrix down;
};

struct LayerCache {
    std::vector<std::vector<float>> keys;
    std::vector<std::vector<float>> values;
};

struct Model {
    Options options;
    int head_dim = 0;
    std::vector<LayerWeights> weights;
};

struct RunResult {
    Timing timing;
    bool verified = true;
    std::vector<float> final_state;
};

struct Backend {
    std::string name;
    std::string gemv_impl;
    bool cnblas = false;
};

static void usage(const char *program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --hidden N              Hidden size, default 256\n"
        << "  --intermediate N        FFN intermediate size, default 512\n"
        << "  --layers N              Decoder layers, default 1\n"
        << "  --tokens N              Decode token count, default 1000\n"
        << "  --warmup N              Warmup tokens, default 10\n"
        << "  --repeat N              Repeated runs, default 3\n"
        << "  --seed N                Fixed random seed\n"
        << "  --backend NAME          all|mlu|cnblas, default all\n"
        << "  --impl NAME             GEMV_IMPL, compare, or all seven MLU kernels\n"
        << "  --report PATH           Markdown report path\n"
        << "  --csv PATH              CSV report path\n"
        << "  --no-verify             Skip one-token CPU verification\n"
        << "  --help                  Show this help\n";
}

static bool parse_int(const std::string &text, int *value) {
    if (!value || text.empty()) {
        return false;
    }
    char *end = nullptr;
    long parsed = std::strtol(text.c_str(), &end, 10);
    if (*end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

static bool parse_options(int argc, char **argv, Options *options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help") {
            usage(argv[0]);
            return false;
        }
        if (arg == "--no-verify") {
            options->verify = false;
            continue;
        }
        if (arg.rfind("--", 0) != 0 || i + 1 >= argc) {
            std::cerr << "Invalid argument: " << arg << "\n";
            return false;
        }
        const std::string key = arg.substr(2);
        const std::string value(argv[++i]);
        int parsed = 0;
        if (key == "backend") {
            options->backend = value;
        } else if (key == "impl") {
            options->impl = value;
        } else if (key == "report") {
            options->report = value;
        } else if (key == "csv") {
            options->csv = value;
        } else if (key == "hidden" && parse_int(value, &parsed)) {
            options->hidden = parsed;
        } else if (key == "intermediate" && parse_int(value, &parsed)) {
            options->intermediate = parsed;
        } else if (key == "layers" && parse_int(value, &parsed)) {
            options->layers = parsed;
        } else if (key == "tokens" && parse_int(value, &parsed)) {
            options->tokens = parsed;
        } else if (key == "warmup" && parse_int(value, &parsed)) {
            options->warmup = parsed;
        } else if (key == "repeat" && parse_int(value, &parsed)) {
            options->repeat = parsed;
        } else if (key == "seed" && parse_int(value, &parsed)) {
            options->seed = parsed;
        } else {
            std::cerr << "Invalid option/value: " << arg << " " << value << "\n";
            return false;
        }
    }

    if (options->backend != "all" && options->backend != "mlu" &&
        options->backend != "cnblas") {
        std::cerr << "--backend must be all, mlu, or cnblas\n";
        return false;
    }
    if (options->hidden <= 0 || options->intermediate <= 0 || options->layers <= 0 ||
        options->tokens <= 0 || options->warmup < 0 || options->repeat <= 0) {
        std::cerr << "All sizes must be positive and warmup must be non-negative\n";
        return false;
    }
    return true;
}

static float uniform_value(std::mt19937 *rng, float scale) {
    std::uniform_real_distribution<float> distribution(-scale, scale);
    return distribution(*rng);
}

static Matrix random_matrix(int rows, int cols, std::mt19937 *rng, float scale) {
    Matrix matrix(rows, cols);
    for (float &value : matrix.data) {
        value = uniform_value(rng, scale);
    }
    return matrix;
}

static Model create_model(const Options &options) {
    Model model;
    model.options = options;
    model.head_dim = options.hidden;
    std::mt19937 rng(static_cast<uint32_t>(options.seed));
    const float projection_scale = 0.02f;

    model.weights.reserve(options.layers);
    for (int layer = 0; layer < options.layers; ++layer) {
        LayerWeights weights;
        weights.q = random_matrix(options.hidden, options.hidden, &rng, projection_scale);
        weights.k = random_matrix(options.hidden, options.hidden, &rng, projection_scale);
        weights.v = random_matrix(options.hidden, options.hidden, &rng, projection_scale);
        weights.o = random_matrix(options.hidden, options.hidden, &rng, projection_scale);
        weights.up = random_matrix(options.intermediate, options.hidden, &rng, projection_scale);
        weights.gate = random_matrix(options.intermediate, options.hidden, &rng, projection_scale);
        weights.down = random_matrix(options.hidden, options.intermediate, &rng, projection_scale);
        model.weights.push_back(std::move(weights));
    }
    return model;
}

static std::vector<float> initial_state(int hidden, int seed) {
    std::mt19937 rng(static_cast<uint32_t>(seed) ^ 0x9e3779b9U);
    std::vector<float> state(hidden);
    for (float &value : state) {
        value = uniform_value(&rng, 0.5f);
    }
    return state;
}

static void matvec_cpu(const Matrix &matrix, const std::vector<float> &input,
                       std::vector<float> *output) {
    output->assign(matrix.rows, 0.0f);
    for (int row = 0; row < matrix.rows; ++row) {
        float sum = 0.0f;
        const float *weights = matrix.row(row);
        for (int col = 0; col < matrix.cols; ++col) {
            sum += weights[col] * input[col];
        }
        (*output)[row] = sum;
    }
}

static void add_inplace(std::vector<float> *lhs, const std::vector<float> &rhs) {
    for (size_t i = 0; i < lhs->size(); ++i) {
        (*lhs)[i] += rhs[i];
    }
}

static void swish_inplace(std::vector<float> *values) {
    for (float &value : *values) {
        value = value / (1.0f + std::exp(-value));
    }
}

static std::vector<float> attention_cpu(const std::vector<float> &query,
                                        const LayerCache &cache, int head_dim) {
    if (cache.keys.empty()) {
        return std::vector<float>(query.size(), 0.0f);
    }

    std::vector<float> scores(cache.keys.size(), 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    float max_score = -std::numeric_limits<float>::infinity();
    for (size_t token = 0; token < cache.keys.size(); ++token) {
        float score = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            score += query[i] * cache.keys[token][i];
        }
        scores[token] = score * scale;
        max_score = std::max(max_score, scores[token]);
    }

    float denominator = 0.0f;
    for (float &score : scores) {
        score = std::exp(score - max_score);
        denominator += score;
    }

    std::vector<float> result(query.size(), 0.0f);
    for (size_t token = 0; token < cache.values.size(); ++token) {
        const float weight = scores[token] / denominator;
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] += weight * cache.values[token][i];
        }
    }
    return result;
}

static void update_next_state(std::vector<float> *state) {
    for (float &value : *state) {
        value = std::tanh(value);
    }
}

static bool nearly_equal(const std::vector<float> &expected, const std::vector<float> &actual,
                         float tolerance, float *max_error) {
    if (expected.size() != actual.size()) {
        return false;
    }
    float observed = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
        observed = std::max(observed, std::fabs(expected[i] - actual[i]));
    }
    if (max_error) {
        *max_error = observed;
    }
    return observed <= tolerance;
}

class ProjectionRunner {
  public:
    ProjectionRunner(DnHandle_t handle, const Backend &backend) : handle_(handle), backend_(backend) {
#if defined(GEMV_ENABLE_CNBLAS)
        if (backend_.cnblas) {
            if (cnblasCreate(&cnblas_handle_) != CNBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("cnblasCreate failed");
            }
            void *stream = nullptr;
            if (dnGetStream(handle_, &stream) != DN_SUCCESS ||
                cnblasSetQueue(cnblas_handle_, static_cast<cnrtQueue_t>(stream)) !=
                    CNBLAS_STATUS_SUCCESS) {
                cnblasDestroy(cnblas_handle_);
                cnblas_handle_ = nullptr;
                throw std::runtime_error("cnblasSetQueue failed");
            }
        }
#else
        if (backend_.cnblas) {
            throw std::runtime_error("cnBLAS support was not compiled");
        }
#endif
    }

    ~ProjectionRunner() {
#if defined(GEMV_ENABLE_CNBLAS)
        if (cnblas_handle_) {
            cnblasDestroy(cnblas_handle_);
        }
#endif
    }

    double project(const Matrix &weights, const std::vector<float> &input,
                   std::vector<float> *output) {
        output->assign(weights.rows, 0.0f);
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const auto begin = Clock::now();

        if (!backend_.cnblas) {
            const DnStatus_t status =
                BLAS_sGEMV(handle_, DnBLAS_OP_T, weights.cols, weights.rows, &alpha,
                           weights.data.data(), weights.cols, input.data(), 1, &beta,
                           output->data(), 1);
            if (status != DN_SUCCESS) {
                std::ostringstream message;
                message << "BLAS_sGEMV failed with status " << status;
                throw std::runtime_error(message.str());
            }
        } else {
#if defined(GEMV_ENABLE_CNBLAS)
            const cnblasStatus_t status =
                cnblasSgemv(cnblas_handle_, CNBLAS_OP_T, weights.cols, weights.rows, &alpha,
                            weights.data.data(), weights.cols, input.data(), 1, &beta,
                            output->data(), 1);
            if (status != CNBLAS_STATUS_SUCCESS) {
                std::ostringstream message;
                message << "cnblasSgemv failed with status " << status;
                throw std::runtime_error(message.str());
            }
            if (dnSynchronize(handle_) != DN_SUCCESS) {
                throw std::runtime_error("cnBLAS queue synchronization failed");
            }
#endif
        }
        return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    }

  private:
    DnHandle_t handle_ = nullptr;
    Backend backend_;
#if defined(GEMV_ENABLE_CNBLAS)
    cnblasHandle_t cnblas_handle_ = nullptr;
#endif
};

static std::vector<float> run_cpu_layer(const LayerWeights &weights,
                                        const std::vector<float> &input,
                                        LayerCache *cache, int head_dim) {
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    matvec_cpu(weights.q, input, &query);
    matvec_cpu(weights.k, input, &key);
    matvec_cpu(weights.v, input, &value);
    cache->keys.push_back(key);
    cache->values.push_back(value);
    const std::vector<float> context = attention_cpu(query, *cache, head_dim);

    std::vector<float> projected;
    matvec_cpu(weights.o, context, &projected);
    std::vector<float> hidden = input;
    add_inplace(&hidden, projected);

    std::vector<float> up;
    std::vector<float> gate;
    matvec_cpu(weights.up, hidden, &up);
    matvec_cpu(weights.gate, hidden, &gate);
    swish_inplace(&gate);
    for (size_t i = 0; i < up.size(); ++i) {
        up[i] *= gate[i];
    }

    std::vector<float> down;
    matvec_cpu(weights.down, up, &down);
    add_inplace(&hidden, down);
    return hidden;
}

static std::vector<float> run_mlu_layer(const LayerWeights &weights,
                                        const std::vector<float> &input,
                                        LayerCache *cache, int head_dim,
                                        ProjectionRunner *runner, Timing *timing) {
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    timing->gemv_ms += runner->project(weights.q, input, &query);
    timing->gemv_ms += runner->project(weights.k, input, &key);
    timing->gemv_ms += runner->project(weights.v, input, &value);
    cache->keys.push_back(key);
    cache->values.push_back(value);

    const auto attention_begin = Clock::now();
    const std::vector<float> context = attention_cpu(query, *cache, head_dim);
    timing->attention_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - attention_begin).count();

    std::vector<float> projected;
    timing->gemv_ms += runner->project(weights.o, context, &projected);
    std::vector<float> hidden = input;
    add_inplace(&hidden, projected);

    std::vector<float> up;
    std::vector<float> gate;
    timing->gemv_ms += runner->project(weights.up, hidden, &up);
    timing->gemv_ms += runner->project(weights.gate, hidden, &gate);
    swish_inplace(&gate);
    for (size_t i = 0; i < up.size(); ++i) {
        up[i] *= gate[i];
    }

    std::vector<float> down;
    timing->gemv_ms += runner->project(weights.down, up, &down);
    add_inplace(&hidden, down);
    return hidden;
}

static RunResult run_backend(const Model &model, const Backend &backend, bool verify) {
    DnHandle_t handle = nullptr;
    if (dnCreate(&handle) != DN_SUCCESS) {
        throw std::runtime_error("dnCreate failed");
    }

    try {
        if (!backend.cnblas) {
            ::setenv("GEMV_IMPL", backend.gemv_impl.c_str(), 1);
        }
        ProjectionRunner runner(handle, backend);
        std::vector<float> state = initial_state(model.options.hidden, model.options.seed);
        std::vector<LayerCache> caches(model.options.layers);

        for (int token = 0; token < model.options.warmup; ++token) {
            for (int layer = 0; layer < model.options.layers; ++layer) {
                Timing ignored;
                state = run_mlu_layer(model.weights[layer], state, &caches[layer],
                                      model.head_dim, &runner, &ignored);
            }
            update_next_state(&state);
        }

        state = initial_state(model.options.hidden, model.options.seed);
        caches.assign(model.options.layers, LayerCache());
        RunResult result;
        if (verify) {
            std::vector<LayerCache> reference_caches(model.options.layers);
            std::vector<float> reference_state = state;
            for (int layer = 0; layer < model.options.layers; ++layer) {
                reference_state =
                    run_cpu_layer(model.weights[layer], reference_state,
                                  &reference_caches[layer], model.head_dim);
            }
            for (int layer = 0; layer < model.options.layers; ++layer) {
                Timing ignored;
                state = run_mlu_layer(model.weights[layer], state, &caches[layer],
                                      model.head_dim, &runner, &ignored);
            }
            float max_error = 0.0f;
            result.verified = nearly_equal(reference_state, state, 1e-3f, &max_error);
            if (!result.verified) {
                std::cerr << "[verify] " << backend.name << " max_abs_error=" << max_error
                          << "\n";
            }
            update_next_state(&state);
        }

        state = initial_state(model.options.hidden, model.options.seed);
        caches.assign(model.options.layers, LayerCache());
        const auto begin = Clock::now();
        for (int token = 0; token < model.options.tokens; ++token) {
            const auto token_begin = Clock::now();
            for (int layer = 0; layer < model.options.layers; ++layer) {
                state = run_mlu_layer(model.weights[layer], state, &caches[layer],
                                      model.head_dim, &runner, &result.timing);
            }
            update_next_state(&state);
            result.timing.other_ms +=
                std::chrono::duration<double, std::milli>(Clock::now() - token_begin).count();
            result.timing.tokens++;
        }
        result.timing.total_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
        result.timing.other_ms =
            std::max(0.0, result.timing.other_ms - result.timing.gemv_ms -
                              result.timing.attention_ms);
        result.final_state = state;
        dnDestroy(handle);
        return result;
    } catch (...) {
        dnDestroy(handle);
        throw;
    }
}

static double per_token(double milliseconds, int tokens) {
    return tokens > 0 ? milliseconds / static_cast<double>(tokens) : 0.0;
}

static void ensure_parent_directory(const std::string &path) {
    const size_t slash = path.find_last_of('/');
    if (slash == std::string::npos) {
        return;
    }
    const std::string directory = path.substr(0, slash);
    if (directory.empty()) {
        return;
    }
    std::string command = "mkdir -p \"" + directory + "\"";
    std::system(command.c_str());
}

static void write_report(const Options &options, const std::vector<Backend> &backends,
                         const std::vector<Timing> &timings,
                         const std::vector<bool> &verified) {
    ensure_parent_directory(options.report);
    ensure_parent_directory(options.csv);
    std::ofstream markdown(options.report);
    std::ofstream csv(options.csv);
    if (!markdown || !csv) {
        throw std::runtime_error("failed to create decoder report");
    }

    markdown << "# Decoder Decode GEMV Profiling Report\n\n";
    markdown << "- hidden: `" << options.hidden << "`\n";
    markdown << "- intermediate: `" << options.intermediate << "`\n";
    markdown << "- layers: `" << options.layers << "`\n";
    markdown << "- measured tokens per repeat: `" << options.tokens << "`\n";
    markdown << "- warmup tokens: `" << options.warmup << "`\n";
    markdown << "- repeats: `" << options.repeat << "`\n";
    markdown << "- seed: `" << options.seed << "`\n";
#if defined(HAVE_MLU)
#if defined(GEMV_MLU_ARCH)
    markdown << "- configured MLU architecture: `mtp_" << GEMV_MLU_ARCH << "`\n";
#else
    markdown << "- configured MLU architecture: `unknown`\n";
#endif
    markdown << "- CNRT version: `" << CNRT_MAJOR_VERSION << "." << CNRT_MINOR_VERSION
             << "." << CNRT_PATCH_VERSION << "`\n";
#if defined(GEMV_BUILD_NRAM_CHUNK_FLOATS)
    markdown << "- GEMV_NRAM_CHUNK_FLOATS: `" << GEMV_BUILD_NRAM_CHUNK_FLOATS << "`\n";
    markdown << "- GEMV_TILE_SRAM_BLOCK_ROWS: `" << GEMV_BUILD_TILE_SRAM_BLOCK_ROWS << "`\n";
    markdown << "- GEMV_UNROLL_FACTOR: `" << GEMV_BUILD_UNROLL_FACTOR << "`\n";
#endif
#else
    markdown << "- configured accelerator runtime: `non-MLU build`\n";
#endif
    markdown << "- note: attention and KV cache are CPU reference paths in this prototype; "
                "all single-row projections use GEMV.\n\n";

    markdown << "## End-to-end latency\n\n";
    markdown << "| backend | avg total ms | ms/token | speedup | GEMV % | attention % | other % | verify |\n";
    markdown << "|---|---:|---:|---:|---:|---:|---:|---|\n";
    csv << "backend,total_ms,ms_per_token,speedup_vs_baseline,gemv_ms,attention_ms,other_ms,gemv_pct,attention_pct,other_pct,verify\n";

    double baseline_ms_per_token = 0.0;
    if (!timings.empty()) {
        baseline_ms_per_token =
            per_token(timings[0].total_ms / options.repeat, options.tokens);
    }

    for (size_t i = 0; i < backends.size(); ++i) {
        const Timing &timing = timings[i];
        const double total = timing.total_ms / options.repeat;
        const double gemv = timing.gemv_ms / options.repeat;
        const double attention = timing.attention_ms / options.repeat;
        const double other = timing.other_ms / options.repeat;
        const double total_per_token = per_token(total, options.tokens);
        const double speedup =
            total_per_token > 0.0 ? baseline_ms_per_token / total_per_token : 0.0;
        const double gemv_pct = total > 0.0 ? 100.0 * gemv / total : 0.0;
        const double attention_pct = total > 0.0 ? 100.0 * attention / total : 0.0;
        const double other_pct = total > 0.0 ? 100.0 * other / total : 0.0;
        markdown << "| " << backends[i].name << " | " << std::fixed << std::setprecision(4)
                 << total << " | " << total_per_token << " | " << speedup << " | "
                 << gemv_pct << " | " << attention_pct << " | " << other_pct << " | "
                 << (verified[i] ? "PASS" : "FAIL") << " |\n";
        csv << backends[i].name << "," << total << "," << total_per_token << "," << speedup
            << "," << gemv << "," << attention << "," << other << "," << gemv_pct << ","
            << attention_pct << "," << other_pct << ","
            << (verified[i] ? "PASS" : "FAIL") << "\n";
    }

    markdown << "\n## Interpretation\n\n";
    markdown << "- GEMV time includes the host-to-device/device-to-host overhead inside the current "
                "BLAS_sGEMV wrapper and the measured MLU kernel path.\n";
    markdown << "- `tile_sram_db` is selected through `GEMV_IMPL` for the optimized MLU path.\n";
    markdown << "- cnBLAS is reported only when the build was configured with "
                "`-DGEMV_ENABLE_CNBLAS=ON` and a usable cnBLAS installation.\n";
    markdown << "- Prefill-vs-decode GEMM comparison is not measured in this phase because the "
                "repository currently provides GEMV but no GEMM implementation; this is an "
                "explicit limitation of the prototype.\n";
}

static std::vector<Backend> select_backends(const Options &options) {
    std::vector<Backend> backends;
    if (options.backend == "all" || options.backend == "mlu") {
        const bool compare = options.impl == "compare";
        const bool all = options.impl == "all";
        const std::vector<std::string> implementations =
            compare ? std::vector<std::string>{"baseline", "tile_sram_db"}
            : all ? std::vector<std::string>{"baseline", "tile_nram", "tile_nram_db",
                                             "tile_sram", "tile_sram_db",
                                             "tile_sram_xreuse", "blas_style"}
                  : std::vector<std::string>{options.impl};
        for (const std::string &implementation : implementations) {
            Backend backend;
            backend.name = "mlu_" + implementation;
            backend.gemv_impl = implementation;
            backend.cnblas = false;
            backends.push_back(backend);
        }
    }
    if (options.backend == "all" || options.backend == "cnblas") {
#if defined(GEMV_ENABLE_CNBLAS)
        Backend backend;
        backend.name = "cnblas_sgemv";
        backend.cnblas = true;
        backends.push_back(backend);
#else
        if (options.backend == "cnblas") {
            throw std::runtime_error(
                "cnBLAS backend requested, but this build has no cnBLAS support");
        }
        std::cerr << "[info] cnBLAS backend unavailable; skipping it\n";
#endif
    }
    return backends;
}

}  // namespace

int main(int argc, char **argv) {
    Options options;
    if (!parse_options(argc, argv, &options)) {
        return argc > 1 && std::string(argv[1]) == "--help" ? 0 : 2;
    }

    try {
        const Model model = create_model(options);
        const std::vector<Backend> backends = select_backends(options);
        if (backends.empty()) {
            throw std::runtime_error("no usable decoder backend selected");
        }

        std::vector<Timing> aggregate(backends.size());
        std::vector<bool> verified(backends.size(), true);
        for (size_t backend_index = 0; backend_index < backends.size(); ++backend_index) {
            std::cout << "[backend] " << backends[backend_index].name << "\n";
            for (int repeat = 0; repeat < options.repeat; ++repeat) {
                RunResult result =
                    run_backend(model, backends[backend_index], options.verify && repeat == 0);
                aggregate[backend_index] += result.timing;
                verified[backend_index] = verified[backend_index] && result.verified;
                std::cout << "  repeat " << (repeat + 1) << "/" << options.repeat
                          << ": " << std::fixed << std::setprecision(4)
                          << per_token(result.timing.total_ms, result.timing.tokens)
                          << " ms/token\n";
            }
        }

        write_report(options, backends, aggregate, verified);
        std::cout << "[report] " << options.report << "\n";
        std::cout << "[csv] " << options.csv << "\n";
        for (size_t i = 0; i < backends.size(); ++i) {
            const double total = aggregate[i].total_ms / options.repeat;
            const double gemv = aggregate[i].gemv_ms / options.repeat;
            std::cout << "[summary] " << backends[i].name << ": "
                      << per_token(total, options.tokens) << " ms/token, GEMV "
                      << (total > 0.0 ? 100.0 * gemv / total : 0.0) << "%\n";
        }
    } catch (const std::exception &error) {
        std::cerr << "[error] " << error.what() << "\n";
        return 1;
    }
    return 0;
}
