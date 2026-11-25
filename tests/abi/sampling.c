/* Model fixture for checking the arguments DStan passes through the C ABI. */
#define TINYSTAN_EXPORT
#include "tinystan.h"

#ifdef _WIN32
__declspec(dllimport) int dstan_runtime_marker(void);
#endif

struct TinyStanModel { int marker; };

TINYSTAN_PUBLIC void tinystan_api_version(int *major, int *minor, int *patch) {
    *major = 0; *minor = 3; *patch = 3;
}

TINYSTAN_PUBLIC void tinystan_stan_version(int *major, int *minor, int *patch) {
    *major = 2; *minor = 39; *patch = 0;
}

TINYSTAN_PUBLIC TinyStanModel *tinystan_create_model(
    const char *data, unsigned int seed, TINYSTAN_PRINT_CALLBACK callback,
    TinyStanError **err) {
    static TinyStanModel model;
#ifdef _WIN32
    model.marker = dstan_runtime_marker();
#endif
    return &model;
}

TINYSTAN_PUBLIC void tinystan_destroy_model(TinyStanModel *model) {}
TINYSTAN_PUBLIC const char *tinystan_model_param_names(const TinyStanModel *model) {
    return "";
}
TINYSTAN_PUBLIC const char *tinystan_get_error_message(const TinyStanError *err) {
    return "sampling fixture error";
}
TINYSTAN_PUBLIC TinyStanErrorType tinystan_get_error_type(const TinyStanError *err) {
    return generic;
}
TINYSTAN_PUBLIC void tinystan_destroy_error(TinyStanError *err) {}

TINYSTAN_PUBLIC int tinystan_sample(
    const TinyStanModel *model, size_t num_chains, const char *inits,
    unsigned int seed, unsigned int chain_id, double init_radius,
    int num_warmup, int num_samples, TinyStanMetric metric_choice,
    const double *init_inv_metric, bool adapt, double delta, double gamma,
    double kappa, double t0, unsigned int init_buffer, unsigned int term_buffer,
    unsigned int window, bool save_warmup, double stepsize,
    double stepsize_jitter, int max_depth, int refresh, int num_threads,
    double *out, size_t out_size, double *stepsize_out, double *inv_metric_out,
    TinyStanError **err) {
    if (out_size < 1) return 1;
    out[0] = (double)num_chains;
    return 0;
}
