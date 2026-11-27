/* Model fixture for checking the arguments DStan passes through the C ABI. */
#define TINYSTAN_EXPORT
#include "tinystan.h"
#include <string.h>

#ifdef _WIN32
__declspec(dllimport) int dstan_runtime_marker(void);
#endif

/* Deliberately different from 0.3.3's delimiter to check symbol lookup. */
TINYSTAN_PUBLIC char tinystan_separator_char(void) { return '\x1e'; }

struct TinyStanModel { int marker; size_t num_free_params; };

TINYSTAN_PUBLIC void tinystan_api_version(int *major, int *minor, int *patch) {
    *major = 0; *minor = 3; *patch = 3;
}

TINYSTAN_PUBLIC void tinystan_stan_version(int *major, int *minor, int *patch) {
    *major = 2; *minor = 39; *patch = 0;
}

TINYSTAN_PUBLIC TinyStanModel *tinystan_create_model(
    const char *data, unsigned int seed, TINYSTAN_PRINT_CALLBACK callback,
    TinyStanError **err) {
    static TinyStanModel normal = {0, 46340}, oversized = {0, 46341};
    TinyStanModel *model = strcmp(data, "{\"oversized\":true}") == 0 ? &oversized : &normal;
#ifdef _WIN32
    model->marker = dstan_runtime_marker();
#endif
    return model;
}

TINYSTAN_PUBLIC void tinystan_destroy_model(TinyStanModel *model) {}
TINYSTAN_PUBLIC const char *tinystan_model_param_names(const TinyStanModel *model) {
    return "adapt,delta,metric";
}
TINYSTAN_PUBLIC size_t tinystan_model_num_free_params(const TinyStanModel *model) {
    return model->num_free_params;
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
    if (out_size < 10) return 1;
    /* Even an empty slice with backing storage must select identity. */
    if (init_inv_metric != NULL) return 1;
    /* Simulate a successful first chain and a second that never wrote its state. */
    if (stepsize_out != NULL) stepsize_out[0] = stepsize;
    out[9] = metric_choice;
    out[7] = adapt;
    out[8] = delta;
    out[6] = max_depth;
    out[5] = stepsize_jitter;
    out[4] = stepsize;
    out[0] = (double)num_chains;
    out[1] = num_threads;
    out[2] = init_radius;
    if (!inits) out[3] = 0;
    else if (!strcmp(inits, "{\"theta\":-2}")) out[3] = 1;
    else if (!strcmp(inits, "{\"theta\":-2}\x1e{\"theta\":2}")) out[3] = 2;
    else return 1;
    return 0;
}
