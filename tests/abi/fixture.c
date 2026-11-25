/* A partial export table is sufficient to exercise load-time rejection. */
#define TINYSTAN_EXPORT
#include "tinystan.h"

TINYSTAN_PUBLIC void tinystan_api_version(int *major, int *minor, int *patch) {
    *major = 0;
    *minor = 3;
    *patch = PATCH;
}

TINYSTAN_PUBLIC bool dstan_check_callback(TinyStanMetric metric, size_t count,
                                          unsigned int seed, TINYSTAN_PRINT_CALLBACK callback) {
    if (metric != diagonal || count != (size_t)-1 || seed != (unsigned int)-1)
        return false;
    callback("ok", 2, true);
    return true;
}
