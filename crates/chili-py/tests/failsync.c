#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>

static int active, remaining, injected_errno, calls;
static int descriptors[16], results[16];
void probe_arm(int failures, int error) {
    remaining = failures; injected_errno = error; calls = 0; active = 1;
}
void probe_disarm(void) { active = 0; }
int probe_calls(void) { return calls; }
int probe_fd(int n) { return descriptors[n]; }
int probe_result(int n) { return results[n]; }

static int sync_impl(int fd, const char *symbol) {
    int (*real_sync)(int) = dlsym(RTLD_NEXT, symbol);
    int result;
    if (active && remaining > 0) {
        --remaining;
        errno = injected_errno;
        result = -1;
    } else {
        result = real_sync(fd);
    }
    int saved = errno;
    if (active && calls < 16) {
        descriptors[calls] = fd;
        results[calls++] = result == 0 ? 0 : saved;
    }
    errno = saved;
    return result;
}
int fdatasync(int fd) { return sync_impl(fd, "fdatasync"); }
int fsync(int fd) { return sync_impl(fd, "fsync"); }
