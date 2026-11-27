#include <atomic>
#include <chrono>
#include <ostream>
#include <stdexcept>
#include <thread>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <csignal>
#include <unistd.h>
#endif

// Deliver one interrupt once every requested worker reaches generated quantities.
inline int interrupt_workers(int workers, std::ostream*) {
    static std::atomic<int> entered{0};
    static thread_local bool seen = false;
    if (!seen) {
        seen = true;
        if (++entered == workers) {
#ifdef _WIN32
            DWORD pid;
            if (GetConsoleProcessList(&pid, 1) != 1 || pid != GetCurrentProcessId()
                || !GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, 0))
                throw std::runtime_error("Expected an isolated console for Ctrl+Break");
#else
            if (kill(getpid(), SIGINT) != 0)
                throw std::runtime_error("Could not deliver SIGINT");
#endif
        } else {
            // Keep the first chain active until the second worker enters.
            // The parent process bounds this rendezvous with a timeout.
            while (entered.load() < workers)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    return 1;
}
