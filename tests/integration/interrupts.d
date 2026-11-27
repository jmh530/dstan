module interrupts;

import dstan;
import std.conv : to;
import std.exception : enforce;
import std.math : isFinite;
import std.stdio : writeln;

version (Windows)
{
    import core.sys.windows.windows;
    private __gshared HANDLE restored;
    private extern(Windows) BOOL priorHandler(DWORD event) nothrow @nogc
    {
        if (event != CTRL_BREAK_EVENT) return FALSE;
        SetEvent(restored);
        return TRUE;
    }
}
else version (Posix)
{
    import core.stdc.signal;
    import core.volatile : volatileLoad, volatileStore;
    import std.traits : Unsigned;
    private __gshared Unsigned!sig_atomic_t restored;
    private extern(C) void priorHandler(int) nothrow @nogc
    {
        volatileStore(&restored, 1);
    }
}

void checkInterrupt(string library, int workers, string[] dirs)
{
    enforce(workers == 1 || workers == 2);
    auto model = Model(library, `{"workers":` ~ workers.to!string ~ "}", 1, dirs);
    version (Windows)
    {
        restored = CreateEventW(null, TRUE, FALSE, null);
        enforce(restored);
        scope(exit) CloseHandle(restored);
        enforce(SetConsoleCtrlHandler(&priorHandler, TRUE));
    }
    else
    {
        auto before = signal(SIGINT, &priorHandler);
        enforce(before != SIG_ERR);
    }
    // Keep the application handler installed for the whole experiment.
    scope(exit)
    {
        version (Windows) SetConsoleCtrlHandler(&priorHandler, FALSE);
        else signal(SIGINT, before);
    }
    SampleOptions options;
    options.chains = options.threads = workers;
    options.warmup = 50;
    options.draws = 50_000;
    try
    {
        model.sample(options);
        assert(0, "Sampling completed without an interrupt");
    }
    catch (StanException e)
    {
        writeln("workers=", workers, " category=", e.category, " message=", e.msg);
        enforce(e.category == 2, "Expected TinyStan's interrupt category");
    }
    version (Windows)
    {
        enforce(WaitForSingleObject(restored, 0) == WAIT_TIMEOUT);
        enforce(GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, 0));
        enforce(WaitForSingleObject(restored, 5_000) == WAIT_OBJECT_0, "Prior console handler missing");
    }
    else
    {
        enforce(!volatileLoad(&restored));
        enforce(raise(SIGINT) == 0 && volatileLoad(&restored), "Prior SIGINT handler missing");
    }
    options.draws = 10;
    auto fit = model.sample(options);
    enforce(fit.shape == [workers, 10, 9]);
    foreach (chain; 0 .. workers)
        foreach (row; 0 .. options.draws)
            enforce(isFinite(fit[chain, row, 7]) && fit[chain, row, 8] == 1);
    writeln("Prior handler restored and same model sampled again");
}
