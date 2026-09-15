# Concurrent host activity

During `native-pipelined-00`, an interactive process observation found PID
1791888 (`cc1plus`) consuming approximately one CPU core concurrently with the
four `ww-x` ranks. This was not the comparison binary's build, which had already
finished before the first Gauss-Seidel sample.

The follow-up command was:

```bash
ps -C cc1plus,g++,gmake,cmake,ww-x -o pid,ppid,lstart,etime,pcpu,args
```

It identified the separate build:

```text
PID 1791868, started Mon Sep 14 14:59:53 2026 local time:
cmake --build build-owt --target unit_Test_sources_ST4 -j8

PID 1791887, same start time:
/usr/bin/g++ ... -c /home/aron/git/Triton_C/test/Test_sources_ST4/tst_test_sources_st4.cpp
```

At the follow-up observation, the OWT sample had started at 15:00:29 local
time and the compiler's `cc1plus` child had exited. These are point observations,
not continuous host monitoring or proof that all other samples were idle.
No external processes were stopped. Timing results must be labeled observations
on a shared, uncontrolled machine; this batch cannot establish a controlled
performance claim. Numerical audits and same-solver repeatability are checked
separately from wall time.

During `owt-krylov-02`, with the application log through step 12, the same
process-list command also found a second separate build:

```text
PID 1796500, elapsed 1 minute 4 seconds:
cmake --build build-owt --target unit_Test_ST4Post -j8

PID 1796520 (cc1plus), elapsed 1 minute 3 seconds, approximately 100% CPU:
compiling /home/aron/git/Triton_C/test/Test_ST4Post/tst_st4_post.cpp
```

The OWT step-12 maximum-rank solve time was 4.703474749 s. The observation
establishes overlap with another build, not how much of the slowdown that build
caused. No samples are discarded.
