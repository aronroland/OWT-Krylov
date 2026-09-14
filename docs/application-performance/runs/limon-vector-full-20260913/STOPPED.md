# Interrupted timing experiment

The planned five-pair run was stopped on 2026-09-13 after two complete pairs,
during `baseline-02`. Completed samples and the interrupted sample are retained.
The active MPI launcher (PID 629443) received SIGTERM; the driver recorded a
nonzero process exit and stopped. No model ranks remained afterward.

The unchanged baseline's integration time rose from 31.736 to 85.643 seconds.
Several baseline per-solve times rose from about 0.8 to 4 seconds across spatial,
spectral and preconditioner phases. This makes the batch unsuitable for a
speedup conclusion. The apparent large advantage in pair 1 must not be cited
as an optimization gain. No observations were deleted or filtered.

Before/after host snapshots are retained in each sample directory; they are
not continuous measurements. Additional point observations were CPU0 frequency
2975045 kHz, thermal-zone-0 temperature 88000 millidegrees C, and load averages
6.13, 5.18, 3.81. These do not establish the cause of the slowdown.

All four completed samples passed the full-state and operation-count gates:
36 steps, 721 total iterations, and maximum accepted relative residual
9.72698039e-9. The interrupted sample is not a completed solver reproduction.
The timed candidate predates the restoration of the two preconditioned-vector
layout checks before its chunked update; no final-source timing claim is made.
