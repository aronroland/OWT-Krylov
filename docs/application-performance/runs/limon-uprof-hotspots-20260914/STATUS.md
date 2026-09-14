# AMD uProf attempt: collection blocked

Documented workflow executed from OWT-Krylov on 2026-09-14:

```bash
METIS_PATH=/home/aron/opt/parmetis_gfortran \
  bash benchmarks/specwave_limon.sh --run-id limon-uprof-hotspots-20260914 \
  --ranks 4 --detail-timings --amd-uprof /opt/AMDuProf_5.3-521
```

AMD uProf version: 5.3.521.0. CPU: AMD Ryzen 7 PRO 7840U.
Kernel profiling restriction: `kernel.perf_event_paranoid=4`.
No sysctls, capabilities, power settings, or user processes were changed.

The current-source double-precision application was rebuilt with the recorded
flags, source hashes and dependency hashes in `bin/ww-x.build.json`. SpecWave
HEAD was `abe2d6ba`; the uncommitted integration changes are retained in the
source manifest and worktree diff. This executable includes the final layout
guards and MPI export-option consistency check missing from the earlier
vector-update performance prototype.

The unprofiled reference completed all 36 timesteps on four ranks:

- 1,778 vertices, 1,296 spectral components, 2,304,288 unknowns.
- 721 total iterations; maximum true relative residual 9.72698039e-9.
- Full finite, nonnegative owned-state coverage passed.
- Process wall time 41.555620122 s; integration wall time 29.790179049 s.
- No second completed sample exists, so no independent bitwise comparison or
  speedup is established by this attempt. `state_bitwise_equal` for the first
  sample only denotes that it established the reference.

The subsequent four AMD collectors all refused collection before launching the
application; the MPI launcher reported exit code 50:

```text
ERROR: For non-root users, perf_event_paranoid value should be <= 3 for launch app or already running processes:
ERROR: Collection of profile data failed
Exit code: 50
```

See `profile-00/run.command.json`, `profile-00/run.log`, `profile-00/run.exit`,
and `profile-00/uprof-logs/`. The `.uprof` files under `profile-00/uprof/` are
empty-session metadata with zero samples, not successful collection evidence.
No report was generated. No solver bottleneck can be attributed from this
attempt. All launched processes exited.

The attempted runner requested `--call-graph dwarf:4096`; AMD warned that this
is unsupported for Hotspots. The current runner corrects this to `-g` for
Hotspots and no call stacks for the other configurations. The original launch
arguments and runner hashes remain unchanged here. Full successful profiling
and report generation still require validation after host access is authorized.

This was a fresh solver reference run plus a failed profile collection. It was
not archived-reference validation, figure regeneration, or production-scale
performance qualification.
