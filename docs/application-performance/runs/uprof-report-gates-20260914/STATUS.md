# Report-parser validation failure

The first validation attempt completed 40 synthetic checks, then failed while
decoding the actual Hotspots CSV as UTF-8. AMD's source/symbol records include
non-UTF-8 bytes. This was a report-reader failure, not failed solver execution
or failed AMD collection. The partial `test-results.json` is not an overall
success result.

The reader was changed to lossless `surrogateescape` decoding, with a retained
non-UTF-8 fixture. Subsequent validation runs are separate directories;
`uprof-report-gates-v2-20260914` passed the Hotspots report and
`uprof-report-gates-v3-20260914` passed both Hotspots and Assess reports.
