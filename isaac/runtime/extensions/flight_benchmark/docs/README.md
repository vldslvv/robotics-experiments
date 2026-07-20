# FlightBenchmark

Opt-in Kit plugin used only by `make benchmark-flight`. It depends on the live
`experiment_runtime` extension, observes its immutable status snapshot, and
wraps selected production callables with benchmark-only timers. Normal runtime
launches do not enable this extension.
