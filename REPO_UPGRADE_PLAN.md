# Repository Review: Current Design and Upgrade Plan

This document summarizes the current architecture in `mock_test_stream.py` and proposes practical upgrades for production-readiness.

## Current architecture

- Single Python script handles:
  - RTSP frame capture thread (`capture_frames`)
  - YOLO inference + centroid tracking + counting thread (`process_stream`)
  - FastAPI HTTP endpoint (`/count`)
- Shared global state is protected by one lock (`frame_lock`):
  - `current_frame`
  - `current_count`
- Counting logic:
  - Detect `bag` and `boiler_opening`
  - Keep the highest-confidence opening
  - Track bag IDs via centroid tracker
  - Count a bag once if IoU or center-in-opening condition is true

## Upgrade priorities

1. **Configuration hardening**
   - Replace hard-coded model path and RTSP URL with environment-driven settings.
   - Validate startup config (model exists, RTSP URL format).
   - Add profile-based toggles for dev/prod (`SHOW_WINDOW`, thresholds).

2. **Service decomposition**
   - Split into modules: API layer, stream ingestion, inference service, tracking/counting domain logic.
   - Keep pure functions for detection post-processing and counting rules.

3. **Concurrency model cleanup**
   - Move from ad-hoc shared globals to structured producer-consumer queues.
   - Ensure only one owner updates count state.
   - Add bounded queues and backpressure policy (drop oldest / newest explicitly).

4. **FastAPI modernization**
   - Convert endpoint(s) to async handlers.
   - Add `/health`, `/ready`, `/metrics` endpoints.
   - Add structured response schema versioning.

5. **State and persistence**
   - Persist count snapshots and reset events to a datastore.
   - Track session metadata (start time, stream id, model hash, thresholds).

6. **Observability**
   - Add structured logging with correlation IDs.
   - Export metrics (FPS, queue depth, inference latency, reconnect count, dropped frames).
   - Emit tracing spans for capture → inference → count pipeline.

7. **Model governance**
   - Add model metadata validation (expected labels include `bag` + `boiler_opening`).
   - Track model version and confidence drift.
   - Support warmup and health checks for model load.

8. **Robust counting semantics**
   - Introduce explicit finite-state transitions for each track: `detected -> near_opening -> counted -> expired`.
   - Add temporal smoothing and minimum dwell frames before counting.
   - Add track lifecycle expiry and anti-double-count rules across occlusion gaps.

9. **Error handling and resilience**
   - Replace print statements with leveled logs.
   - Wrap major loops with fail-safe recoveries and metrics.
   - Handle invalid frames and corrupted decoder outputs defensively.

10. **Testing strategy**
   - Unit tests for IoU, center checks, detection extraction, tracker update.
   - Deterministic simulation tests for counting behavior.
   - API tests for `/count`, health endpoints, and startup failures.

11. **Security and API hardening**
   - Add auth for operational endpoints.
   - Add rate limiting and CORS policy.
   - Restrict sensitive runtime details in responses.

12. **Deployment readiness**
   - Containerize with multi-stage build.
   - Add graceful shutdown hooks.
   - Add process supervision and restart strategy.

## Recommended phased rollout

- **Phase 1 (stability):** config, logging, health endpoints, bounded queue, tests for counting core.
- **Phase 2 (correctness):** state-machine counting, persistence, metrics dashboard.
- **Phase 3 (scale):** modular services, async APIs, model governance, security hardening.
