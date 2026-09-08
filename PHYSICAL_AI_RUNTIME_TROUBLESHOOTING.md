# physical-ai-runtime-gz — Troubleshooting & Recovery Notes

## Mandatory Troubleshooting Practice
Before diagnosing or changing code, read this file and match the observed symptom against existing entries. Use documented verification before trying a new experiment. After resolution, append a dated entry with: symptom and affected robot/component; evidence-backed root cause; smallest corrective change; exact verification commands and measured result; and remaining limitation or follow-up.

Do not label an unverified workaround as a fix. Integration, perception, planning, camera, grasp, and execution tests require the official full stack; when a required dependency is absent, fail closed rather than substituting fake ROS state, TF, or sensor data.

## Known-good result
- UR5e: PASS
- Panda: PASS

## Startup

Terminal 1:
source /opt/ros/jazzy/setup.bash
cd ~/physical-ai-runtime-gz
python3 launch/dual_robot.launch.py

Terminal 2:
source /opt/ros/jazzy/setup.bash
cd ~/physical-ai-runtime-gz
python3 launch/dual_moveit.launch.py

Terminal 3:
source /opt/ros/jazzy/setup.bash
cd ~/physical-ai-runtime-gz
python3 -m tests.motion.test_dual_runtime

## Important known issues

### ros2 command not found
Fix:
source /opt/ros/jazzy/setup.bash

### Duplicate move_group / stale process
Fix:
pkill -9 -f move_group
pkill -f robot_state_publisher
pkill -f ros_gz_bridge
pkill -f controller_manager
pkill -f gz
pkill -f spawner

ros2 daemon stop
ros2 daemon start

### Panda frame offset ~0.6 m
Use:
request.ik_request.pose_stamped.header.frame_id = self.base_frame

with:
self.base_frame = "panda_link0"

### UR5e IK error -31
Current workaround:
request.ik_request.avoid_collisions = False

This is temporary. Final fix should restore collision checking.

### UR5e false TARGET_NOT_REACHED
Cause:
TF/state was read too quickly after trajectory completion.

Fix:
poll TCP until error <= tolerance or timeout.

### Controller namespaces

UR5e:
/ur5e/controller_manager
/ur5e/joint_states
/ur5e/scaled_joint_trajectory_controller/follow_joint_trajectory
/ur5e/compute_ik

Panda:
/panda/controller_manager
/panda/joint_states
/panda/panda_arm_controller/follow_joint_trajectory
/panda/compute_ik

### Readiness check
ros2 control list_controllers -c /ur5e/controller_manager
ros2 control list_controllers -c /panda/controller_manager
ros2 service list | grep compute_ik
ros2 action list | grep follow_joint_trajectory
ros2 topic echo /clock --once

### Final known-good test
UR5e : PASS
Panda: PASS

## Debug rule
1. Read this file first.
2. Match the symptom to an existing issue.
3. Run the documented verification.
4. Only try a new experiment if the problem is actually new.
5. Append new root cause + fix after solving it.


## 2026-09-05 - Multi-step semantic runtime PASS

Test: `python3 -m scripts.runtime_demo`

Result:
- UR5e: PASS
- Panda: PASS
- Runtime: PASS

Semantic sequence:
- Step 1: [0.02, 0.02, 0.02]
- Step 2: [0.04, -0.02, 0.04]
- Step 3: [-0.02, 0.03, 0.01]

Max TCP error:
- UR5e: 0.000144 m
- Panda: 0.001387 m

Controller backends:
- UR5e -> ScaledJointTrajectoryController
- Panda -> JointGroupPositionController

Demonstrated:
- Same semantic MoveTCP contract across different robot embodiments
- Same workspace-frame semantics
- Different controller backends hidden behind adapters
- Standardized SkillResult
- Multi-step task execution

Next: move sequencing into RobotRuntime using TaskSequence.


## 2026-09-05 - TaskSequence runtime integration PASS

Result:
- UR5e: PASS
- Panda: PASS
- Runtime: PASS

Implemented:
- TaskSequence
- TaskResult
- RobotRuntime.execute_task()
- stop_on_failure behavior
- multi-step semantic sequencing inside runtime layer

Architecture milestone:
TaskSequence -> RobotRuntime.execute_task() -> execute() -> robot adapter -> controller backend

Confirmed same 3-step MoveTCP task executes successfully on UR5e and Panda.


## 2026-09-05 - Mixed semantic task PASS

Task:
MoveTCP -> MoveJoints -> MoveTCP

Result:
- UR5e: PASS
- Panda: PASS
- Runtime: PASS

Confirmed:
- Multiple semantic skill types can be dispatched inside one TaskSequence
- MoveTCP and MoveJoints share the same RobotRuntime abstraction
- Different robot embodiments and controller backends remain hidden behind adapters
- Standardized TaskResult and SkillResult semantics remain valid

Observed errors:
- UR5e MoveJoints: 1.63e-05
- Panda MoveJoints: 6.04e-04
- UR5e final TCP error: 4.22e-05 m
- Panda final TCP error: 1.12e-03 m

Next: add Grasp/Release and test unsupported capability semantics.

## 2026-09-05 — Panda semantic gripper + cross-robot capability negotiation

- Panda native Gazebo JointTrajectoryController controls both fingers synchronously.
- ROS command bridge: `/panda_gripper_native_trajectory` (`trajectory_msgs/msg/JointTrajectory` <-> `gz.msgs.JointTrajectory`).
- Gazebo JointStatePublisher exposes finger feedback on `/panda_gripper_joint_states` and bridges to `sensor_msgs/msg/JointState`.
- `GazeboPandaAdapter.move_gripper()` now uses trajectory command + closed-loop finger feedback.
- Semantic `Release(width=0.08)` and `Grasp(width=0.00)` pass on Panda.
- UR5e rejects Grasp/Release with `UNSUPPORTED_CAPABILITY`.
- `test_semantic_gripper.py`: Panda PASS, UR5e PASS, Runtime PASS.
- `test_cross_robot_manipulation.py`: Panda completes MoveTCP -> Release -> Grasp -> MoveTCP; UR5e completes MoveTCP then correctly fails at step 2 with `UNSUPPORTED_CAPABILITY`; Runtime PASS.
- UR5e MoveTCP duration 2.0 s caused `Aborted due to path tolerance violation`; 3.0 s passed and is used in cross-robot regression.

## 2026-09-05 — Panda gripper cleanup regression

- Removed obsolete Panda ros2_control gripper/debug controllers from panda_controllers_dual.yaml.
- Removed panda_gripper_controller spawner and startup event from dual_robot.launch.py.
- Native Gazebo gripper backend remains independent of ros2_control gripper controllers.
- After clean restart, test_semantic_gripper.py: Panda PASS, UR5e PASS, Runtime PASS.
- After cleanup, test_cross_robot_manipulation.py: Panda PASS, UR5e PASS, Runtime PASS.
- UR5e intermittently produced `Aborted due to path tolerance violation` during MoveTCP. Controller remained active, interfaces correctly claimed, configured path tolerance was 0.2 rad, idle tracking error was near zero, direct MoveJoints passed, and repeated MoveTCP test passed 10/10. No retry/workaround added because root cause is not yet established.

## 2026-09-07 - Panda wrist camera mount did not update in live TF

Status: source fix applied; canonical full-stack verification remains required before this entry can be marked PASS.

Symptom:
- The wrist RGB image looked outward, away from the Panda gripper/grasp zone.
- Editing the xacro appeared not to change the live `panda_hand -> panda_wrist_camera_optical_frame` transform.

Established cause:
- The live `/panda/robot_state_publisher` had loaded an older `robot_description`. `robot_state_publisher` parameters do not hot-reload after a xacro edit.
- The previous optical +Z direction in `panda_hand` was `[1, 0, 0]`, outward.

Source of truth and corrective change:
- The canonical full-stack launch loads `panda_gz/panda_gazebo.urdf.xacro` through `launch/dual_robot.launch.py`.
- The wrist-camera mount was moved to `[0.100, 0, 0.020]` m and pitched so optical +Z targets the finger-mesh-derived grasp-zone midpoint `[0, 0, 0.085390365]` in `panda_hand`.
- `viewpoint_pose.py` now obtains the camera extrinsic from runtime TF and fails closed when it is unavailable; no Panda mount constant remains in generic viewpoint conversion.

Static verification evidence:
```bash
source .venv/bin/activate
source /opt/ros/jazzy/setup.bash
python -m tests.integration.test_panda_wrist_camera_tf
python -m tests.unit.test_viewpoint_pose
```

Measured result:
- optical forward in `panda_hand`: `[-0.8369475733, 0.0, 0.5472830708]`
- alignment with camera-to-grasp-zone direction: `1.0`
- viewpoint position round-trip error: `2.22e-16` m
- viewpoint rotation round-trip error: `2.48e-16`

Required follow-up before declaring the camera issue resolved:
- Terminate stale ROS/Gazebo processes and start only `python3 launch/full_dual_demo.launch.py` after sourcing ROS.
- Verify clock, Panda robot_state_publisher, merged Panda joint states, Panda TF, wrist RGB/depth/CameraInfo, and no duplicate TF authority.
- Rerun live TF consistency, wrist image, RobotSelfMask, and snapshot-geometry regressions using the healthy canonical stack. Do not manually spawn a model, publish TF, joint state, or image data to make these tests pass.

## 2026-09-07 - Package import failed without Gemini credentials

Symptom:
- Import-smoke validation after the production package refactor failed when
  `GEMINI_API_KEY` was not configured.

Root cause:
- `scripts.chat_gemini` constructed `genai.Client` and started its interactive
  loop at module import time.

Fix:
- The script now exposes `main()` and constructs the client only inside that
  entrypoint. It raises a clear `RuntimeError` only when the Gemini CLI is
  explicitly started without `GEMINI_API_KEY`.

Verification:
```bash
source .venv/bin/activate
source /opt/ros/jazzy/setup.bash
python -m compileall -q -f physical_ai_runtime scripts tests launch
python -c "import scripts.chat_gemini"
```

Result: import succeeds without a Gemini secret; the CLI remains fail-closed
when invoked without one. No ROS or robot motion was executed.

## 2026-09-07 - Cube query selected a cylinder distractor in side camera

Symptom:
- With the launch-owned `cube_with_distractors` scene, a `cube` query located
  the cube in the main camera but selected the green cylinder in the side
  camera. The two metric positions were about `0.218 m` apart.

Root cause:
- `SnapshotGeometryProcessor` ranked each camera's semantic detections
  independently by detector confidence. It had no cross-view association when
  several scene objects matched an open-vocabulary query.

Fix:
- The main camera's metric object position is now passed to side-camera
  candidate ranking as a reference. The chosen side candidate records
  `reference_distance_m`; no robot model, task-layer offset, fake SceneState,
  or motion workaround is involved.
- `test_cube_distractor_consistency.py` uses the canonical full stack and the
  launch-owned scene. It retries only camera/TF warm-up and asserts that both
  camera positions match the actual cube, agree with one another, and remain
  separated from the cylinder distractor.

Verification:
```bash
export PHYSICAL_AI_SCENE_VARIANT=cube_with_distractors
python3 launch/full_dual_demo.launch.py
source .venv/bin/activate
source /opt/ros/jazzy/setup.bash
python -m tests.integration.test_cube_distractor_consistency
```

Result:
- main cube position error: `0.002305 m`
- side cube position error: `0.014928 m`
- main/side agreement: `0.014191 m`
- side/cylinder separation: `0.201217 m`
- `CUBE_DISTRACTOR_CONSISTENCY=PASS`; no robot motion executed.

## 2026-09-08 - Conversational discovery and semantic execution contract drift

Symptoms:
- The new Gemini-backed object discovery path always fell back before using Gemini.
- The conversational prompt advertised `PICK_AND_PLACE` and `GRASP`, but
  `RobotRuntime.execute_semantic_program()` rejected them as unimplemented.
- Panda execution used a type-name fast path that skipped re-observation after
  approach and before gripper close.
- `HOLD` was declared as a semantic action but validation incorrectly required
  an object id.

Root causes:
- `MultiviewVisionReasoner.discover_objects()` called the target-analysis API
  with undefined `scene_context` and `task` names, and expected the wrong
  response schema.
- The task prompt and runtime executable action set had diverged.
- A Panda-specific conditional bypassed the normal closed-loop perception path.
- `SemanticManipulationTask.validate()` grouped `HOLD` with object-bearing
  actions.

Smallest corrective changes:
- Added a dedicated Gemini `discover_objects()` response schema and normalized
  candidate labels/cameras/confidence before deterministic grounding.
- Limited conversational prompts to executable `MOVE_POSE`, `PICK`, `HOLD`,
  and `RELEASE` actions; unsupported future contract values fail closed.
- Reused a fresh verified scene without skipping post-approach or pre-close
  re-observation, and removed the Panda type-name branch.
- Made `HOLD` a no-object semantic action and routed `HOLD`/`RELEASE` through
  common runtime controls.
- Kept `HOLD`/`RELEASE` side-effect free in `execute=False` plan-only mode.

Verification:
```bash
source .venv/bin/activate
source /opt/ros/jazzy/setup.bash
python -m compileall -q -f physical_ai_runtime scripts tests launch
python -m unittest -v tests.unit.test_multiview_vision_reasoner tests.unit.test_groq_task_program tests.unit.test_runtime_execution_authorization tests.unit.test_runtime_semantic_program tests.unit.test_semantic_manipulation_task tests.unit.test_semantic_terminal_policy tests.unit.test_viewpoint_planner tests.unit.test_viewpoint_pose tests.unit.test_viewpoint_selector_cases
```

Result:
- compile check: PASS
- unit regression: `21/21 PASS`
- no ROS/Gazebo stack or robot motion was run for this correction.

Remaining limitation:
- `PICK_AND_PLACE` and `GRASP` remain validated contract values for a future
  placement/grasp-only runtime path, but are intentionally not emitted by the
  conversational prompt until that path has its own verified implementation.

## 2026-09-08 - Grasp confirmation reloaded the detector on every check

Symptom:
- A conversational Panda pick became slow because the detector appeared to
  load again after approach, before gripper close, and after lift.

Root cause:
- The initial target observation used the persistent `CameraManager`, but
  `RobotRuntime.reobserve_grasp_object()` and
  `RobotRuntime.verify_lifted_object()` constructed a new
  `GraspVerifySceneObserver` with its default `initialize_detector=True`.
- That rebuilt GroundingDINO instead of reusing the detector owned by the
  current perception session.

Fix:
- `CameraManager` now creates one shared detector per session and attaches it
  to its persistent main, side, and wrist observers.
- Grasp rechecks reuse the persistent side observer and its TF/subscription
  state. They still capture current sensor state and run fresh detector
  inference; only model/node construction is reused.
- Runtime passes the same perception manager through approach, pre-close, and
  post-lift verification. Safety checks and fail-closed behavior remain in
  place.

Verification:
```bash
source .venv/bin/activate
source /opt/ros/jazzy/setup.bash
python -m compileall -q -f physical_ai_runtime scripts tests launch
python -m unittest -v tests.unit.test_camera_manager_detector_reuse tests.unit.test_runtime_semantic_program
```

Result:
- Shared-detector construction is asserted once per `CameraManager` session.
- Persistent side-observer recheck and runtime manager propagation tests pass.
- Full compile and unit regression: `23/23 PASS`.
- Legacy `observe`, `plan grasp`, `advise grasp`, `grasp`, and `task_program`
  terminal paths now receive the same persistent perception manager; failed
  perception in these paths fails closed instead of creating a new detector.
- No ROS/Gazebo stack or robot motion was run for this correction.

## 2026-09-08 - Continuous object perception blocked command latency

Symptom:
- Every user command waited for synchronous open-vocabulary detector work,
  even though camera callbacks were already running continuously.
- Repeated confirmations were safer after detector reuse, but still blocked on
  the next full inference call.

Root cause:
- The repository had persistent RGB-D/TF observers but no background inference
  owner and no timestamped perception cache.
- Detector results were not tracked between observations, so the terminal had
  no current labeled object state to read immediately.

Fix:
- Added a deterministic `ObjectTracker` that preserves detector-provided
  labels, assigns stable per-camera track IDs, predicts only through short
  detection gaps, and rejects stale timestamps.
- Added `CameraManager.start_continuous_perception()` with a background
  detector/tracker worker and a bounded-age `ContinuousPerceptionFrame` cache.
- `observe_manipulation_target()` consumes the cache only while it is fresh;
  stale or unavailable data falls back to fresh detection. Motion verification
  remains on the runtime safety path and is not authorized from stale tracks.
- Added terminal options `--continuous-query <label>` and
  `--continuous-rate-hz <hz>`, plus track visibility in `status`.

Verification:
```bash
source .venv/bin/activate
source /opt/ros/jazzy/setup.bash
python -m compileall -q -f physical_ai_runtime scripts tests launch
python -m unittest -q tests.unit.test_object_tracker tests.unit.test_camera_manager_detector_reuse tests.unit.test_multiview_vision_reasoner tests.unit.test_groq_task_program tests.unit.test_runtime_execution_authorization tests.unit.test_runtime_semantic_program tests.unit.test_semantic_manipulation_task tests.unit.test_semantic_terminal_policy tests.unit.test_viewpoint_planner tests.unit.test_viewpoint_pose tests.unit.test_viewpoint_selector_cases
```

Result:
- Compile: PASS.
- Unit regression: `27/27 PASS`.
- No ROS/Gazebo stack or robot motion was run for this change.

## 2026-09-09 - Live conversational pick exposed lift failure and camera time drift

Observed evidence from a canonical terminal run:
- The first semantic `PICK` request completed perception, approach planning,
  approach execution, pre-close verification, and authorization, then failed
  during the lift path with `PICK_EXECUTION_FAILED:MOTION_FAILED`.
- A later multiview request failed closed with
  `CROSS_CAMERA_TIME_MISMATCH` instead of producing a synchronized scene.
- A second pick request was rejected at pre-planning for the same cross-camera
  timestamp mismatch.
- The conversational response to `what happen` did not report the already
  observed `MOTION_FAILED` result and instead described the pick as if it were
  still in progress. This is a status-reporting correctness issue, not evidence
  that the motion succeeded.

Confirmed scope:
- Runtime safety did not authorize the failed lift as a success; the terminal
  surfaced a structured motion failure.
- Cross-camera synchronization is currently a hard prerequisite for the
  multiview perception path, and the observed snapshot did not satisfy it.

Not yet proven:
- The exact controller/trajectory cause of the lift `MOTION_FAILED`.
- Whether the timestamp mismatch is caused by simulator clock skew, delayed
  camera callbacks, or snapshot collection policy.
- Whether the visual scene changed after the failed lift; the subsequent
  capture was invalid and must not be interpreted as an empty scene.

Required follow-up:
- Inspect the lift trajectory result and adapter execution trace before changing
  motion planning or controller settings.
- Capture per-camera message timestamps, `/clock`, and snapshot age/skew during
  a clean full-stack run; fail closed on mismatch without claiming that no
  objects exist.
- Make conversational status queries read the latest structured runtime result
  and preserve failure state until a new command changes it.

Diagnosis update:
- The Panda MoveIt log for the failed lift records four goal-sampling failures,
  followed by `Unable to solve the planning problem` and
  `Planner 'OMPL' failed with error code FAILURE`. The failure occurred before
  trajectory authorization or controller execution.
- The current `PICK` path constructs its lift goal from
  `replan["grasp_target"]` and adds `lift_height` directly to base-frame
  Z. It only proves IK feasibility for the grasp and approach poses; it never
  proves that this derived lift pose is collision-free/reachable before it is
  sent to MoveIt.
- This direct lift branch bypasses the existing
  `GazeboPandaAdapter.plan_and_execute_lift()` implementation, which derives
  lift from the measured current TCP and transforms world-up through TF.

Conclusion:
- The observed 100% failure in the fixed demonstration scene is caused by a
  deterministic invalid lift goal in the current semantic `PICK` path. MoveIt
  rejects the goal before motion begins, so increasing controller timeout or
  changing detector latency cannot fix it.
- The log cannot yet distinguish whether the invalid goal is caused by
  reachability, self/world collision, or a frame offset. That final geometric
  discriminator must be recorded from the requested and current TCP poses plus
  MoveIt state validation before changing the implementation.

## 2026-09-09 - Legacy semantic aliases bypassed the primitive-only boundary

Symptom:
- The public task schema advertised `MOVE_TO`, `OPEN`, `CLOSE`, and `STOP`, but
  silently accepted `MOVE_POSE`, `RELEASE`, `GRASP`, and `HOLD` aliases.
- `RobotRuntime.execute_semantic_program()` retained an unreachable
  `PICK_INTERNAL` branch, and the Panda adapter retained a dead legacy lift
  implementation.
- Hard terminal `task` and emergency `stop` results were not consistently
  recorded in conversational runtime feedback.

Root cause:
- Compatibility aliases and dead branches remained after the command surface
  was narrowed. Feedback was implemented independently in multiple terminal
  branches.

Fix:
- Enforce exactly `MOVE_TO`, `OPEN`, `CLOSE`, and `STOP` at the public schema.
- Remove the dead `PICK_INTERNAL` branch and legacy Panda lift implementation.
- Keep emergency `stop` and `abort` hard-wired outside the language model while
  recording their actual result and measured TCP for subsequent AI turns.
- Return hard `task` results and record exceptions as failed runtime feedback.

Verification:
```bash
source .venv/bin/activate
source /opt/ros/jazzy/setup.bash
python -m compileall -q physical_ai_runtime scripts tests
python -m unittest discover -s tests/unit -p 'test_*.py' -v
```

Result:
- Compile: PASS.
- Unit regression: `36/36 PASS`.
- Public object actions and legacy aliases fail closed.

## 2026-09-09 - Repeated benchmark bypassed the public primitive path

Symptom:
- The Panda repeated-run test reset the arm with direct `robot.go_home()` and
  tested the older `MoveTCP` skill instead of the AI-facing primitive program.
- A command launched from a new WSL shell initially failed before motion with
  `.venv/bin/activate: No such file or directory` because its working directory
  was not inherited.

Fix:
- Capture the clean-stack initial TCP and use authorized `MOVE_TO` programs for
  both reset and target motion. No Panda coordinate compensation was added.
- Explicitly `cd /home/pond/physical-ai-runtime-gz` in WSL test commands.

Verification result on the canonical full stack:
- target TCP: `[0.450000, 0.150000, 0.450000]`
- initial/reset TCP: `[0.307020, -0.000000, 0.590270]`
- `passed=20/20`
- `max_error=0.008123 m`
- `mean_error=0.006222 m`
- `timeout_count=0`
- `failures=0`
- Every reset and target trajectory passed `ExecutionAuthorizationGate`.

## 2026-09-09 - MoveIt processes segfault during full-stack shutdown

Symptom:
- After the benchmark completed successfully, Ctrl-C shutdown caused both the
  Panda and UR5e `move_group` processes to exit with code `-11`.

Evidence:
- The crash happened after the launch process received SIGINT and after all
  benchmark motion and KPI reporting had completed.
- A process scan after launch teardown found no stale ROS or Gazebo process.

Impact:
- This is a teardown defect, not a motion-result failure.
- It does not invalidate the completed `20/20` benchmark, but it must not be
  silently treated as a clean MoveIt shutdown.

Follow-up:
- Reproduce with a launch-only start/stop cycle and capture MoveIt backtraces
  separately from motion testing.
- Keep process hygiene checks after every stack shutdown until the upstream or
  launch-order cause is isolated.

## 2026-09-09 - Object requests had no safe perception handshake

Symptom:
- The conversation protocol supported only `chat` and `task`.
- An object request such as "pick up the cube" could neither invent coordinates
  nor request deterministic metric geometry, so it had no valid path forward.

Root cause:
- The AI/runtime boundary omitted a read-only observation turn.
- Robot feedback existed after task execution, but current SceneState geometry
  could not be requested before proposing a primitive.

Fix:
- Added `mode=observe` with a required semantic query and no task program.
- Added a bounded terminal observation loop that calls the persistent
  `CameraManager.observe_manipulation_target()` path.
- Returned measured TCP, base frame, object position, size, height, support
  plane, camera, and verification reason to the next AI turn.
- Marked context as `geometry_verified=True` and
  `motion_authorized=False`; perception never grants motion authority.
- Repeated observation requests beyond the bounded handshake fail closed.

Implementation issue caught before integration:
- A generated patch wrote literal `\\n` text into two Python lines.
- `compileall` failed with `SyntaxError`; the literal text was replaced with
  real newlines, then compile and focused tests passed.

Verification:
```bash
source .venv/bin/activate
source /opt/ros/jazzy/setup.bash
python -m compileall -q physical_ai_runtime scripts tests
python -m unittest discover -s tests/unit -p 'test_*.py' -q
python -m unittest -v tests.integration.test_terminal_observation_loop
```

Measured live result on the canonical full stack:
- camera: `main`
- cube position in robot base:
  `[0.2479778909, -0.3511061626, 0.7549991317]`
- cube position in world:
  `[0.8479778909, -0.3511061626, 0.7549991317]`
- `geometry_verified=True`
- `motion_authorized=False`
- unit regression: `41/41 PASS`
- live integration: `1/1 PASS`
- robot motion: none

Remaining limitation:
- The WSL user configuration file containing `GROQ_API_KEY` was not present,
  so the real cloud conversation call was not exercised.
- This slice supplies verified geometry to AI but does not yet implement the
  multi-turn primitive grasp/place controller or claim grasp success.

## 2026-09-09 - Git automatic maintenance paused by loose objects

Symptom:
- Commit succeeded, but Git reported too many unreachable loose objects and
  left `.git/gc.log`, which disables subsequent automatic cleanup.

Cause:
- Repository object maintenance is overdue; source history and push succeeded.

Action:
- No destructive prune was run automatically.
