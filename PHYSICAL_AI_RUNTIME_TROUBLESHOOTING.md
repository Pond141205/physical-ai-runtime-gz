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
