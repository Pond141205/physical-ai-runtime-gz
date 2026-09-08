import time

import numpy as np
import rclpy
from scipy.spatial.transform import Rotation

from physical_ai_runtime.core.skills import MoveTCP, MoveJoints, Grasp, Release, GraspObject, TaskSequence, TaskResult
from physical_ai_runtime.core.frames import FrameTransform
from physical_ai_runtime.perception.rgbd_scene_observer import RGBDSceneObserver
from physical_ai_runtime.perception.grasp_verify_scene_observer import GraspVerifySceneObserver
from physical_ai_runtime.planning.workspace_observer import WorkspaceObserver
from physical_ai_runtime.manipulation.grasp_evaluator import GraspEvaluator
from physical_ai_runtime.core.feedback import SemanticFeedback
from physical_ai_runtime.core.status import RuntimeStatus
from physical_ai_runtime.perception.camera_manager import CameraManager
from physical_ai_runtime.planning.execution_authorization import (
    ExecutionAuthorizationGate,
)
from physical_ai_runtime.ai.semantic_manipulation_task import (
    CLOSE,
    MOVE_TO,
    OPEN,
    STOP,
    SemanticTaskProgram,
)


class RobotRuntime:

    def __init__(self):

        self.frames = FrameTransform()

        # Valid only after a grasp reaches CONTACT_ACCEPTABLE.
        self.active_grasp_context = None
        self.last_perception_source = None

        self.frames.register_robot(
            "ur5e",
            np.array([0.60, 0.05, 0.35])
        )

        self.frames.register_robot(
            "panda",
            np.array([0.45, 0.00, 0.45])
        )

    @staticmethod
    def _result_success(result):

        if isinstance(result, dict):
            return bool(
                result.get("success", False)
            )

        return bool(
            getattr(result, "success", False)
        )

    @staticmethod
    def _failure_reason(result):

        if isinstance(result, dict):
            return result.get(
                "failure_reason"
            )

        return getattr(
            result,
            "failure_reason",
            None
        )

    @staticmethod
    def _get_gripper_aperture_limits(robot):
        getter = getattr(
            robot,
            "get_gripper_aperture_limits",
            None,
        )

        if not callable(getter):
            return None

        limits = getter()

        if not isinstance(limits, dict):
            return None

        try:
            minimum = float(limits["min_aperture"])
            maximum = float(limits["max_aperture"])
        except (KeyError, TypeError, ValueError):
            return None

        if not (
            np.isfinite(minimum)
            and np.isfinite(maximum)
            and 0.0 <= minimum <= maximum
        ):
            return None

        return {
            "min_aperture": minimum,
            "max_aperture": maximum,
        }

    @staticmethod
    def _move_gripper_aperture(
        robot,
        *,
        aperture,
        tolerance,
        timeout,
    ):
        semantic_command = getattr(
            robot,
            "move_gripper_aperture",
            None,
        )

        if callable(semantic_command):
            result = semantic_command(
                aperture,
                tolerance=tolerance,
                timeout=timeout,
            )

            if result is not None:
                return result

        # Compatibility for adapters that predate semantic aperture control.
        # Their existing move_gripper contract accepts per-finger position.
        legacy_command = getattr(robot, "move_gripper", None)

        if not callable(legacy_command):
            return {
                "success": False,
                "failure_reason":
                    "GRIPPER_APERTURE_CONTROL_UNAVAILABLE",
            }

        return legacy_command(
            float(aperture) / 2.0,
            tolerance=float(tolerance) / 2.0,
            timeout=timeout,
        )

    @staticmethod
    def _execute_authorized_trajectory(
        robot,
        trajectory,
        planned_at_monotonic,
    ):
        """Bind a planned trajectory to the final safety authorization."""
        RuntimeStatus.safety(
            "Checking trajectory authorization..."
        )

        gate = ExecutionAuthorizationGate(robot)
        authorization = gate.authorize(
            trajectory,
            planned_at_monotonic=planned_at_monotonic,
        )

        if not authorization.authorized:
            RuntimeStatus.fail(
                "Trajectory authorization denied: "
                + str(authorization.reason)
            )
            return None, (
                "EXECUTION_AUTHORIZATION_DENIED:"
                + authorization.reason
            )

        RuntimeStatus.ok(
            "Trajectory authorized"
        )

        verified, reason = gate.verify_authorization(
            trajectory,
            authorization,
        )

        if not verified:
            RuntimeStatus.fail(
                "Trajectory authorization became invalid: "
                + str(reason)
            )
            return None, (
                "EXECUTION_AUTHORIZATION_INVALID:"
                + reason
            )

        RuntimeStatus.motion(
            "Executing authorized trajectory..."
        )

        result = robot.execute_planned_trajectory(
            trajectory
        )

        if getattr(result, "success", False):
            RuntimeStatus.ok(
                "Trajectory execution completed"
            )
        else:
            reason = getattr(
                result,
                "failure_reason",
                "UNKNOWN_MOTION_FAILURE",
            )

            RuntimeStatus.fail(
                "Trajectory execution failed: "
                + str(reason)
            )

        return result, None

    @staticmethod
    def _task_pose_in_robot_base(robot, task):
        """Resolve an AI task-space pose with live embodiment TF."""
        position = np.asarray(task.position, dtype=float)

        if task.orientation is None:
            _, orientation = robot.get_tcp_pose()
        else:
            orientation = np.asarray(task.orientation, dtype=float)

        if task.frame_id == robot.base_frame:
            return {
                "success": True,
                "position": position,
                "orientation": orientation,
            }

        try:
            transform = robot.tf_buffer.lookup_transform(
                robot.base_frame,
                task.frame_id,
                rclpy.time.Time(),
            )
        except Exception as exc:
            return {
                "success": False,
                "failure_reason": "TASK_FRAME_TF_UNAVAILABLE:" + str(exc),
            }

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        matrix = Rotation.from_quat([
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        ]).as_matrix()
        target = matrix @ position + np.array([
            translation.x,
            translation.y,
            translation.z,
        ])

        if task.orientation is not None:
            orientation = (
                Rotation.from_matrix(matrix)
                * Rotation.from_quat(orientation)
            ).as_quat()

        return {
            "success": True,
            "position": target,
            "orientation": orientation,
        }

    def _plan_world_vertical_motion(
        self,
        robot,
        *,
        distance,
    ):
        """Plan a semantic world-up lift from measured embodiment state."""
        distance = float(distance)

        if not np.isfinite(distance) or distance <= 0.0:
            return {
                "success": False,
                "failure_reason": "INVALID_LIFT_DISTANCE",
            }

        get_up = getattr(
            robot,
            "get_world_up_vector_in_base",
            None,
        )

        if not callable(get_up):
            return {
                "success": False,
                "failure_reason": "WORLD_UP_VECTOR_UNAVAILABLE",
            }

        try:
            start_tcp, orientation = robot.get_tcp_pose()
            world_up_in_base = np.asarray(
                get_up(),
                dtype=float,
            )
        except Exception as exc:
            return {
                "success": False,
                "failure_reason": (
                    "WORLD_UP_VECTOR_UNAVAILABLE:" + str(exc)
                ),
            }

        if (
            world_up_in_base.shape != (3,)
            or not np.all(np.isfinite(world_up_in_base))
        ):
            return {
                "success": False,
                "failure_reason": "WORLD_UP_VECTOR_INVALID",
            }

        norm = float(np.linalg.norm(world_up_in_base))

        if norm <= 1e-9:
            return {
                "success": False,
                "failure_reason": "WORLD_UP_VECTOR_INVALID",
            }

        world_up_in_base /= norm
        start_tcp = np.asarray(start_tcp, dtype=float)
        orientation = np.asarray(orientation, dtype=float)
        attempted_distances = []
        plan = None
        target = None

        # Exact intent is always preferred. Smaller candidates are only used
        # when the embodiment cannot produce a collision-free IK solution.
        for candidate_distance in (
            distance,
            distance * 0.75,
            distance * 0.5,
            distance * 0.25,
        ):
            if candidate_distance in attempted_distances:
                continue

            attempted_distances.append(candidate_distance)
            target = (
                start_tcp
                + world_up_in_base * candidate_distance
            )
            candidate_plan = robot.plan_tcp_pose(
                target=target,
                orientation=orientation,
            )

            if candidate_plan.get("success", False):
                plan = candidate_plan
                break

        if plan is None:
            plan = candidate_plan

        return {
            "success": bool(plan.get("success", False)),
            "failure_reason": plan.get("failure_reason"),
            "plan": plan,
            "start_tcp": start_tcp,
            "lift_target": target,
            "world_up_in_base": world_up_in_base,
            "requested_distance": distance,
            "planned_distance": (
                None
                if not plan.get("success", False)
                else candidate_distance
            ),
            "attempted_distances": attempted_distances,
        }

    def execute_semantic_program(
        self,
        robot,
        program,
        *,
        execute=False,
        perception_manager=None,
    ):
        """Plan or execute a generic AI task-space program safely."""
        if not isinstance(program, SemanticTaskProgram):
            return TaskResult(
                success=False,
                results=[],
                failed_step=0,
                failure_reason="INVALID_SEMANTIC_TASK_PROGRAM",
            )

        results = []

        for index, task in enumerate(program.steps):

            if task.action in {STOP, OPEN, CLOSE} and not execute:
                results.append({
                    "success": True,
                    "planned": True,
                    "action": task.action,
                })
                continue

            if task.action == STOP:
                result = robot.stop()
                results.append(result)

                if not self._result_success(result):
                    return TaskResult(
                        success=False,
                        results=results,
                        failed_step=index,
                        failure_reason=(
                            self._failure_reason(result)
                            or "STOP_FAILED"
                        ),
                    )

                continue

            if task.action in {OPEN, CLOSE}:
                result = self.execute(
                    robot,
                    Release() if task.action == OPEN else Grasp(),
                )
                results.append(result)

                if not self._result_success(result):
                    return TaskResult(
                        success=False,
                        results=results,
                        failed_step=index,
                        failure_reason=(
                            self._failure_reason(result)
                            or (task.action + "_FAILED")
                        ),
                    )

                self.active_grasp_context = None
                continue

            if task.action != MOVE_TO:
                return TaskResult(
                    success=False,
                    results=results,
                    failed_step=index,
                    failure_reason=(
                        "SEMANTIC_TASK_ACTION_NOT_IMPLEMENTED:"
                        + task.action
                    ),
                )

            pose = self._task_pose_in_robot_base(robot, task)
            if not pose["success"]:
                return TaskResult(
                    success=False,
                    results=results,
                    failed_step=index,
                    failure_reason=pose["failure_reason"],
                )

            plan = robot.plan_tcp_pose(
                target=pose["position"],
                orientation=pose["orientation"],
            )
            results.append(plan)

            if not plan.get("success", False):
                return TaskResult(
                    success=False,
                    results=results,
                    failed_step=index,
                    failure_reason=plan.get("failure_reason", "TASK_POSE_PLAN_FAILED"),
                )

            if not execute:
                continue

            execution, error = self._execute_authorized_trajectory(
                robot,
                plan["trajectory"],
                plan.get("planned_at_monotonic"),
            )
            if error is not None or not execution.success:
                return TaskResult(
                    success=False,
                    results=results + [execution],
                    failed_step=index,
                    failure_reason=error or execution.failure_reason,
                )
            results.append(execution)

        return TaskResult(success=True, results=results)


    def plan_grasp_object(
        self,
        robot,
        skill,
        scene=None
    ):

        if not isinstance(skill, GraspObject):
            return {
                "success": False,
                "failure_reason": "INVALID_GRASP_OBJECT_SKILL"
            }

        capabilities = robot.get_capabilities()

        if not capabilities.grasp:
            return {
                "success": False,
                "failure_reason": "UNSUPPORTED_CAPABILITY"
            }

        if scene is None:
            observer = RGBDSceneObserver(
                query=skill.object_id
            )

            try:
                scene = observer.observe_once(
                    timeout=5.0
                )
            finally:
                observer.destroy_node()

        if scene is None or skill.object_id not in scene.objects:
            return {
                "success": False,
                "failure_reason": "OBJECT_NOT_FOUND"
            }

        obj = scene.objects[
            skill.object_id
        ]

        target = np.array(
            obj.position_robot,
            dtype=float
        )

        tool_offset = robot.get_grasp_tool_offset()

        if tool_offset is None:
            return {
                "success": False,
                "failure_reason": "GRASP_TOOL_GEOMETRY_UNAVAILABLE"
            }

        if obj.height is None or obj.support_z is None:
            return {
                "success": False,
                "failure_reason": "OBJECT_GEOMETRY_UNAVAILABLE"
            }

        # position_robot currently represents the detected top surface.
        # Estimate object center from perceived support plane + height.
        object_center = target.copy()

        object_center[2] = (
            float(obj.support_z)
            + float(obj.height) / 2.0
        )

        pose_result = robot.find_reachable_grasp_pose(
            contact_point=object_center,
            support_z=float(obj.support_z),
            approach_distance=float(skill.approach_height),
        strategy=skill.strategy,
        )

        if not pose_result["success"]:
            return {
                "success": False,
                "failure_reason": pose_result[
                    "failure_reason"
                ]
            }

        grasp = pose_result[
            "grasp_target"
        ]

        approach = pose_result[
            "approach_target"
        ]

        approach_orientation = pose_result[
            "orientation"
        ]

        selected_tilt_deg = pose_result[
            "tilt_deg"
        ]

        # Lift is defined in the robot/world-up direction after grasp,
        # not along the tilted tool axis.
        lift = grasp.copy()
        lift[2] += float(
            skill.lift_height
        )

        return {
            "success": True,
            "skill": "GRASP_OBJECT_PLAN",
            "object_id": skill.object_id,
            "strategy": skill.strategy,
            "object_position_robot": target,
            "approach_target": approach,
            "grasp_target": grasp,
            "lift_target": lift,
            "approach_orientation": approach_orientation,
            "selected_tilt_deg": selected_tilt_deg,
            "actual_approach_distance": pose_result[
                "actual_approach_distance"
            ]
        }

    def reobserve_grasp_object(
        self,
        robot,
        skill,
        previous_scene,
        workspace,
        perception_manager=None,
    ):
        """
        Re-observe the target from the side verification camera after
        approach, using the previous main-camera observation as a
        spatial prior.
        """

        if (
            previous_scene is None
            or skill.object_id not in previous_scene.objects
        ):
            return {
                "success": False,
                "failure_reason": "PREVIOUS_SCENE_UNAVAILABLE"
            }

        previous_obj = previous_scene.objects[
            skill.object_id
        ]

        if perception_manager is not None:
            scene = perception_manager.observe_grasp_scene(
                query=skill.object_id,
                reference_world=previous_obj.position_world,
                reference_size=previous_obj.size_xyz,
                workspace=workspace,
                timeout=10.0,
            )
        else:
            observer = GraspVerifySceneObserver(
                query=skill.object_id,
                reference_world=previous_obj.position_world,
                reference_size=previous_obj.size_xyz,
                workspace=workspace
            )

            try:
                scene = observer.observe_once(
                    timeout=10.0
                )
            finally:
                observer.destroy_node()

        if (
            scene is None
            or skill.object_id not in scene.objects
        ):
            return {
                "success": False,
                "failure_reason": "VERIFY_OBJECT_NOT_FOUND"
            }

        obj = scene.objects[
            skill.object_id
        ]

        # Preserve stable object geometry from the main observation.
        obj.height = previous_obj.height
        obj.support_z = previous_obj.support_z

        # Keep fresh apparent size measured by the verification camera.
        # Only fall back to the previous size if the verify observer
        # could not estimate one.
        if obj.size_xyz is None:
            obj.size_xyz = previous_obj.size_xyz

        #
        # Sensor handoff fusion.
        #
        # Before grasp/contact the object should still be stationary.
        # The top RGB-D observation is therefore retained as the primary
        # planar position estimate, while the side camera independently
        # verifies that the object is still present.
        #
        # This prevents a camera-specific lateral calibration error from
        # shifting the grasp center during handoff.
        #
        verify_world = obj.position_world.copy()
        verify_robot = obj.position_robot.copy()

        previous_world = previous_obj.position_world.copy()
        previous_robot = previous_obj.position_robot.copy()

        planar_residual = float(
            np.linalg.norm(
                verify_world[:2]
                - previous_world[:2]
            )
        )

        self.last_grasp_verify_residual = planar_residual

        # Object has not been intentionally moved yet:
        # retain the better initial XY estimate.
        obj.position_world[0:2] = previous_world[0:2]
        obj.position_robot[0:2] = previous_robot[0:2]

        # Keep the side-camera Z/depth observation available through
        # the returned SceneObject, but grasp center Z continues to be
        # derived from support_z + object height.

        return {
            "success": True,
            "scene": scene
        }

    def verify_lifted_object(
        self,
        robot,
        skill,
        timeout=10.0,
        perception_manager=None,
    ):
        """
        Verify that the grasped target moved above its original
        support surface after lift.

        This is evidence-based:
        - previous verified object context
        - fresh side-camera observation
        - object height relative to original support surface

        It does not assume that TCP motion alone means object transport.
        """

        context = self.active_grasp_context

        if context is None:
            return {
                "success": False,
                "state": "LIFT_VERIFY_CONTEXT_MISSING",
            }

        if context.get("object_id") != skill.object_id:
            return {
                "success": False,
                "state": "LIFT_VERIFY_OBJECT_MISMATCH",
            }

        reference_world = np.asarray(
            context["object_position_world"],
            dtype=float,
        )

        reference_size = None

        # The target may partially overlap the gripper after lift,
        # so post-lift verification uses a more permissive self-mask
        # overlap threshold while still rejecting strong robot-only
        # detections.
        if perception_manager is not None:
            scene = perception_manager.observe_grasp_scene(
                query=skill.object_id,
                reference_world=reference_world,
                reference_size=reference_size,
                workspace=None,
                timeout=timeout,
                self_mask_overlap_reject=0.80,
            )
        else:
            observer = GraspVerifySceneObserver(
                query=skill.object_id,
                reference_world=reference_world,
                reference_size=reference_size,
                workspace=None,
                self_mask_overlap_reject=0.80,
            )

            try:
                scene = observer.observe_once(
                    timeout=timeout
                )
            finally:
                observer.destroy_node()

        if (
            scene is None
            or skill.object_id not in scene.objects
        ):
            return {
                "success": False,
                "state": "LIFT_VERIFY_OBJECT_NOT_VISIBLE",
            }

        obj = scene.objects[
            skill.object_id
        ]

        observed_z = float(
            obj.position_world[2]
        )

        support_z = float(
            context["support_z"]
        )

        object_height = float(
            context["height"]
        )

        original_center_z = (
            support_z
            + 0.5 * object_height
        )

        vertical_displacement = (
            observed_z
            - original_center_z
        )

        # Require evidence that the object is no longer resting at
        # its original support level. Threshold scales with the
        # perceived object height rather than hardcoding cube size.
        minimum_evidence = max(
            0.25 * object_height,
            0.005,
        )

        lifted = (
            vertical_displacement
            > minimum_evidence
        )

        return {
            "success": lifted,
            "state": (
                "LIFT_VISUALLY_CONFIRMED"
                if lifted
                else "LIFT_NOT_VISUALLY_CONFIRMED"
            ),
            "observed_world":
                np.asarray(
                    obj.position_world,
                    dtype=float,
                ),
            "vertical_displacement":
                float(vertical_displacement),
            "minimum_evidence":
                float(minimum_evidence),
        }

    def verify_active_grasp(
        self,
        robot,
        skill,
    ):
        """
        Verify the currently held object without commanding motion.
        Requires both persistent gripper contact and fresh visual evidence.
        """

        if (
            self.active_grasp_context is None
            or not isinstance(skill, GraspObject)
        ):
            return SemanticFeedback(
                state="GRASP_VERIFY_FAILED",
                object_id=getattr(
                    skill,
                    "object_id",
                    None,
                ),
                message=(
                    "Verified active grasp context "
                    "is unavailable."
                ),
                available_actions=[
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        if (
            self.active_grasp_context.get("object_id")
            != skill.object_id
        ):
            return SemanticFeedback(
                state="GRASP_VERIFY_FAILED",
                object_id=skill.object_id,
                message=(
                    "Active grasp context does not "
                    "match the requested object."
                ),
                available_actions=[
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        current_gripper = np.asarray(
            robot.current_gripper_position,
            dtype=float,
        )

        if current_gripper.size != 2:
            return SemanticFeedback(
                state="GRASP_VERIFY_UNCERTAIN",
                object_id=skill.object_id,
                message=(
                    "Current gripper contact state "
                    "is unavailable."
                ),
                available_actions=[
                    "REOBSERVE",
                    "ABORT",
                ],
            )

        evaluator = GraspEvaluator()

        contact = evaluator.evaluate_contact(
            object_id=skill.object_id,
            gripper_positions=current_gripper,
        )

        if contact.state not in {
            "CONTACT_ACCEPTABLE",
            "CONTACT_ASYMMETRIC",
        }:
            return SemanticFeedback(
                state="GRASP_VERIFY_FAILED",
                object_id=skill.object_id,
                message=(
                    "Persistent grasp contact "
                    "is no longer present."
                ),
                gripper_positions=current_gripper,
                metrics={
                    "contact_state":
                        contact.state,
                },
                available_actions=[
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        visual = self.verify_lifted_object(
            robot=robot,
            skill=skill,
            timeout=10.0,
        )

        if not visual["success"]:
            return SemanticFeedback(
                state="GRASP_VERIFY_UNCERTAIN",
                object_id=skill.object_id,
                message=(
                    "Gripper contact remains, but "
                    "fresh perception could not confirm "
                    "the lifted target."
                ),
                gripper_positions=current_gripper,
                metrics={
                    "contact_state":
                        contact.state,
                    "visual_state":
                        visual.get("state"),
                    "vertical_displacement":
                        visual.get(
                            "vertical_displacement"
                        ),
                },
                available_actions=[
                    "REOBSERVE",
                    "ABORT",
                ],
            )

        return SemanticFeedback(
            state="GRASP_VERIFIED",
            object_id=skill.object_id,
            message=(
                "Persistent contact and fresh visual "
                "evidence confirm the object remains held."
            ),
            object_pose=np.asarray(
                visual["observed_world"],
                dtype=float,
            ),
            gripper_positions=current_gripper,
            metrics={
                "contact_state":
                    contact.state,
                "vertical_displacement":
                    float(
                        visual["vertical_displacement"]
                    ),
            },
            available_actions=[],
        )

    def _execute_verified_grasp_vertical_motion(
        self,
        robot,
        skill,
        perception_manager=None,
    ):
        """
        Execute a semantic lift using the active grasp skill.
        Motion generation and safety remain inside the robot adapter.
        """

        if not isinstance(skill, GraspObject):
            return SemanticFeedback(
                state="LIFT_FAILED",
                object_id=getattr(
                    skill,
                    "object_id",
                    None,
                ),
                message="Active grasp context is unavailable.",
                available_actions=[
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        if self.active_grasp_context is None:
            return SemanticFeedback(
                state="LIFT_FAILED",
                object_id=skill.object_id,
                message=(
                    "No verified active grasp context "
                    "is available for lift."
                ),
                available_actions=[
                    "REOBSERVE",
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        if (
            self.active_grasp_context.get(
                "object_id"
            )
            != skill.object_id
        ):
            return SemanticFeedback(
                state="LIFT_FAILED",
                object_id=skill.object_id,
                message=(
                    "Active grasp context does not "
                    "match the requested object."
                ),
                available_actions=[
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        lift = self._plan_world_vertical_motion(
            robot,
            distance=skill.lift_height,
        )

        if not lift["success"]:
            return SemanticFeedback(
                state="LIFT_FAILED",
                object_id=skill.object_id,
                message=(
                    "Collision-aware lift planning failed."
                ),
                metrics={
                    "failure_reason": lift["failure_reason"],
                    "requested_lift_distance": lift.get(
                        "requested_distance"
                    ),
                    "attempted_lift_distances": lift.get(
                        "attempted_distances"
                    ),
                    "lift_start_tcp": lift.get("start_tcp"),
                    "lift_target": lift.get("lift_target"),
                    "world_up_in_base": lift.get(
                        "world_up_in_base"
                    ),
                },
                available_actions=[
                    "RETURN_READY",
                    "REOBSERVE",
                    "ABORT",
                ],
            )

        result, authorization_error = (
            self._execute_authorized_trajectory(
                robot,
                lift["plan"]["trajectory"],
                lift["plan"].get(
                    "planned_at_monotonic"
                ),
            )
        )

        if authorization_error is not None:
            return SemanticFeedback(
                state="LIFT_FAILED",
                object_id=skill.object_id,
                message="Lift trajectory execution was denied.",
                metrics={
                    "failure_reason": authorization_error,
                    "requested_lift_distance": lift.get(
                        "requested_distance"
                    ),
                    "planned_lift_distance": lift.get(
                        "planned_distance"
                    ),
                    "lift_start_tcp": lift["start_tcp"],
                    "lift_target": lift["lift_target"],
                    "world_up_in_base": lift[
                        "world_up_in_base"
                    ],
                },
                available_actions=[
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        if not self._result_success(result):
            return SemanticFeedback(
                state="LIFT_FAILED",
                object_id=skill.object_id,
                message="Lift trajectory execution failed.",
                metrics={
                    "failure_reason": self._failure_reason(result),
                    "requested_lift_distance": lift.get(
                        "requested_distance"
                    ),
                    "planned_lift_distance": lift.get(
                        "planned_distance"
                    ),
                    "lift_start_tcp": lift["start_tcp"],
                    "lift_target": lift["lift_target"],
                    "world_up_in_base": lift[
                        "world_up_in_base"
                    ],
                },
                available_actions=[
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        #
        # Verify that persistent contact still exists after lift.
        #
        current_gripper = np.asarray(
            robot.current_gripper_position,
            dtype=float,
        )

        if current_gripper.size != 2:
            return SemanticFeedback(
                state="LIFT_FAILED",
                object_id=skill.object_id,
                message=(
                    "Post-lift gripper contact state "
                    "is unavailable."
                ),
                available_actions=[
                    "REOBSERVE",
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        evaluator = GraspEvaluator()

        contact_feedback = (
            evaluator.evaluate_contact(
                object_id=skill.object_id,
                gripper_positions=current_gripper,
            )
        )

        if contact_feedback.state not in {
            "CONTACT_ACCEPTABLE",
            "CONTACT_ASYMMETRIC",
        }:
            return SemanticFeedback(
                state="LIFT_FAILED",
                object_id=skill.object_id,
                message=(
                    "Persistent grasp contact was "
                    "not maintained after lift."
                ),
                gripper_positions=
                    current_gripper,
                metrics={
                    "contact_state":
                        contact_feedback.state,
                    "lift_height_requested":
                        float(
                            skill.lift_height
                        ),
                    "lift_error":
                        float(result.error),
                },
                available_actions=[
                    "REOBSERVE",
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        #
        # Verify object transport with fresh perception.
        #
        visual = self.verify_lifted_object(
            robot=robot,
            skill=skill,
            timeout=10.0,
            perception_manager=perception_manager,
        )

        if not visual["success"]:
            return SemanticFeedback(
                state="LIFT_VERIFY_UNCERTAIN",
                object_id=skill.object_id,
                message=(
                    "Lift motion completed and grasp contact remains, "
                    "but object transport was not visually confirmed."
                ),
                gripper_positions=current_gripper,
                metrics={
                    "lift_height_requested":
                        float(skill.lift_height),
                    "lift_error":
                        float(result.error),
                    "post_lift_contact_state":
                        contact_feedback.state,
                    "visual_state":
                        visual.get("state"),
                    "vertical_displacement":
                        visual.get(
                            "vertical_displacement"
                        ),
                    "minimum_evidence":
                        visual.get(
                            "minimum_evidence"
                        ),
                },
                available_actions=[
                    "VERIFY_GRASP",
                    "REOBSERVE",
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        return SemanticFeedback(
            state="OBJECT_LIFTED",
            object_id=skill.object_id,
            message=(
                "Object lift was confirmed by robot motion, "
                "persistent grasp contact, and fresh visual evidence."
            ),
            object_pose=np.asarray(
                visual["observed_world"],
                dtype=float,
            ),
            gripper_positions=current_gripper,
            metrics={
                "lift_height_requested":
                    float(skill.lift_height),
                "lift_error":
                    float(result.error),
                "post_lift_contact_state":
                    contact_feedback.state,
                "vertical_displacement":
                    float(
                        visual[
                            "vertical_displacement"
                        ]
                    ),
                "minimum_evidence":
                    float(
                        visual[
                            "minimum_evidence"
                        ]
                    ),
            },
            available_actions=[
                "VERIFY_GRASP",
                "REOBSERVE",
                "RETURN_READY",
            ],
        )

    def execute_grasp_attempt(
        self,
        robot,
        skill,
        max_verify_planar_residual=0.03,
        initial_scene=None,
        perception_manager=None,
    ):
        """
        Execute one perception-driven, collision-aware grasp attempt.

        Arm motion is always planned through MoveIt before execution.
        Recovery policy is not handled here; this method returns
        SemanticFeedback for the agent/runtime recovery loop.
        """

        if not isinstance(skill, GraspObject):
            return SemanticFeedback(
                state="REPLAN_REQUIRED",
                object_id=getattr(
                    skill,
                    "object_id",
                    None
                ),
                message="Invalid grasp skill.",
                available_actions=[
                    "ABORT"
                ],
            )

        capabilities = robot.get_capabilities()

        if not capabilities.grasp:
            return SemanticFeedback(
                state="REPLAN_REQUIRED",
                object_id=skill.object_id,
                message=(
                    "Robot does not expose grasp capability."
                ),
                available_actions=[
                    "ABORT"
                ],
            )

        RuntimeStatus.perception(
            f"Verifying target: {skill.object_id}"
        )

        #
        # 1. Target perception.
        #
        # Reuse the fresh, manipulation-safe SceneState already verified
        # immediately before entering this execution function.
        #
        if (
            initial_scene is not None
            and skill.object_id in initial_scene.objects
        ):
            main_scene = initial_scene
            observation_status = "OBJECT_VISIBLE"
            self.last_perception_source = "verified_initial_scene"

            RuntimeStatus.ok(
                "Using fresh verified target geometry"
            )

        else:
            main = RGBDSceneObserver(
                query=skill.object_id
            )

            try:
                main_scene = main.observe_once(
                    timeout=10.0
                )
            finally:
                main.destroy_node()

            observation_status = getattr(
                main,
                "last_observation_status",
                None
            )

            if (
                main_scene is None
                or skill.object_id not in main_scene.objects
            ):
                return SemanticFeedback(
                    state="OBJECT_LOST",
                    object_id=skill.object_id,
                    message="Target object was not found.",
                    metrics={
                        "observation_status":
                            observation_status
                    },
                    available_actions=[
                        "REOBSERVE",
                        "RETURN_READY",
                        "ABORT",
                    ],
                )

        #
        # 2. Detect workspace from perception.
        #
        workspace_observer = WorkspaceObserver()

        try:
            ws = workspace_observer.observe_once(
                timeout=5.0
            )
        finally:
            workspace_observer.destroy_node()

        if ws is None:
            return SemanticFeedback(
                state="REPLAN_REQUIRED",
                object_id=skill.object_id,
                message=(
                    "Workspace could not be observed."
                ),
                available_actions=[
                    "REOBSERVE",
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        workspace = {
            "x_min": ws["x_min"],
            "x_max": ws["x_max"],
            "y_min": ws["y_min"],
            "y_max": ws["y_max"],
            "z": ws["table_z"],
        }

        RuntimeStatus.ok(
            f"Target verified: {skill.object_id}"
        )
        RuntimeStatus.geometry(
            "Resolving grasp geometry..."
        )
        RuntimeStatus.planner(
            "Generating initial grasp plan..."
        )

        #
        # 3. Initial semantic grasp plan.
        #
        initial_plan = self.plan_grasp_object(
            robot,
            skill,
            scene=main_scene
        )

        if not initial_plan["success"]:
            return SemanticFeedback(
                state="REPLAN_REQUIRED",
                object_id=skill.object_id,
                message=(
                    "Initial grasp plan failed."
                ),
                metrics={
                    "failure_reason":
                        initial_plan[
                            "failure_reason"
                        ]
                },
                available_actions=[
                    "REOBSERVE",
                    "CHANGE_GRASP_STRATEGY",
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        approach_target = np.asarray(
            initial_plan[
                "approach_target"
            ],
            dtype=float
        )

        approach_orientation = np.asarray(
            initial_plan[
                "approach_orientation"
            ],
            dtype=float
        )

        RuntimeStatus.ok(
            "Initial grasp geometry available"
        )
        RuntimeStatus.info(
            "Opening gripper..."
        )

        #
        # 4. Open gripper using embodiment limits.
        #
        gripper_limits = self._get_gripper_aperture_limits(
            robot
        )

        if gripper_limits is None:
            return SemanticFeedback(
                state="RECOVERY_REQUIRED",
                object_id=skill.object_id,
                message="Gripper aperture control is unavailable.",
                metrics={
                    "failure_reason":
                        "GRIPPER_APERTURE_CONTROL_UNAVAILABLE"
                },
                available_actions=["RETURN_READY", "ABORT"],
            )

        open_result = self._move_gripper_aperture(
            robot,
            aperture=gripper_limits["max_aperture"],
            tolerance=0.004,
            timeout=3.0,
        )

        if not self._result_success(
            open_result
        ):
            return SemanticFeedback(
                state="RECOVERY_REQUIRED",
                object_id=skill.object_id,
                message=(
                    "Gripper could not reach "
                    "the requested open state."
                ),
                metrics={
                    "failure_reason":
                        self._failure_reason(
                            open_result
                        )
                },
                available_actions=[
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        RuntimeStatus.planner(
            "Planning collision-aware approach..."
        )

        #
        # 5. Collision-aware approach planning.
        #
        approach_plan = robot.plan_tcp_pose(
            target=approach_target,
            orientation=
                approach_orientation,
        )

        if not approach_plan["success"]:
            return SemanticFeedback(
                state="MOTION_FAILED",
                object_id=skill.object_id,
                message=(
                    "Collision-aware approach "
                    "planning failed."
                ),
                metrics={
                    "failure_reason":
                        approach_plan[
                            "failure_reason"
                        ]
                },
                available_actions=[
                    "REOBSERVE",
                    "CHANGE_GRASP_STRATEGY",
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        approach_result, authorization_error = (
            self._execute_authorized_trajectory(
                robot,
                approach_plan["trajectory"],
                approach_plan.get(
                    "planned_at_monotonic",
                ),
            )
        )

        if authorization_error is not None:
            return SemanticFeedback(
                state="MOTION_FAILED",
                object_id=skill.object_id,
                message="Approach trajectory execution was denied.",
                metrics={
                    "failure_reason": authorization_error,
                },
                available_actions=[
                    "RETURN_READY",
                    "REOBSERVE",
                    "ABORT",
                ],
            )

        if not approach_result.success:
            return SemanticFeedback(
                state="MOTION_FAILED",
                object_id=skill.object_id,
                message=(
                    "Approach trajectory "
                    "execution failed."
                ),
                metrics={
                    "failure_reason":
                        approach_result.failure_reason
                },
                available_actions=[
                    "RETURN_READY",
                    "REOBSERVE",
                    "ABORT",
                ],
            )

        #
        # 6-7. Side-camera verification and grasp replanning.
        #
        RuntimeStatus.verify(
            "Re-observing target after approach..."
        )

        verify = self.reobserve_grasp_object(
            robot=robot,
            skill=skill,
            previous_scene=main_scene,
            workspace=workspace,
            perception_manager=perception_manager,
        )

        if not verify["success"]:
            return SemanticFeedback(
                state="OBJECT_LOST",
                object_id=skill.object_id,
                message=(
                    "Target could not be verified "
                    "after approach."
                ),
                metrics={
                    "failure_reason":
                        verify["failure_reason"]
                },
                available_actions=[
                    "REOBSERVE",
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        verify_scene = verify["scene"]

        obj = verify_scene.objects[
            skill.object_id
        ]

        object_center = np.asarray(
            obj.position_robot,
            dtype=float,
        ).copy()

        object_center[2] = (
            float(obj.support_z)
            + float(obj.height) / 2.0
        )

        replan = robot.find_reachable_grasp_pose(
            contact_point=object_center,
            support_z=float(obj.support_z),
            approach_distance=float(
                skill.approach_height
            ),
            strategy=skill.strategy,
        )

        if not replan["success"]:
            return SemanticFeedback(
                state="REPLAN_REQUIRED",
                object_id=skill.object_id,
                message=(
                    "Verified grasp pose "
                    "could not be generated."
                ),
                metrics={
                    "failure_reason":
                        replan["failure_reason"]
                },
                available_actions=[
                    "CHANGE_GRASP_STRATEGY",
                    "REOBSERVE",
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        #
        # 8. Collision-aware descend planning.
        #
        descend_plan = robot.plan_tcp_pose(
            target=np.asarray(
                replan[
                    "grasp_target"
                ],
                dtype=float
            ),
            orientation=np.asarray(
                replan[
                    "orientation"
                ],
                dtype=float
            ),
        )

        if not descend_plan["success"]:
            return SemanticFeedback(
                state="MOTION_FAILED",
                object_id=skill.object_id,
                message=(
                    "Collision-aware descend "
                    "planning failed."
                ),
                metrics={
                    "failure_reason":
                        descend_plan[
                            "failure_reason"
                        ]
                },
                available_actions=[
                    "REOBSERVE",
                    "CHANGE_GRASP_STRATEGY",
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        descend_result, authorization_error = (
            self._execute_authorized_trajectory(
                robot,
                descend_plan["trajectory"],
                descend_plan.get(
                    "planned_at_monotonic",
                ),
            )
        )

        if authorization_error is not None:
            return SemanticFeedback(
                state="MOTION_FAILED",
                object_id=skill.object_id,
                message="Descend trajectory execution was denied.",
                metrics={
                    "failure_reason": authorization_error,
                },
                available_actions=[
                    "RETURN_READY",
                    "REOBSERVE",
                    "ABORT",
                ],
            )

        if not descend_result.success:
            return SemanticFeedback(
                state="MOTION_FAILED",
                object_id=skill.object_id,
                message=(
                    "Descend trajectory "
                    "execution failed."
                ),
                metrics={
                    "failure_reason":
                        descend_result.failure_reason
                },
                available_actions=[
                    "RETURN_READY",
                    "REOBSERVE",
                    "ABORT",
                ],
            )

        #
        # 9. Pre-close policy.
        #
        RuntimeStatus.verify(
            "Checking target before closing gripper..."
        )

        post_verify = self.reobserve_grasp_object(
            robot=robot,
            skill=skill,
            previous_scene=verify_scene,
            workspace=workspace,
            perception_manager=perception_manager,
        )

        if not post_verify["success"]:
            return SemanticFeedback(
                state="PERCEPTION_UNCERTAIN",
                object_id=skill.object_id,
                message=(
                    "Object could not be safely verified "
                    "before gripper close."
                ),
                metrics={
                    "failure_reason":
                        post_verify["failure_reason"]
                },
                available_actions=[
                    "REOBSERVE",
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        #
        # 10. Close gripper.
        #
        close_result = self._move_gripper_aperture(
            robot,
            aperture=gripper_limits["min_aperture"],
            tolerance=0.004,
            timeout=3.0,
        )

        gripper_positions = np.asarray(
            close_result.actual,
            dtype=float
        )

        if gripper_positions.size != 2:
            return SemanticFeedback(
                state="RECOVERY_REQUIRED",
                object_id=skill.object_id,
                message=(
                    "Gripper contact state "
                    "is unavailable."
                ),
                available_actions=[
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        #
        # 10. Convert contact into semantic feedback.
        #
        evaluator = GraspEvaluator()

        feedback = evaluator.evaluate_contact(
            object_id=
                skill.object_id,
            gripper_positions=
                gripper_positions,
        )

        if feedback.state in {
            "CONTACT_ACCEPTABLE",
            "CONTACT_ASYMMETRIC",
        }:
            post_scene = post_verify["scene"]
            post_obj = post_scene.objects[
                skill.object_id
            ]

            tcp_position, tcp_orientation = (
                robot.get_tcp_pose(
                    timeout=5.0
                )
            )

            self.active_grasp_context = {
                "object_id":
                    skill.object_id,
                "skill":
                    skill,
                "object_position_world":
                    np.asarray(
                        post_obj.position_world,
                        dtype=float,
                    ).copy(),
                "object_position_robot":
                    np.asarray(
                        post_obj.position_robot,
                        dtype=float,
                    ).copy(),
                "support_z":
                    float(post_obj.support_z),
                "height":
                    float(post_obj.height),
                "gripper_positions":
                    gripper_positions.copy(),
                "tcp_position":
                    np.asarray(
                        tcp_position,
                        dtype=float,
                    ).copy(),
                "tcp_orientation":
                    np.asarray(
                        tcp_orientation,
                        dtype=float,
                    ).copy(),
                "selected_tilt_deg":
                    float(
                        replan["tilt_deg"]
                    ),
            }

        # PICK means grasp plus a verified, safety-authorized world-up lift.
        if feedback.state in {
            "CONTACT_ACCEPTABLE",
            "CONTACT_ASYMMETRIC",
        }:
            RuntimeStatus.ok(
                "Gripper contact detected — attempting lift"
            )

            return self._execute_verified_grasp_vertical_motion(
                robot,
                skill,
                perception_manager=perception_manager,
            )

        #
        # Preserve context useful to the semantic agent.
        #
        if feedback.metrics is None:
            feedback.metrics = {}

        feedback.metrics.update({
            "selected_tilt_deg":
                float(
                    replan[
                        "tilt_deg"
                    ]
                ),
            "verify_planar_residual":
                float(
                    getattr(
                        self,
                        "last_grasp_verify_residual",
                        0.0
                    )
                ),
        })

        return feedback


    def execute_task(
        self,
        robot,
        task
    ):

        if not isinstance(
            task,
            TaskSequence
        ):
            return TaskResult(
                success=False,
                results=[],
                failed_step=0,
                failure_reason="INVALID_TASK"
            )

        if not task.skills:
            return TaskResult(
                success=False,
                results=[],
                failed_step=0,
                failure_reason="EMPTY_TASK"
            )

        results = []

        for step_index, skill in enumerate(
            task.skills,
            start=1
        ):

            result = self.execute(
                robot,
                skill
            )

            results.append(
                result
            )

            if not self._result_success(
                result
            ):

                if task.stop_on_failure:

                    return TaskResult(
                        success=False,
                        results=results,
                        failed_step=step_index,
                        failure_reason=
                            self._failure_reason(
                                result
                            )
                    )

        all_success = all(
            self._result_success(result)
            for result in results
        )

        return TaskResult(
            success=all_success,
            results=results,
            failed_step=None,
            failure_reason=None
        )

    def execute(
        self,
        robot,
        skill
    ):

        if isinstance(
            skill,
            MoveJoints
        ):

            capabilities = robot.get_capabilities()

            if not capabilities.move_joints:
                return {
                    "success": False,
                    "failure_reason":
                        "UNSUPPORTED_CAPABILITY"
                }

            positions = np.array(
                skill.positions,
                dtype=float
            )

            try:
                state = robot.get_state()
                expected_shape = (
                    state.joint_position.shape
                )
            except Exception:
                return {
                    "success": False,
                    "failure_reason":
                        "ROBOT_STATE_UNAVAILABLE"
                }

            if positions.shape != expected_shape:
                return {
                    "success": False,
                    "failure_reason":
                        "INVALID_JOINT_TARGET"
                }

            safe_joint_motion = getattr(
                robot,
                "plan_and_execute_joints",
                None
            )

            if safe_joint_motion is None:
                return {
                    "success": False,
                    "failure_reason":
                        "SAFE_JOINT_PLANNING_UNAVAILABLE"
                }

            return safe_joint_motion(
                positions
            )

        if isinstance(
            skill,
            (Grasp, Release)
        ):

            capabilities = robot.get_capabilities()

            if not capabilities.grasp:
                return {
                    "success": False,
                    "failure_reason":
                        "UNSUPPORTED_CAPABILITY"
                }

            aperture = float(skill.width)

            if aperture < 0.0 or aperture > 0.08:
                return {
                    "success": False,
                    "failure_reason":
                        "INVALID_GRIPPER_WIDTH"
                }

            return self._move_gripper_aperture(
                robot,
                aperture=aperture,
                tolerance=skill.tolerance,
                timeout=skill.timeout,
            )

        if isinstance(
            skill,
            MoveTCP
        ):

            capabilities = (
                robot.get_capabilities()
            )

            if not capabilities.move_tcp:
                return {
                    "success": False,
                    "failure_reason":
                        "UNSUPPORTED_CAPABILITY"
                }

            position = np.array(
                skill.position,
                dtype=float
            )

            if position.shape != (3,):
                return {
                    "success": False,
                    "failure_reason":
                        "INVALID_TARGET"
                }

            try:

                target = (
                    self.frames.to_robot_base(
                        robot_name=
                            robot.robot_name,

                        position=
                            position,

                        frame=
                            skill.frame
                    )
                )

            except RuntimeError as e:

                return {
                    "success": False,
                    "failure_reason":
                        str(e)
                }

            return robot.execute_move(
                x=target[0],
                y=target[1],
                z=target[2],

                tolerance=
                    skill.tolerance,

                timeout=
                    skill.timeout,

                duration=
                    skill.duration
            )

        return {
            "success": False,
            "failure_reason":
                "UNKNOWN_SKILL"
        }
