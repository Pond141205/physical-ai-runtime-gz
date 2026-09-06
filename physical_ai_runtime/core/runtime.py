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
from physical_ai_runtime.perception.camera_manager import CameraManager
from physical_ai_runtime.planning.execution_authorization import (
    ExecutionAuthorizationGate,
)
from physical_ai_runtime.ai.semantic_manipulation_task import (
    MOVE_POSE,
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
    def _execute_authorized_trajectory(
        robot,
        trajectory,
        planned_at_monotonic,
    ):
        """Bind a planned trajectory to the final safety authorization."""
        gate = ExecutionAuthorizationGate(robot)
        authorization = gate.authorize(
            trajectory,
            planned_at_monotonic=planned_at_monotonic,
        )

        if not authorization.authorized:
            return None, (
                "EXECUTION_AUTHORIZATION_DENIED:"
                + authorization.reason
            )

        verified, reason = gate.verify_authorization(
            trajectory,
            authorization,
        )

        if not verified:
            return None, (
                "EXECUTION_AUTHORIZATION_INVALID:"
                + reason
            )

        return robot.execute_planned_trajectory(trajectory), None

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

    def execute_semantic_program(self, robot, program, *, execute=False):
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
            if task.action != MOVE_POSE:
                return TaskResult(
                    success=False,
                    results=results,
                    failed_step=index,
                    failure_reason="SEMANTIC_TASK_ACTION_NOT_IMPLEMENTED:" + task.action,
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
        workspace
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

    def execute_lift(
        self,
        robot,
        skill,
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

        if not hasattr(
            robot,
            "plan_and_execute_lift",
        ):
            return SemanticFeedback(
                state="LIFT_FAILED",
                object_id=skill.object_id,
                message=(
                    "Robot does not expose "
                    "collision-aware lift execution."
                ),
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

        result = robot.plan_and_execute_lift(
            distance=float(
                skill.lift_height
            )
        )

        if not result.success:
            return SemanticFeedback(
                state="LIFT_FAILED",
                object_id=skill.object_id,
                message=(
                    "Collision-aware lift failed."
                ),
                metrics={
                    "failure_reason":
                        result.failure_reason,
                    "lift_error":
                        (
                            None
                            if result.error is None
                            else float(result.error)
                        ),
                },
                available_actions=[
                    "RETURN_READY",
                    "REOBSERVE",
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

        #
        # 1. Fresh main-camera perception.
        #
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
            or skill.object_id
            not in main_scene.objects
        ):
            if observation_status == "SELF_OCCLUDED":
                #
                # Main view is known to be self-occluded.
                # Do NOT immediately retry the same camera.
                #
                # Ask CameraManager for an alternate sensor only.
                # At the moment this means the wrist RGB-D camera.
                #
                camera_manager = CameraManager(
                    query=skill.object_id,
                )

                alternate = (
                    camera_manager.observe_best(
                        timeout_per_camera=4.0,
                        include_main=False,
                        include_wrist=True,
                    )
                )

                if (
                    alternate.scene is not None
                    and alternate.status
                    == "OBJECT_VISIBLE"
                    and skill.object_id
                    in alternate.scene.objects
                ):
                    #
                    # Preserve the existing downstream grasp
                    # pipeline while changing only the perception
                    # source. World/object coordinates are produced
                    # through TF by the selected observer.
                    #
                    main_scene = alternate.scene

                    observation_status = (
                        "OBJECT_VISIBLE"
                    )

                    self.last_perception_source = (
                        alternate.camera
                    )

                else:
                    #
                    # Both the primary view and currently available
                    # alternate view failed to establish a safe
                    # target observation.
                    #
                    # REOBSERVE is intentionally NOT exposed here:
                    # repeating the same geometry can create an
                    # occlusion loop.
                    #
                    return SemanticFeedback(
                        state="OBJECT_OCCLUDED",
                        object_id=skill.object_id,
                        message=(
                            "The main RGB-D view is "
                            "self-occluded and the alternate "
                            "wrist view could not safely "
                            "recover the target."
                        ),
                        metrics={
                            "main_status":
                                "SELF_OCCLUDED",
                            "alternate_camera":
                                alternate.camera,
                            "alternate_status":
                                alternate.status,
                            "alternate_metrics":
                                alternate.metrics,
                        },
                        available_actions=[
                            "RETURN_READY",
                            "ABORT",
                        ],
                    )

            if observation_status == "OBJECT_NOT_DETECTED":
                return SemanticFeedback(
                    state="OBJECT_LOST",
                    object_id=skill.object_id,
                    message=(
                        "Target object was not detected in the "
                        "main RGB-D observation."
                    ),
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

            return SemanticFeedback(
                state="OBJECT_LOST",
                object_id=skill.object_id,
                message=(
                    "Target object was not found "
                    "in the main RGB-D observation."
                ),
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

        #
        # 4. Open gripper using embodiment limits.
        #
        gripper_limits = (
            robot.get_gripper_limits()
        )

        open_position = float(
            gripper_limits[
                "max_position"
            ]
        )

        open_result = robot.move_gripper(
            open_position
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
        # 6. Side-camera verification after approach.
        #
        verify = self.reobserve_grasp_object(
            robot=robot,
            skill=skill,
            previous_scene=main_scene,
            workspace=workspace,
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
                        verify[
                            "failure_reason"
                        ]
                },
                available_actions=[
                    "REOBSERVE",
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        verify_scene = verify["scene"]

        verify_residual = float(
            getattr(
                self,
                "last_grasp_verify_residual",
                float("inf"),
            )
        )

        if (
            verify_residual
            > float(max_verify_planar_residual)
        ):
            return SemanticFeedback(
                state="PERCEPTION_UNCERTAIN",
                object_id=skill.object_id,
                message=(
                    "Verification camera disagreed "
                    "with the primary observation "
                    "beyond the allowed planar residual."
                ),
                metrics={
                    "verify_planar_residual":
                        verify_residual,
                    "max_verify_planar_residual":
                        float(
                            max_verify_planar_residual
                        ),
                },
                available_actions=[
                    "REOBSERVE",
                    "CHANGE_GRASP_STRATEGY",
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        obj = verify_scene.objects[
            skill.object_id
        ]

        if (
            obj.height is None
            or obj.support_z is None
        ):
            return SemanticFeedback(
                state="REPLAN_REQUIRED",
                object_id=skill.object_id,
                message=(
                    "Verified object geometry "
                    "is incomplete."
                ),
                available_actions=[
                    "REOBSERVE",
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        #
        # 7. Recompute grasp from fresh verified state.
        #
        object_center = np.asarray(
            obj.position_robot,
            dtype=float
        ).copy()

        object_center[2] = (
            float(obj.support_z)
            + float(obj.height) / 2.0
        )

        replan = (
            robot.find_reachable_grasp_pose(
                contact_point=
                    object_center,
                support_z=
                    float(obj.support_z),
                approach_distance=
                    float(
                        skill.approach_height
                    ),
                strategy=skill.strategy,
            )
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
                        replan[
                            "failure_reason"
                        ]
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
        # 9. Post-descend verification before closing.
        #
        post_verify = self.reobserve_grasp_object(
            robot=robot,
            skill=skill,
            previous_scene=verify_scene,
            workspace=workspace,
        )

        if not post_verify["success"]:
            return SemanticFeedback(
                state="PERCEPTION_UNCERTAIN",
                object_id=skill.object_id,
                message=(
                    "Object could not be safely verified "
                    "after descend. Gripper close was blocked."
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

        post_verify_residual = float(
            getattr(
                self,
                "last_grasp_verify_residual",
                float("inf"),
            )
        )

        if (
            post_verify_residual
            > float(max_verify_planar_residual)
        ):
            return SemanticFeedback(
                state="PERCEPTION_UNCERTAIN",
                object_id=skill.object_id,
                message=(
                    "Post-descend verification disagreed "
                    "with the expected object position. "
                    "Gripper close was blocked."
                ),
                metrics={
                    "verify_planar_residual":
                        post_verify_residual,
                    "max_verify_planar_residual":
                        float(
                            max_verify_planar_residual
                        ),
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
        close_position = float(
            gripper_limits[
                "min_position"
            ]
        )

        close_result = robot.move_gripper(
            close_position
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

        if feedback.state == "CONTACT_ACCEPTABLE":
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

            joint_target = aperture / 2.0

            return robot.move_gripper(
                joint_target,
                tolerance=skill.tolerance / 2.0,
                timeout=skill.timeout
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
