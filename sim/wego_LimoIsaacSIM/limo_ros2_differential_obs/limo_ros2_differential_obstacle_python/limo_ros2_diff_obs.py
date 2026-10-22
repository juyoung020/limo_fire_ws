import os
import numpy as np
import omni
import carb
from pxr import UsdLux, Sdf, UsdGeom, Gf, UsdPhysics, PhysxSchema
from isaacsim.core.prims import SingleArticulation, XFormPrim
from isaacsim.core.api import World
import omni.graph.core as og

from isaacsim.core.utils.stage import get_current_stage, add_reference_to_stage
from isaacsim.core.api.objects import DynamicCuboid

from isaacsim.examples.interactive.base_sample import BaseSample

class CubeSpec:
    def __init__(self, name, color, position):
        self.name = name
        self.color = np.array(color)
        self.position = np.array(position)

class LimoROS2DiffObs(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        
        extension_enabled = omni.kit.app.get_app().get_extension_manager().is_extension_enabled("isaacsim.ros2.bridge")
        if not extension_enabled:
            msg = "ROS2 Bridge is not enabled. Please enable the extension to use this feature."
            carb.log_error(msg)
        
        # set the diretory 
        self._base_dir = os.path.dirname(os.path.abspath(__file__))
        self._path_to_robot_usd = os.path.join(self._base_dir, "../../USD/WegoLimo/Limo/limo_diff_thin.usd")
        self._path_to_track_usd = os.path.join(self._base_dir, "../../USD/WegoLimo/LimoTrack/LIMO_simulation_table.usd")
        
        # set the Prim Path
        self._robot_prim_path = "/World/Limo"
        self._track_prim_path = "/World/LimoTrack"
        self._articulation = None
        
        # robot setup
        self._maxLinearSpeed = 1e6
        self._wheelDistance = 0.43
        self._wheelRadius = 0.045
        self._front_jointNames = ["rear_left_wheel", "rear_right_wheel"]
        self._rear_jointNames = ["front_left_wheel", "front_right_wheel"]
        self._camera_tf_name = "depth_link"
        self._lidar_tf_name = "laser_link"

        self._basePrim= "/World/Limo/base_link"
        self._imuPrim= "/World/Limo/base_link/Imu_Sensor"
        self._cameraPath = "/World/Limo/depth_link/limo_camera"
        self._lidarPath = "/World/Limo/laser_link/RPLidar_S2e/RPLIDAR_S2E"
        self._sensorPrims = ["/World/Limo/depth_link", "/World/Limo/laser_link", "/World/Limo/imu_link"]
        
        # cube setup
        self._width = 0.15
        
        return

    def add_light(self):
        """
        A new stage does not have a light by default.  This function creates a spherical light
        """
        sphereLight = UsdLux.SphereLight.Define(get_current_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        XFormPrim(str(sphereLight.GetPath())).set_world_poses(np.array([[0, 0, 12]]))

    def add_background(self):
        add_reference_to_stage(usd_path=self._path_to_track_usd, prim_path=self._track_prim_path)
        bg_prim = self._stage.GetPrimAtPath(self._track_prim_path)
        if bg_prim.IsValid():
            bg_xform = UsdGeom.Xform(bg_prim)
            xform_ops = bg_xform.GetOrderedXformOps()
            existing_ops = {op.GetOpName() for op in xform_ops}
            
            if "xformOp:scale" in existing_ops:
                for op in xform_ops:
                    if op.GetOpName() == "xformOp:scale":
                        op.Set(Gf.Vec3f(0.01, 0.01, 0.01))
                        break
            else:
                bg_xform.AddScaleOp().Set(Gf.Vec3f(0.01, 0.01, 0.01))
            
            if "xformOp:translate" in existing_ops:
                for op in xform_ops:
                    if op.GetOpName() == "xformOp:translate":
                        op.Set(Gf.Vec3f(0.28, 0.49, 0.01))
                        break
            else:
                bg_xform.AddTranslateOp().Set(Gf.Vec3f(0.28, 0.49, 0.01))
            
            quat = Gf.Quatf(0.707, Gf.Vec3f(0.0, 0.0, 0.707))
            if "xformOp:orient" in existing_ops:
                for op in xform_ops:
                    if op.GetOpName() == "xformOp:orient":
                        op.Set(quat)
                        break
            else:
                bg_xform.AddOrientOp().Set(quat)
        else:
            print("background prim invalid")
    
    def add_robot(self):
        add_reference_to_stage(self._path_to_robot_usd, self._robot_prim_path)
        self._articulation = SingleArticulation(self._robot_prim_path)
        self._world.scene.add(self._articulation)

    def add_physics(self):
        scene = UsdPhysics.Scene.Define(self._stage, Sdf.Path("/physicsScene"))
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)

        physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physx_scene_api.CreateSolverTypeAttr().Set("PGS")
        physx_scene_api.CreateEnableCCDAttr().Set(True)
        physx_scene_api.CreateEnableStabilizationAttr().Set(True)
        physx_scene_api.CreateEnableGPUDynamicsAttr().Set(False)
    
    def og_setup(self):
        try:
            og.Controller.edit(
                {"graph_path": "/ROS2DifferentialDrive", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("onPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("context", "isaacsim.ros2.bridge.ROS2Context"),
                        ("twistSubscriber", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                        ("linearVector", "omni.graph.nodes.BreakVector3"),
                        ("angularVector", "omni.graph.nodes.BreakVector3"),
                        ("stageUnit", "isaacsim.core.nodes.OgnIsaacScaleToFromStageUnit"),
                        ("differentialDriver", "isaacsim.robot.wheeled_robots.DifferentialController"),
                        ("frontArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                        ("rearArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("twistSubscriber.inputs:topicName", '/sim/cmd_vel'),
                        ("differentialDriver.inputs:maxLinearSpeed", self._maxLinearSpeed),
                        ("differentialDriver.inputs:wheelDistance", self._wheelDistance),
                        ("differentialDriver.inputs:wheelRadius", self._wheelRadius),
                        ("frontArticulationController.inputs:targetPrim", Sdf.Path(self._basePrim)),
                        ("rearArticulationController.inputs:targetPrim", Sdf.Path(self._basePrim)),
                        ("frontArticulationController.inputs:jointNames", self._front_jointNames),
                        ("rearArticulationController.inputs:jointNames", self._rear_jointNames),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("onPlaybackTick.outputs:tick", "twistSubscriber.inputs:execIn"),
                        ("context.outputs:context", "twistSubscriber.inputs:context"),
                        ("twistSubscriber.outputs:linearVelocity", "stageUnit.inputs:value"),
                        ("twistSubscriber.outputs:angularVelocity", "angularVector.inputs:tuple"),
                        ("stageUnit.outputs:result", "linearVector.inputs:tuple"),
                        ("linearVector.outputs:x", "differentialDriver.inputs:linearVelocity"),
                        ("angularVector.outputs:z", "differentialDriver.inputs:angularVelocity"),
                        ("onPlaybackTick.outputs:deltaSeconds", "differentialDriver.inputs:dt"),
                        ("onPlaybackTick.outputs:tick", "differentialDriver.inputs:execIn"),
                        ("differentialDriver.outputs:velocityCommand", "frontArticulationController.inputs:velocityCommand"),
                        ("differentialDriver.outputs:velocityCommand", "rearArticulationController.inputs:velocityCommand"),
                        ("onPlaybackTick.outputs:tick", "frontArticulationController.inputs:execIn"),
                        ("onPlaybackTick.outputs:tick", "rearArticulationController.inputs:execIn"),
                    ],
                },
            )
            
            og.Controller.edit(
                {"graph_path": "/ROS2Odom", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("onPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("context", "isaacsim.ros2.bridge.ROS2Context"),
                        ("readSimulationTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                        ("computeOdometry", "isaacsim.core.nodes.IsaacComputeOdometry"),
                        ("publishOdometry", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                        ("rawTFTree", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                        ("TFTree", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("computeOdometry.inputs:chassisPrim", Sdf.Path(self._basePrim)),
                        ("TFTree.inputs:parentPrim", Sdf.Path(self._basePrim)),
                        ("TFTree.inputs:targetPrims", self._sensorPrims),
                        ("publishOdometry.inputs:topicName", "/sim/odom"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("onPlaybackTick.outputs:tick", "computeOdometry.inputs:execIn"),
                        ("onPlaybackTick.outputs:tick", "rawTFTree.inputs:execIn"),
                        ("onPlaybackTick.outputs:tick", "TFTree.inputs:execIn"),
                        ("context.outputs:context", "publishOdometry.inputs:context"),
                        ("context.outputs:context", "rawTFTree.inputs:context"),
                        ("context.outputs:context", "TFTree.inputs:context"),
                        ("readSimulationTime.outputs:simulationTime", "rawTFTree.inputs:timeStamp"),
                        ("readSimulationTime.outputs:simulationTime", "TFTree.inputs:timeStamp"),
                        ("readSimulationTime.outputs:simulationTime", "publishOdometry.inputs:timeStamp"),
                        ("computeOdometry.outputs:execOut", "publishOdometry.inputs:execIn"),
                        ("computeOdometry.outputs:angularVelocity", "publishOdometry.inputs:angularVelocity"),
                        ("computeOdometry.outputs:linearVelocity", "publishOdometry.inputs:linearVelocity"),
                        ("computeOdometry.outputs:position", "publishOdometry.inputs:position"),
                        ("computeOdometry.outputs:orientation", "publishOdometry.inputs:orientation"),
                        ("computeOdometry.outputs:position", "rawTFTree.inputs:translation"),
                        ("computeOdometry.outputs:orientation", "rawTFTree.inputs:rotation"),
                    ],
                },
            )
            
            og.Controller.edit(
                {"graph_path": "/ROS2IMU", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("onPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("context", "isaacsim.ros2.bridge.ROS2Context"),
                        ("readSimulationTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                        ("readIMU", "isaacsim.sensors.physics.IsaacReadIMU"),
                        ("publishIMU", "isaacsim.ros2.bridge.ROS2PublishImu"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("readIMU.inputs:imuPrim", Sdf.Path(self._imuPrim)),
                        ("publishIMU.inputs:frameId", "imu_link"),
                        ("publishIMU.inputs:topicName", "/sim/imu"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("onPlaybackTick.outputs:tick", "readIMU.inputs:execIn"),
                        ("context.outputs:context", "publishIMU.inputs:context"),
                        ("readSimulationTime.outputs:simulationTime", "publishIMU.inputs:timeStamp"),
                        ("readIMU.outputs:execOut", "publishIMU.inputs:execIn"),
                        ("readIMU.outputs:angVel", "publishIMU.inputs:angularVelocity"),
                        ("readIMU.outputs:linAcc", "publishIMU.inputs:linearAcceleration"),
                        ("readIMU.outputs:orientation", "publishIMU.inputs:orientation"),
                    ],
                },
            )

            og.Controller.edit(
                {"graph_path": "/ROS2Camera", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("onPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("runOneSimulationFrame", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
                        ("rendererProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                        ("rosContext", "isaacsim.ros2.bridge.ROS2Context"),
                        ("RGBcameraHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("depthcameraHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("pccameraHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("cameraInfoHelper", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("rendererProduct.inputs:cameraPrim", Sdf.Path(self._cameraPath)),
                        ("RGBcameraHelper.inputs:frameId", self._camera_tf_name),
                        ("RGBcameraHelper.inputs:topicName", '/sim/camera/color/image_raw'),
                        ("RGBcameraHelper.inputs:type", 'rgb'),
                        ("depthcameraHelper.inputs:frameId", self._camera_tf_name),
                        ("depthcameraHelper.inputs:topicName", '/sim/camera/depth/image_raw'),
                        ("depthcameraHelper.inputs:type", 'depth'),
                        ("pccameraHelper.inputs:frameId", self._camera_tf_name),
                        ("pccameraHelper.inputs:topicName", '/sim/camera_points'),
                        ("pccameraHelper.inputs:type", 'depth_pcl'),
                        ("cameraInfoHelper.inputs:frameId", self._camera_tf_name),
                        ("cameraInfoHelper.inputs:topicName", '/sim/camera/camera_info'),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("onPlaybackTick.outputs:tick", "runOneSimulationFrame.inputs:execIn"),
                        ("runOneSimulationFrame.outputs:step", "rendererProduct.inputs:execIn"),
                        ("rendererProduct.outputs:execOut", "RGBcameraHelper.inputs:execIn"),
                        ("rendererProduct.outputs:renderProductPath", "RGBcameraHelper.inputs:renderProductPath"),
                        ("rosContext.outputs:context", "RGBcameraHelper.inputs:context"),
                        ("rendererProduct.outputs:execOut", "depthcameraHelper.inputs:execIn"),
                        ("rendererProduct.outputs:renderProductPath", "depthcameraHelper.inputs:renderProductPath"),
                        ("rosContext.outputs:context", "depthcameraHelper.inputs:context"),
                        ("rendererProduct.outputs:execOut", "pccameraHelper.inputs:execIn"),
                        ("rendererProduct.outputs:renderProductPath", "pccameraHelper.inputs:renderProductPath"),
                        ("rosContext.outputs:context", "pccameraHelper.inputs:context"),
                        ("rendererProduct.outputs:execOut", "cameraInfoHelper.inputs:execIn"),
                        ("rendererProduct.outputs:renderProductPath", "cameraInfoHelper.inputs:renderProductPath"),
                        ("rosContext.outputs:context", "cameraInfoHelper.inputs:context"),
                    ],
                },
            )

            og.Controller.edit(
                {"graph_path": "/ROS2Lidar", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("onPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("runOneSimulationFrame", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
                        ("rendererProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                        ("rosContext", "isaacsim.ros2.bridge.ROS2Context"),
                        ("RtxLidarHelper", "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("rendererProduct.inputs:cameraPrim", Sdf.Path(self._lidarPath)),
                        ("RtxLidarHelper.inputs:frameId", self._lidar_tf_name),
                        ("RtxLidarHelper.inputs:topicName", '/sim/scan'),
                        ("RtxLidarHelper.inputs:type", 'laser_scan'),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("onPlaybackTick.outputs:tick", "runOneSimulationFrame.inputs:execIn"),
                        ("runOneSimulationFrame.outputs:step", "rendererProduct.inputs:execIn"),
                        ("rendererProduct.outputs:execOut", "RtxLidarHelper.inputs:execIn"),
                        ("rendererProduct.outputs:renderProductPath", "RtxLidarHelper.inputs:renderProductPath"),
                        ("rosContext.outputs:context", "RtxLidarHelper.inputs:context"),
                    ],
                },
            )
            
            og.Controller.edit(
                {"graph_path": "/ROS2Clock", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("onPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("readSimulationTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                        ("rosContext", "isaacsim.ros2.bridge.ROS2Context"),
                        ("publishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("onPlaybackTick.outputs:tick", "publishClock.inputs:execIn"),
                        ("readSimulationTime.outputs:simulationTime", "publishClock.inputs:timeStamp"),
                        ("rosContext.outputs:context", "publishClock.inputs:context"),
                    ],
                },
            )
        except Exception as e:
            print(e)
    
    def add_obstacle(self):
        obs_specs = [
            CubeSpec("RedCube_1", [0.7, 0.0, 0.0], [0.76609, 0.51694, 0.1]),
            CubeSpec("GreenCube_1", [0.0, 0.7, 0.0], [0.30995, 0.84962, 0.1]),
            CubeSpec("RedCube_2", [0.7, 0.0, 0.0], [-0.40531, 0.84673, 0.1]),
            CubeSpec("BlueCube_2", [0.0, 0.0, 0.7], [1.16526, 0.0645, 0.1]),
            CubeSpec("RedCube_3", [0.7, 0.0, 0.0], [-0.52328, -0.53648, 0.1]),
            CubeSpec("GreenCube_3", [0.0, 0.7, 0.0], [-0.74971, -0.41684, 0.1]),
            CubeSpec("BlueCube_3", [0.0, 0.0, 0.7], [-0.0646, -0.90848, 0.1]),
            CubeSpec("RedCube_4", [0.7, 0.0, 0.0], [0.7189, -0.51325, 0.1]),
            CubeSpec("GreenCube_4", [0.0, 0.7, 0.0], [0.31817, -0.4963, 0.1]),
            CubeSpec("BlueCube_4", [0.0, 0.0, 0.7], [0.50956, -0.53134, 0.1]),
            CubeSpec("GreenCube_5", [0.0, 0.7, 0.0], [-1.09802, 0.02784, 0.1]),
            CubeSpec("RedCube_5", [0.7, 0.0, 0.0], [-0.60602, 0.42274, 0.1]),
            CubeSpec("BlueCube_5", [0.0, 0.0, 0.7], [-0.60098, 0.9886, 0.1]),
            CubeSpec("GreenCube_6", [0.0, 0.7, 0.0], [-1.02633, 1.28037, 0.1]),
            CubeSpec("GreenCube_7", [0.0, 0.7, 0.0], [0.89581, 1.76869, 0.1]),
            CubeSpec("BlueCube_6", [0.0, 0.0, 0.7], [0.87236, 1.53685, 0.1]),
            CubeSpec("GreenCube_8", [0.0, 0.7, 0.0], [-0.26329, 1.58919, 0.1]),
            CubeSpec("RedCube_7", [0.7, 0.0, 0.0], [-0.00822, 1.57711, 0.1]),
            CubeSpec("RedCube_8", [0.7, 0.0, 0.0], [1.25307, 1.13621, 0.1]),
            CubeSpec("GreenCube_9", [0.0, 0.7, 0.0], [1.30969, 1.40769, 0.1]),
            CubeSpec("RedCube_9", [0.7, 0.0, 0.0], [1.61672, 0.85947, 0.1]),
        ]
        
        for spec in obs_specs:
            obj = self._world.scene.add(
                DynamicCuboid(
                    prim_path="/World/Obs/{}".format(spec.name),
                    name=spec.name,
                    size=self._width,
                    color=spec.color,
                    position=spec.position
                )
            )
            
    def setup_scene(self):
        self._world = World.instance()
        self._stage = get_current_stage()
        
        self.add_physics()
        self.add_light()
        self.add_background()
        self.add_robot()
        self.og_setup()
        self.add_obstacle()
    
        return

    async def setup_post_load(self):
        return

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        return

    def world_cleanup(self):
        return
