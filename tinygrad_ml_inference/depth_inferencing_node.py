import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from std_msgs.msg import Header
from ament_index_python.packages import get_package_share_directory
import os
from collections.abc import Callable
from tinygrad import Tensor, TinyJit
from tinygrad.helpers import Context
from tinygrad.nn.state import safe_load, load_state_dict
import numpy as np
from pathlib import Path
from mobileNetModels import mobileNetDepthMedium
from pprint import pprint


class DepthInferencingNode(Node):
    try:
        packagae_share_directory = get_package_share_directory("tinygrad_ml_inference")
    except:
        package_share_directory = Path.cwd()
    datatype= np.dtype(np.float32).itemsize

    def __init__(self,model:Callable, model_name:str, supported_backends=["cuda","nv","clang"]):
        super().__init__("depth_model_inferencing_to_PCL")
        self.declare_parameter('tinygrad_backend',"cuda")
        self.declare_parameter('model_file',"MobileNetV4_28.safetensors")
        self.supported_backends = supported_backends
        backend = self.get_parameter("tinygrad_backend")
        if str(backend.value).lower() in supported_backends:
            if str(backend.value).lower != "clang":
                os.environ[str(backend.value).upper()] = "1"
        self.imageSubscriber = self.create_subscription(Image, "CameraStream",self.get_depth,10)
        self.imageSubscriber = self.create_subscription(CameraInfo, "CameraInfo",self.get_camera_intrinsics,10)
        self.depthPublisher = self.create_publisher(PointCloud2, "DepthEstimations",10)
        self.bridge = CvBridge() # convert image to numpy array to run inference
        self.model = model
        load_state_dict(model, safe_load(f"{self.package_share_directory}/models/{model_name}"))
        self.fields = [PointField(name=n,offset=i*self.datatype, datatype=PointField.FLOAT32, count=1) for i,n in enumerate('xyz')]
        self.camera_info_msg = None

    @TinyJit
    @Tensor.test()
    def model_inference(self, x:Tensor)->Tensor:
        x = self.model(x)
        return x

    def get_camera_intrinsics(self, msg:CameraInfo)-> None:
        self.camera_info_msg = msg

    def get_depth(self, msg:Image)->None:
        msg_header = Header(frame_id=msg.header.frame_id)
        height:int
        width:int
        if self.camera_info_msg == None:
            '''short cut if a camera intrinsics method has been recieved'''
            self.get_logger().info("no camera info provided for pcl generation", throttle_duration_sec=1)
            return
        try:
            img_np:np.ndarray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.float32)
        except CvBridgeError as e:
            '''short cut as image cannot be proccessed'''
            self.get_logger().error(f"{e} cv bridge Error handling the msg is not processed", throttle_duration_sec=1)
            return
        height = msg.height
        width = msg.width
        result: Tensor = self.model_inference(Tensor(img_np).reshape(-1, 3, height, width)).numpy()
        _, _, height, width = result.shape
        points = self.convertToPointCloud(result)
        pcl_msg = PointCloud2(header=msg_header,
                                  width=1,
                                  height=height*width,
                                  point_step= self.datatype * 3,
                                  row_step= self.datatype * 3 * width * height,
                                  is_bigendian=False,
                                  is_dense=False,
                                  fields=self.fields,
                                  data=points.tobytes())
        self.depthPublisher.publish(pcl_msg)

    '''does not work with rgb pointclouds'''
    def convertToPointCloud(self,arr:np.ndarray)->np.ndarray:
        #TODO: benchmark faster implementations -> either numba jit or cython
        cx:float = self.camera_info_msg.k[2]
        cy:float = self.camera_info_msg.k[5]
        fx:float = self.camera_info_msg.k[0]
        fy:float = self.camera_info_msg.k[4]
        depth_scale:float = 1000.0
        _,_, height, width = arr.shape
        arr.reshape(width*height)
        def indexConversion(inp, width=width,height=height):
            return (inp%width,inp//width)
        pcl_list = []
        for d in range(arr.shape[0]):
            u,v = indexConversion(d)
            z = -arr[d]/depth_scale
            x = ((u - cx) * z)/fx
            y = ((v - cy) * z)/fy
            pcl_list.append([x, y, z])
        return np.array(pcl_list,dtype=np.float32)


if __name__ == "__main__":
    model = mobileNetDepthMedium()
    model_name = "MobileNetV4_28.safetensors"
    rclpy.init(args=None)
    node = DepthInferencingNode(model, model_name)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
