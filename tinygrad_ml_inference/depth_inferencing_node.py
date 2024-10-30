import rclpy
from rclpy import Node
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from std_msgs.msg import Header

import os
from tinygrad import Tensor, TinyJit
from tinygrad.helpers import Context
from tinygrad.nn.state import safe_load
import numpy as np
from typing import TypeAlias


#TODO replace the pointcloud convert with extrinsics

class DepthInferencingNode(Node):
    datatype= np.dtype(np.float32).itemsize
    def __init__(self, model_name, supported_backends=["cuda","nv","clang"]):
        super().__init__("depth_model_inferencing")
        self.declare_parameter('tinygrad_backend',rclpy.Parameter.String,"cuda")
        self.declare_parameter('model_file',rclpy.Parameter.String,"model_name")
        self.supported_backends = supported_backends
        backend = self.get_parameter("tinygrad_backend",).get_parameter_value()
        if backend.lower() in supported_backends:
            if backend.lower != "clang":
                os.environ[backend.upper()] = "1"
        self.imageSubscriber = rclpy.create_subscriber(Image, "video_stream_1",self.get_depth,10)
        self.imageSubscriber = rclpy.create_subscriber(CameraInfo, "video_stream_1/cameraInfo",self.get_camera_intrensics,10)
        self.depthPublisher = rclpy.create_publisher(PointCloud2, "depthToCloud")
        self.bridge = CvBridge() # convert image to numpy array to run inference
        self.model = safe_load(f"{package_share_directory}/models/{model_name}")
        self.model.eval()
        self.fields = [PointField(name=n,offset=i*self.datatype, datatype=PointField.FLOAT32, count=1) for i,n in enumerate('xyz')]
        self.camera_info_msg = None


    @Tinyjit
    def model_inference(x:Tensor)->Tensor:
        x = self.model(x)
        return x

    def get_camera_intrinsics(msg:CameraInfo)-> None:
        self.camera_info_msg = msg

    def get_depth(msg:Image)->None:
        msg_header = Header(frame_id=msg.header.frame_id)
        height:int
        width:int
        if self.camera_info_msg == None:
            '''short cut if a camera intrinsics method has been recieved'''
            return
        try:
            img_np:np.ndarray = self.bridge.imgmsg_to_cv(msg, desired_encoding="passthrough").astype(np.float32)
        except CvBridgeError as e:
            '''short cut as image cannot be proccessed'''
            self.get_logger().error(f"{e} cv bridge Error handling the msg is not processed", throttle_duration=1)
            return

        result: Tensor = model_inference(Tensor(img_np.reshape(-1,3,))
        _, _, height, width = result.shape
        points = convertToPointCloud(result.reshape(height,width))
        if self.unstructured == True:
            pcl_msg = PointCloud2(header=msg_header,
                                  width=1,
                                  height=height*width,
                                  point_step= self.datatype * 3,
                                  row_step= self.datatype * 3 * width * height,
                                  is_bigendian=False,
                                  is_dense=False,
                                  fields=self.fields,
                                  data=points)
        self.depth_publisher.publish(pcl_msg)

    '''does not work with rgb pointclouds'''
    def convertToPointCloud(arr:nd_array)->np.ndarray:
        #TODO: convert to a .pyx module so its faster will need to be benchmarked as serialization even despite SIMD speedups
        height:int
        width:int
        cx:float = self.camera_info_msg.k[2]
        cy:float = self.camera_info_msg.k[5]
        fx:float = self.camera_info_msg.k[0]
        fy:float = self.camera_info_msg.k[4]
        depth_scale:float = 1.0
        _,_, height, width = arr.shape
        arr.reshape(width*height)
        def indexConversion(in, width=width,height=height):
            return (in%width,in//width)
        #TODO: change to a list comprehension maybe or do it in 
        pcl_list = []
        for d in range(arr.shape()):
            u,v = indexConversion(d)
            z = arr[d]/depth_scale
            x = ((u - cx) * z)/fx
            y = ((v - cy) * z)/fy
            pcl_list.append([x y z])
        return np.array(pcl_list,dtype=np.float32)

if __name__ == "__main__":
    model_name = "mobilenetv4_28.safetensors"
    node = DepthInferencingNode(model_name)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
