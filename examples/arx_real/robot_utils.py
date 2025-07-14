# Ignore lint errors because this file is mostly copied from ACT (https://github.com/tonyzhaozh/act).
# ruff: noqa
from collections import deque
import datetime
import json
import time
import threading
import queue
import cv2
from aloha.msg import RGBGrayscaleImage
from cv_bridge import CvBridge
from interbotix_xs_msgs.msg import JointGroupCommand
from interbotix_xs_msgs.msg import JointSingleCommand
import numpy as np
import rospy
from sensor_msgs.msg import JointState

from examples.arx_real import constants
from examples.arx_real.arx5_config import config

OFFLINE = True

if not OFFLINE:
    import pyrealsense2 as rs
    from arx5_interface import Arx5CartesianController, Arx5JointController, Gain, LogLevel
else:
    from offline_mocks import MockController, MockJointState, MockEefState, MockCamera, MockRs
    rs = MockRs

# TODO: Make these configurable.
# Maximum size of the frames and joint states queues.
FRAME_RATE = 120 # TODO: Set to maximum framerate of the cameras.
JOINT_RATE = 100 # TODO: Set to maximum joint state polling rate.
FRAME_MAX_SIZE = FRAME_RATE * 5
JOINT_MAX_SIZE = JOINT_RATE * 5

CONTROL_FREQ = 50 # Control loop frequency in Hz.
DROID_CONTROL_FREQUENCY = 10  # DROID control frequency in Hz for consistent timing
MAX_SECONDS = 20 # Maximum recording duration in seconds.
MAX_STEPS = MAX_SECONDS * CONTROL_FREQ  # Maximum number of steps to record.

MAX_EPISODES = 100 # Maximum number of episodes to record.

class ImageRecorder:
    def __init__(self, queue_event=None, is_debug=False):
        self.is_debug = is_debug
        self.camera_names = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]

        self.queue_event = queue_event

        self.pipelines = initialize_cameras(config)
        for camera_name, pipeline in self.pipelines.items():
            image = pipeline.wait_for_frames().get_color_frame()
            if image:
                image = np.asanyarray(image.get_data())
                print(f"Initial image from camera {camera_name} has shape: {image.shape}")

        self.camera_threads = []
        if len(self.pipelines) != 0:
            self.frame_queues = {}
            for camera_name, pipeline in self.pipelines.items():
                # Create a queue for each camera's frames
                self.frame_queues[camera_name] = queue.Queue(maxsize=FRAME_MAX_SIZE)
                # Start a thread to capture frames from each camera pipeline
                t_camera = threading.Thread(
                    target=separate_frame_capture_loop,
                    args=(
                        pipeline, 
                        self.stop_recording_event, 
                        self.queue_event, 
                        self.frame_queues[camera_name], 
                        self.track_frame_index
                    )
                )
                t_camera.start()
                self.camera_threads.append(t_camera)

        # time.sleep(0.5)

    def get_images(self):
        visual_obs = {}
        for camera_name, frames_queue in self.frame_queues.items():
        # Get the latest frame from the queue
            frame_index, timestamp, color_image = frames_queue.get()
            visual_obs[camera_name] = cv2.cvtColor(color_image, cv2.COLOR_BGRA2RGB)
            visual_obs[f"{camera_name}_frame_index"] = frame_index
            visual_obs[f"{camera_name}_timestamp"] = timestamp
        return visual_obs # no depth images


class Recorder:
    def __init__(self, robot_name, robot_controller, is_debug=False):
        self.secs = None
        self.nsecs = None
        self.qpos = None
        self.effort = None
        self.arm_command = None
        self.gripper_command = None
        self.is_debug = is_debug

        # Start polling joint states for each follower controller
        self.joint_state_thread = {}
        self.ctrl_stop_event = {}
        self.joint_state_queue = queue.Queue(maxsize=JOINT_MAX_SIZE)
        # Create a stop event for each follower controller's control loop
        ctrl_stop_event = threading.Event()
        t_follower = threading.Thread(
            target=poll_joint_states,
            args=(robot_controller, self.joint_state_queue, ctrl_stop_event, self.queue_event, 0.01)
        )
        t_follower.start()

        if self.is_debug:
            self.joint_timestamps = deque(maxlen=50)
            self.arm_command_timestamps = deque(maxlen=50)
            self.gripper_command_timestamps = deque(maxlen=50)
        time.sleep(0.1)

    def get_qpos(self):
        return self.joint_state_queue.get()[1].position

def poll_joint_states(controller, joint_states_queue, stop_event, queue_event, rate = 0.01):
    """Poll joint states at default 100 Hz and store them in joint_states_queue."""
    while not stop_event.is_set():
        if queue_event.is_set():
            timestamp = controller.get_timestamp()
            if OFFLINE:
                state = controller.get_joint_state()
            else:
                state = controller.get_state()
            joint_states_queue.put((timestamp, state))
        time.sleep(rate)

def initialize_cameras(config):
    pipelines = {}

    if not OFFLINE:
        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            print(dev)
            dev.hardware_reset()

        def init_camera_pipeline(serial_no):
            # Create a RealSense pipeline
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial_no)
            config.enable_stream(rs.stream.color, 848, 480, rs.format.bgra8, 30)
            pipeline.start(config)
            return pipeline

        for camera_dict in config['cameras']:
            camera_name = camera_dict["name"]
            camera_serial = camera_dict["serial"]
            pipeline = init_camera_pipeline(camera_serial)
            pipelines[camera_name] = pipeline
    else:
        # Use mock cameras for offline testing
        print("Using mock cameras for offline testing")
        for camera_dict in config['cameras']:
            camera_name = camera_dict["name"]
            pipeline = MockCamera(camera_name)
            pipelines[camera_name] = pipeline

    return pipelines

def separate_frame_capture_loop(pipeline, stop_event, queue_event, frames_queue, track_frame_index=True):
    """
    Capture frames from a _single_ camera pipeline and add them to the frames_queue.
    """
    frame_index = 0
    while not stop_event.is_set():
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Get timestamp and convert frame to numpy array.
        timestamp = color_frame.get_timestamp()
        color_image = np.asanyarray(color_frame.get_data())

        # If recording is active, add the frame and metadata to the queue.
        if queue_event.is_set():
            frames_queue.put((frame_index, timestamp, color_image))
            if track_frame_index:
                frame_index += 1

def move_arms(bot_list, target_pose_list):
    for bot_id, bot in enumerate(bot_list):
        joint_cmd = target_pose_list[bot_id]
        bot.set_joint_positions(joint_cmd)

