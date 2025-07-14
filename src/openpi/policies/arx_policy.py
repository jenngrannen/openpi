import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_arx_example() -> dict:
    """Creates a random input example for the ARX5 policy."""
    cam_names = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
    visual_obs = {}
    for camera_name in cam_names:
        # Get the latest frame from the queue
        visual_obs[camera_name] = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
        visual_obs[f"{camera_name}_frame_index"] = 0
        visual_obs[f"{camera_name}_timestamp"] = 0

    obs = {}
    obs["state"] = np.random.rand(14) # np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])
    obs["images"] = visual_obs
    return obs


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H,W,C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class ArxInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # Process ARX5 raw observation format
        visual_obs = data["images"]
        state_obs = data["state"]
        
        # Extract joint positions and gripper positions from all follower arms
        state = state_obs
        state = np.array(state)
        state = transforms.pad_to_dim(state, self.action_dim)

        # Select external camera for policy (use left_camera as default)
        if "cam_high" in visual_obs:
            base_left_image = _parse_image(visual_obs["cam_high"])
        if "cam_low" in visual_obs:
            base_right_image = _parse_image(visual_obs["cam_low"])
        
        wrist_left_image = None
        wrist_right_image = None
        
        if "cam_left_wrist" in visual_obs:
            wrist_left_image = _parse_image(visual_obs["cam_left_wrist"])
        else:
            wrist_left_image = np.zeros_like(base_left_image)

        if "cam_right_wrist" in visual_obs:
            wrist_right_image = _parse_image(visual_obs["cam_right_wrist"])
        else:
            wrist_right_image = np.zeros_like(base_left_image)
        
        match self.model_type:
            case _model.ModelType.PI0:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                # Default to both left base image and both wrist images
                images = (base_left_image, wrist_left_image, wrist_right_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # Default to both base images and left wrist image
                images = (base_left_image, base_right_image, wrist_left_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.array(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ArxOutputs(transforms.DataTransformFn): # TODO(jenn): Implement this
    """Transform outputs for ARX5 bimanual control."""
    
    def __call__(self, data: dict) -> dict:
        # For ARX5, we need to handle bimanual actions
        # The model outputs 8-dimensional actions, but we need to split them for left/right arms
        actions = np.asarray(data["actions"])
        
        # For now, we'll use the same action for both arms
        # In the future, this could be extended to handle separate left/right actions
        # actions shape: (horizon, 8) where 8 = [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, gripper, unused]
        
        return {"actions": actions} 
    
def main():
    # Create a random example input
    example_input = make_arx_example()
    
    # Initialize the input transform
    input_transform = ArxInputs(action_dim=8)

    # Apply the input transform
    transformed_input = input_transform(example_input)
    
    # Print the transformed input
    print(transformed_input)

if __name__ == "__main__":
    main()