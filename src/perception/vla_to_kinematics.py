import numpy as np

class VLAtoKinematicsBridge:
    def __init__(self, camera_matrix=None, hand_eye_matrix=None, mount_type="eye_to_hand"):
        """
        Operational bridge converting OpenVLA action tokens into actionable Robot Base Frame coordinates.
        
        Args:
            camera_matrix (np.ndarray): 3x3 Intrinsic calibration matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            hand_eye_matrix (np.ndarray): 4x4 Homogeneous transformation matrix mapping Camera -> Robot Frame
            mount_type (str): "eye_to_hand" (static camera) or "eye_in_hand" (wrist-mounted camera)
        """
        self.mount_type = mount_type
        
        # 1. Camera Intrinsics (Placeholders fallback to standard RealSense D435 values if None)
        if camera_matrix is not None:
            self.K = camera_matrix
        else:
            # Typical factory calibration values for 1920x1080 resolution
            self.K = np.array([
                [1380.0,    0.0, 960.0],
                [   0.0, 1380.0, 540.0],
                [   0.0,    0.0,   1.0]
            ])
            
        self.fx, self.fy = self.K[0, 0], self.K[1, 1]
        self.cx, self.cy = self.K[0, 2], self.K[1, 2]
        
        # 2. Hand-Eye Calibration Matrix (4x4 Homogeneous Transform: T_camera_to_robot)
        # Translates a point from Camera Frame to Robot/Tool Frame
        if hand_eye_matrix is not None:
            self.T_c2r = hand_eye_matrix
        else:
            # Fallback Mock Calibration: Camera placed 500mm above and 200mm behind the robot base,
            # angled slightly downward (30 degrees rotation around X-axis)
            theta = np.radians(30.0)
            R_x = np.array([
                [1.0,           0.0,            0.0],
                [0.0, np.cos(theta), -np.sin(theta)],
                [0.0, np.sin(theta),  np.cos(theta)]
            ])
            t = np.array([0.0, -200.0, 500.0]) # Translation vector in mm
            
            self.T_c2r = np.eye(4)
            self.T_c2r[:3, :3] = R_x
            self.T_c2r[:3, 3] = t

        # Physical workspaces limits (X, Y, Z in robot base frame) to prevent self-collisions
        self.safety_bounds = {
            "x": (-600.0, 600.0),
            "y": (-600.0, 600.0),
            "z": (50.0, 500.0)  # Stop gripper 50mm above desk surface
        }

    def decode_tokens(self, tokens):
        """Extracts spatial bin indices and gripper action from standard 7-token OpenVLA arrays."""
        if len(tokens) != 7:
            raise ValueError(f"Expected 7 action tokens, received {len(tokens)}")
        return tokens[0], tokens[1], tokens[2], tokens[6]

    def compute_3d_coordinates(self, x_bin, y_bin, z_bin, image_w=1920, image_h=1080):
        """Converts discrete 0-255 action tokens back into metric Camera coordinates."""
        # Map 0-255 bins back to raw pixel coordinates (u, v)
        u_pixel = (x_bin / 255.0) * image_w
        v_pixel = (y_bin / 255.0) * image_h
        
        # De-quantize Z depth bin (Assume workspace tracking depth ranges between 300mm to 1000mm)
        min_z, max_z = 300.0, 1000.0
        z_camera = min_z + ((z_bin / 255.0) * (max_z - min_z))
        
        # Backward Projective Pinhole Camera Math (Image Plane -> Camera Frame)
        x_camera = (u_pixel - self.cx) * z_camera / self.fx
        y_camera = (v_pixel - self.cy) * z_camera / self.fy
        
        return np.array([x_camera, y_camera, z_camera]) 

    def transform_to_robot_frame(self, p_camera_homog, T_tool_to_base=None):
        """
        Transforms coordinates from camera space to the robot base coordinate frame.
        
        For static cameras (eye_to_hand): Uses the fixed T_c2r calibration matrix.
        For wrist cameras (eye_in_hand): Chains camera->tool and tool->base transforms.
        """
        if self.mount_type == "eye_to_hand":
            # Direct matrix dot product: P_robot = T_camera_to_robot * P_camera
            p_robot = np.dot(self.T_c2r, p_camera_homog)
            return p_robot[:3]
            
        elif self.mount_type == "eye_in_hand":
            if T_tool_to_base is None:
                raise ValueError("Eye-in-Hand mode requires the current joint-derived Tool-to-Base matrix!")
            # Chain transform: P_robot = T_tool_to_base * T_camera_to_tool * P_camera
            p_tool = np.dot(self.T_c2r, p_camera_homog)
            p_robot = np.dot(T_tool_to_base, p_tool)
            return p_robot[:3]

    def apply_safety_guardrails(self, target_coords):
        """Clamps targets to physical safety envelopes to protect physical robotic hardware."""
        clamped_coords = [
            np.clip(target_coords[0], self.safety_bounds["x"][0], self.safety_bounds["x"][1]),
            np.clip(target_coords[1], self.safety_bounds["y"][0], self.safety_bounds["y"][1]),
            np.clip(target_coords[2], self.safety_bounds["z"][0], self.safety_bounds["z"][1])
        ]
        was_clamped = not np.allclose(target_coords, clamped_coords)
        return np.array(clamped_coords), was_clamped

    def execute_bridge(self, vla_tokens, T_tool_to_base=None):
        """The primary operational interface to compile raw actions into physical robot commands."""
        
        # Format normalization
        tokens = np.asarray(vla_tokens)
        
        x_b, y_b, z_b, gripper_bin = self.decode_tokens(tokens)
        
        # Step 1: Project to Camera coordinate Frame
        p_cam = self.compute_3d_coordinates(x_b, y_b, z_b)
        
        # Step 2: Transform to Robotic base frame
        # Convert to homogeneous coordinates by appending 1.0
        p_camera_homog = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])
        p_robot_raw = self.transform_to_robot_frame(p_camera_homog, T_tool_to_base)
        
        # Step 3: Enforce hardware workspace envelopes
        p_robot_safe, violated = self.apply_safety_guardrails(p_robot_raw)
        
        gripper_state = "CLOSED" if gripper_bin > 127 else "OPEN"
        
        # Clean terminal logging
        print(f"[KINEMATICS] Camera Space Target: X={p_cam[0]:.2f}mm, Y={p_cam[1]:.2f}mm, Z={p_cam[2]:.2f}mm")
        print(f"[KINEMATICS] Robot Base Target:   X={p_robot_safe[0]:.2f}mm, Y={p_robot_safe[1]:.2f}mm, Z={p_robot_safe[2]:.2f}mm")
        
        if violated:
            print(f"[KINEMATICS] WARNING: Coordinates clamped due to hardware bounds (Raw: X={p_robot_raw[0]:.0f}, Y={p_robot_raw[1]:.0f}, Z={p_robot_raw[2]:.0f})")
            
        print(f"[KINEMATICS] Gripper State:       {gripper_state}")
        
        return {
            "target_position": p_robot_safe,
            "gripper_state": gripper_state,
            "out_of_bounds": violated
        }

if __name__ == "__main__":
    # Example action tokens produced by OpenVLA model
    mock_prediction = [100, 128, 200, 128, 128, 128, 255]
    
    # Initialize the bridge
    robot_bridge = VLAtoKinematicsBridge(mount_type="eye_to_hand")
    
    # Execute transform
    robot_bridge.execute_bridge(mock_prediction)