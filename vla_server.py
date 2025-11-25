import argparse
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import cv2
import base64
from typing import List, Optional
from inference import run_inference_to_server, get_policy
import time

# --- Template for VLA Model Integration ---
# You should import your actual VLA model library here
# e.g. from openvla import OpenVLA

class VLAInferenceServer:
    def __init__(self):
        print("Loading VLA Model...")
        # TODO: Initialize your model here
        # self.model = OpenVLA.load("path/to/model")
        self.prev_action = {}
        self.policy = get_policy(model_path="/home/yuxin/models/GR00T-N1.5-3B")

    def predict(self, image: np.ndarray, instruction: str, ee_pos: Optional[List[float]] = None, ee_quat: Optional[List[float]] = None, joint_pos = None) -> np.ndarray:
        print(f"image shape: {image.shape}, instruction: {instruction}, ee_pos: {ee_pos}, ee_quat: {ee_quat}, joint_pos: {joint_pos}")
        # TODO: Replace this with actual model inference
        # return self.model.predict(image, instruction)
        
        # Dummy logic for testing: Move joints in a sine wave pattern based on image brightness
        avg_brightness = np.mean(image) / 255.0
        # action = np.zeros(7) # 7 DOF for Panda
        action = np.array([0, -0.58, 0, -1.68, 0, 1.13, 0.8, 0])
        action[1] = np.sin(avg_brightness * 10)
        print(f"Predicted action based on brightness {avg_brightness}: {action}")
        return action
    
    def predict_2(self, image: np.ndarray, instruction: str, ee_pos: Optional[List[float]] = None, ee_quat: Optional[List[float]] = None, joint_pos = None) -> dict:
        action = run_inference_to_server(image, instruction, ee_pos=np.array(ee_pos), ee_quat=np.array(ee_quat), joint_pos=np.array(joint_pos), prev_action=self.prev_action, policy=self.policy)
        self.prev_action = action
        # print(f"Predicted action from VLA inference server: {action}")
        return action

# ------------------------------------------

app = FastAPI()
model_wrapper = VLAInferenceServer()

class PredictionRequest(BaseModel):
    image_base64: str
    instruction: str
    robot_id: Optional[str] = "robot1"
    ee_pos: Optional[List[float]] = None
    ee_quat: Optional[List[float]] = None
    joint_pos: Optional[List[float]] = None


class PredictionResponse(BaseModel):
    action: List
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    time.sleep(2.0)  # Simulate processing delay
    try:
        # Decode image
        img_bytes = base64.b64decode(request.image_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Run inference
        # action = model_wrapper.predict(image, request.instruction, ee_pos=request.ee_pos, ee_quat=request.ee_quat, joint_pos=request.joint_pos)
        vla_action = model_wrapper.predict_2(image, request.instruction, ee_pos=request.ee_pos, ee_quat=request.ee_quat, joint_pos=request.joint_pos)

        vla_action = vla_action.get("action.arm_joint_positions", np.zeros((16, 9), dtype=np.float32)).tolist()
        # print(f"Final action to return: {vla_action}")
        
        return PredictionResponse(action=vla_action)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    parser = argparse.ArgumentParser(description='VLA Inference Server')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()

    print(f"Starting VLA Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
