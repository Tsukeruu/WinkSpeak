from dataclasses import dataclass
from typing import List, Union, Any, Tuple, Dict
from importlib import import_module

class Initialize:
    def __init__(self) -> None:
        self.libraries_list: List[str] = ["cv2","mediapipe","time","numpy","scipy","pathlib","os"]
        self.stored_libraries: Dict[str,Any] = {}
        self.morse_code_dict: Dict[str, str] = {
            '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
            '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
            '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
            '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
            '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
            '--..': 'Z', '-----': '0', '.----': '1', '..---': '2', '...--': '3',
            '....-': '4', '.....': '5', '-....': '6', '--...': '7', '---..': '8',
            '----.': '9'
        }
        self.EAR_THRESHOLD: float = 0.21 # < Means eye close, otherwise >=, means open eye
        self.DOT_DASH_THRESHOLD: float = 0.4
        self.CHARACTER_PAUSE_TIME: float = 1.2
        self.SPACE_PAUSE_TIME: float = 2.5
        
        self.CURRENT_MORSE_SEQUENCE: str = ""
        self.DECODED_TEXT: str = ""

        #FORMAT (MEDIAPIPE) = [left_corner,top_left,top_right,right_corner,bottom_right,bottom_left]
        self.LEFT_EYE: List[int] = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE: List[int] = [362, 385, 387, 263, 373, 380]

    def import_check(self) -> None:
        try:
            for library in self.libraries_list:
                if library == "scipy":
                    self.stored_libraries[library] = import_module(f"{library}.spatial.distance")
                else:
                    self.stored_libraries[library] = import_module(library)

        except ImportError as IE:
            print(f"[!] IMPORT ERROR: {IE}")
            return

        print(f"[+] IMPORTS AVAIALBLE: {self.libraries_list}")


class Main_methods(Initialize):
    def calculate_EAR(self, eye_landmarks: List[int]) -> float: 
        #VERTICAL DISTANCE
        v1: float = self.stored_libraries["scipy"].euclidean(eye_landmarks[1], eye_landmarks[5])
        v2: float = self.stored_libraries["scipy"].euclidean(eye_landmarks[2], eye_landmarks[4])
        
        #HORIZONTAL DISTANCE
        h1: float = self.stored_libraries["scipy"].euclidean(eye_landmarks[0], eye_landmarks[3])

        self.EAR_VALUE: float = (v1 + v2) / (2.0 * h1)
        return self.EAR_VALUE
    
    def initiate_webcam(self, webcam_index: int = 0) -> None:
        #In videocapture, it starts video streaming from an index (webcams, 0 being primary)
        self.capture: self.stored_libraries["cv2"].VideoCapture = self.stored_libraries["cv2"].VideoCapture(webcam_index)

        self.eye_closed: bool = False
        self.blink_start_time: int = 0
        self.eyes_open_start_time: float = self.stored_libraries["time"].time()

        self.base_options = self.stored_libraries["mediapipe"].tasks.BaseOptions

        self.face_landmarker = self.stored_libraries["mediapipe"].tasks.vision.FaceLandmarker
        self.face_landmarker_options = self.stored_libraries["mediapipe"].tasks.vision.FaceLandmarkerOptions

        self.visionrunningmode = self.stored_libraries["mediapipe"].tasks.vision.RunningMode
        
        self.task_path: str = self.stored_libraries["pathlib"].Path(__file__).parent / "face_landmarker.task" 

        self.final_options = self.face_landmarker_options(
                base_options=self.base_options(model_asset_path=self.task_path),
                running_mode=self.visionrunningmode.IMAGE,
                num_faces=1
            )
       
    def main_webcam(self) -> None:
       pass 
