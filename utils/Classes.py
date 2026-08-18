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
        self.EAR_THRESHOLD: float = 0.13 # < Means eye close, otherwise >=, means open eye, the good zone is usually < 0.13/0.14
        self.DOT_DASH_THRESHOLD: float = 0.4
        self.CHARACTER_PAUSE_TIME: float = 1.2
        self.SPACE_PAUSE_TIME: float = 2.5
        
        self.CURRENT_MORSE_SEQUENCE: str = ""
        self.DECODED_TEXT: str = ""

        #FORMAT (MEDIAPIPE) = [left_corner,top_left,top_right,right_corner,bottom_right,bottom_left]
        #These landmarks are crucial for our later calculation in 'landmarks' they represenet the specific points, a needle, in a haystack
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
        """
        CALCULATE_EAR 
        - A simple method that calculates euclidean distance from eye points in the eye landmarks, returns an average EAR
        - An average EAR is an indicator of how open your eye is 
        - FORMULA:
          - (EUCLIDEAN DISTANCE OF V1 + EUCLIDEAN DISTANCE OF V2) / 2 (to find the average of the vertical distances)
          - then divided by the horizontal euclidean distance!
        """

        #VERTICAL DISTANCE
        v1: float = self.stored_libraries["scipy"].euclidean(eye_landmarks[1], eye_landmarks[5])
        v2: float = self.stored_libraries["scipy"].euclidean(eye_landmarks[2], eye_landmarks[4])
        
        #HORIZONTAL DISTANCE
        h1: float = self.stored_libraries["scipy"].euclidean(eye_landmarks[0], eye_landmarks[3])

        self.EAR_VALUE: float = ((v1 + v2) / 2) / h1
        return self.EAR_VALUE

    def calculate_YAW(self, landmarks: List[int]) -> float:
        """
        CALCULATE_YAW
        - A simple method that returns whether your head is turned or not based on distance from either cheek with nose
        - Creates a ratio, yaw is calculated simply because left and right distort cheek distance and is used to determine left or right
        """

        self.left_cheek: float = landmarks[234].x
        self.right_cheek: float = landmarks[454].x
        self.nose: float = landmarks[1].x
        
        self.face_width: float = abs(self.left_cheek - self.right_cheek)
        
        self.cheek_length: float = abs(self.nose - self.left_cheek) / self.face_width

        return self.cheek_length
    
    def initiate_webcam(self, webcam_index: int = 0) -> None:
        #In videocapture, it starts video streaming from an index (webcams, 0 being primary)
        self.capture: self.stored_libraries["cv2"].VideoCapture = self.stored_libraries["cv2"].VideoCapture(webcam_index)

        self.eye_closed: bool = False
        self.blink_start_time: int = 0
        self.eyes_open_start_time: float = self.stored_libraries["time"].time()
            
        self.stored_libraries["cv2"].namedWindow("WINKSPEAK V 0.0.1", self.stored_libraries["cv2"].WINDOW_NORMAL)
        self.stored_libraries["cv2"].resizeWindow("WINKSPEAK V 0.0.1", 1280, 720)
        self.base_options = self.stored_libraries["mediapipe"].tasks.BaseOptions

        self.face_landmarker = self.stored_libraries["mediapipe"].tasks.vision.FaceLandmarker
        self.face_landmarker_options = self.stored_libraries["mediapipe"].tasks.vision.FaceLandmarkerOptions

        self.visionrunningmode = self.stored_libraries["mediapipe"].tasks.vision.RunningMode
        
        self.task_path: str = self.stored_libraries["pathlib"].Path(__file__).parent / "face_landmarker.task" 

        self.final_options = self.face_landmarker_options(
                base_options=self.base_options(model_asset_path=str(self.task_path)),
                running_mode=self.visionrunningmode.IMAGE,
                num_faces=1
            )
       
    def main_webcam(self) -> None:
        with self.face_landmarker.create_from_options(self.final_options) as landmarker:
            while self.capture.isOpened():
                s, f = self.capture.read() #gives us success, frame as return values, frame being the frame of webcam, s indicating success
                if not s: 
                    break

                f = self.stored_libraries["cv2"].flip(f,1) #re adjusts the frame to mirror differently (as in looking at the mirror)
                h, w, _ = f.shape #returns height and width from the frame

                #Mediapipe expects coloros in rgb, cv2 returns it in bgr, the line below converts the frame to rgb format
                rgb_frame = self.stored_libraries["cv2"].cvtColor(f, self.stored_libraries["cv2"].COLOR_BGR2RGB)
                mp_image = self.stored_libraries["mediapipe"].Image(image_format=self.stored_libraries["mediapipe"].ImageFormat.SRGB, data=rgb_frame)
                result = landmarker.detect(mp_image)
                
                if result.face_landmarks:
                    self.landmarks: List[int] = result.face_landmarks[0]
                    
                    #This uses landmarks' normalized values which is 0.0-1.0, from mediapipe and turns them to pixel by multiplying it by width and height respectively so we can use it in cv2
                    left_eye_points: List[List[int]] = self.stored_libraries["numpy"].array([(int(self.landmarks[i].x * w), int(self.landmarks[i].y * h)) for i in self.LEFT_EYE])
                    right_eye_points: List[List[int]] = self.stored_libraries["numpy"].array([(int(self.landmarks[i].x * w), int(self.landmarks[i].y * h)) for i in self.RIGHT_EYE])
                    
                    self.avg_EAR: float = (self.calculate_EAR(left_eye_points) + self.calculate_EAR(right_eye_points)) / 2.0
                    self.current_time: float = self.stored_libraries["time"].time()
                    
                    if self.calculate_YAW(self.landmarks) >= 0.07 and self.calculate_YAW(self.landmarks) <= 0.8:
                        if self.avg_EAR < self.EAR_THRESHOLD:
                            if not self.eye_closed:
                                self.eye_closed: bool = True
                                self.blink_start_time: float = self.current_time
                        else:
                            if self.eye_closed:
                                self.eye_closed: bool = False
                                self.blink_duration: float = self.current_time - self.blink_start_time
                                self.eyes_open_start_time: float = self.current_time
                                if self.blink_duration < self.DOT_DASH_THRESHOLD:
                                    self.CURRENT_MORSE_SEQUENCE += "."
                                else:
                                    self.CURRENT_MORSE_SEQUENCE += "-"
                            else:
                                self.idle_duration: float = self.current_time - self.eyes_open_start_time 
                                if self.CURRENT_MORSE_SEQUENCE and (self.idle_duration > self.CHARACTER_PAUSE_TIME):
                                    char: str = self.morse_code_dict.get(self.CURRENT_MORSE_SEQUENCE, "?")
                                    self.DECODED_TEXT += char
                                    self.CURRENT_MORSE_SEQUENCE: str = ""
                                elif self.idle_duration > self.SPACE_PAUSE_TIME and self.DECODED_TEXT and not self.DECODED_TEXT.endswith(" "):
                                    self.DECODED_TEXT += " "
                    else:
                        print("[!] TOO TILTED!")

                    self.status_text: str = "CLOSED" if self.eye_closed else "OPEN"
                    self.status_color: Any = (0,0,255) if self.eye_closed else (0, 255, 0)

                    self.stored_libraries["cv2"].polylines(f, [left_eye_points], True, (0, 255, 0), 1)
                    self.stored_libraries["cv2"].polylines(f, [right_eye_points], True, (0, 255, 0), 1)

                    self.stored_libraries["cv2"].putText(f, f"EAR: {round(self.avg_EAR, 1)} STATUS: {self.status_text}", (30, 40), self.stored_libraries["cv2"].FONT_HERSHEY_SIMPLEX, 0.7, self.status_color, 2)

                    self.stored_libraries["cv2"].putText(f, f"MORSE SEQUENCE: {self.CURRENT_MORSE_SEQUENCE}", (30, 80), self.stored_libraries["cv2"].FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                        
                    self.stored_libraries["cv2"].putText(f, f"TEXT SEQUENCE: {self.DECODED_TEXT}", (30, 100), self.stored_libraries["cv2"].FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                    self.stored_libraries["cv2"].putText(f, f"Q TO QUIT, C TO CLEAR", (30, 120), self.stored_libraries["cv2"].FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                       
                    self.stored_libraries["cv2"].imshow("WINKSPEAK V 0.0.1", f)

                    key: int = self.stored_libraries["cv2"].waitKey(1) & 0xFF
                    #Ord returns the number associated with each key / letter press, waitKey awaits keypresses
                    if key == ord('q'):
                        break
                    elif key == ord('c'):
                        self.DECODED_TEXT: str = ""
                        self.CURRENT_MORSE_SEQUENCE: str = ""


