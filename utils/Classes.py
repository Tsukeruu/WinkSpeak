from dataclasses import dataclass
from typing import List, Union, Any, Tuple, Dict
from importlib import import_module

class Initialize:
    def __init__(self) -> None:
        self.libraries_list: List[str] = ["cv2","mediapipe","time","numpy"]
        self.stored_libraries: Dict[str,Any] = {}

    def import_check(self) -> None:
        try:
            for library in self.libraries_list:
                self.stored_libraries[library] = import_module(library)

        except ImportError as IE:
            print(f"[!] IMPORT ERROR: {IE}")
            return

        print(f"[+] IMPORTS AVAIALBLE: {self.libraries_list}")
