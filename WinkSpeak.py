from utils.Classes import Initialize, Main_methods
from typing import List, Dict, Any, Union, Tuple

class Main(Main_methods):
    def initial_checks(self) -> None:
        super().import_check()
    
    def run(self) -> None:
        self.initial_checks()
        self.initiate_webcam()
        self.main_webcam()

app = Main()
app.run()
