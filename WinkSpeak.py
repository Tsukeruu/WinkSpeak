from utils.Classes import Initialize
from typing import List, Dict, Any, Union, Tuple

class Main(Initialize):
    def initial_checks(self) -> None:
        super().import_check()
    
    def run(self) -> None:
        self.initial_checks()

app = Main()
app.run()
