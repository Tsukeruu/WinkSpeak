from utils.Classes import Initialize

class Main(Initialize):
    def __init__(self) -> None:
        super().__init__()
        super().import_check()

app = Main()
