from model import Orchestrator
from model import Converter
c = Converter("./test_folder", "jpg")
c.convert_images()
o = Orchestrator("./test_folder/converted-images")
o.run()