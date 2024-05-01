from model import Orchestrator
from model import Converter
import constants
c = Converter("./test_folder")
c.convert_images()
o = Orchestrator("./test_folder/converted-images")
o.run()