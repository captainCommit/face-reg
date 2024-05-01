from models.Converter import Converter
from models.Orchestrator import Orchestrator
import utils.constants

c = Converter("./test_folder")
c.convert_images()
o = Orchestrator("./test_folder/converted-images")
o.run()