from part_4.data import Data
from part_4.process_data import ProcessData
from part_4.tfl_manager import TFLManager


class Controller:
    def process(self, pls_path):
        data = Data()
        process_data = ProcessData(pls_path)
        tfl_manager = TFLManager()

        for curr_img_data in process_data.process_the_data(data):
            tfl_manager.run(curr_img_data)
