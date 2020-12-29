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

        # first_img_id = process_data.get_first_img_id()
        # frames_sum = process_data.get_frames_sum()
        #
        # for i in range(first_img_id, first_img_id + frames_sum):
        #     process_data.process_the_data(data)
        #     tfl_manager.run(data)





