import numpy as np

import part_4.utils as utils
from part_3.SFM_standAlone import FrameContainer


class ProcessData:
    def __init__(self, pls_path):
        self.pls_path = pls_path

    def get_first_img_id(self):
        pls_lines = utils.read_txt_file(self.pls_path)
        return int(pls_lines[1][31:33])

    def get_frames_sum(self):
        pls_lines = utils.read_txt_file(self.pls_path)
        return len(pls_lines) - 1

    def process_the_data(self, frame_data_container):
        pls_lines = utils.read_txt_file(self.pls_path)
        pkl_path = pls_lines[0]

        data = utils.read_pkl_file(pkl_path)
        frame_data_container.init_pp_and_focal(data['principle_point'], data['flx'])

        for img_path in pls_lines[1:]:
            print(img_path)
            curr_frame_id = int(img_path[-18:-16])
            curr_frame_container = FrameContainer(img_path)
            curr_frame_container.traffic_light = np.array(data['points_' + str(curr_frame_id)][0])

            EM = np.eye(4)

            # if frame_data_container.prev_frame_id is None:
            #     prev_frame_id = curr_frame_id - 1
            #
            # else:
            #     prev_frame_id = frame_data_container.prev_frame_id

            prev_frame_id = curr_frame_id - 1

            for i in range(prev_frame_id, curr_frame_id):
                EM = np.dot(data['egomotion_' + str(i) + '-' + str(i + 1)], EM)

            curr_frame_container.EM = EM

            frame_data_container.update(curr_frame_container, curr_frame_id)

            yield frame_data_container
