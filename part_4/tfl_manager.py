import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from part_4.adapter import Adapter
from part_1.run_attention import find_tfl_lights as find_tfl_lights
from part_3.SFM import calc_TFL_dist as calc_TFL_dist
import part_3.SFM as SFM
import part_4.utils as utils


class TFLManager:
    def visualize(self, prev_container, curr_container, focal, principle_point, curr_frame_id, prev_frame_id):
        norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = \
            SFM.prepare_3D_data(prev_container, curr_container, focal, principle_point)
        norm_rot_pts = SFM.rotate(norm_prev_pts, R)
        rot_pts = SFM.unnormalize(norm_rot_pts, focal, principle_point)
        foe = np.squeeze(SFM.unnormalize(np.array([norm_foe]), focal, principle_point))

        fig, (curr_sec, prev_sec) = plt.subplots(1, 2, figsize=(12, 6))
        prev_sec.set_title('prev(' + str(prev_frame_id) + ')')
        prev_sec.imshow(prev_container.img)
        prev_p = prev_container.traffic_light
        prev_sec.plot(prev_p[:, 0], prev_p[:, 1], 'b+')

        curr_sec.set_title('curr(' + str(curr_frame_id) + ')')
        curr_sec.imshow(curr_container.img)
        curr_p = curr_container.traffic_light
        curr_sec.plot(curr_p[:, 0], curr_p[:, 1], 'b+')

        for i in range(len(curr_p)):
            curr_sec.plot([curr_p[i, 0], foe[0]], [curr_p[i, 1], foe[1]], 'b')
            if curr_container.valid[i]:
                curr_sec.text(curr_p[i, 0], curr_p[i, 1],
                              r'{0:.1f}'.format(curr_container.traffic_lights_3d_location[i, 2]), color='r')
        curr_sec.plot(foe[0], foe[1], 'r+')
        curr_sec.plot(rot_pts[:, 0], rot_pts[:, 1], 'g+')
        plt.show()

    def run(self, data):
        adapter_project_parts = Adapter()
        image = np.array(data.curr.img)

        candidates, auxiliary = adapter_project_parts.adapter_part_1_to_part_2(find_tfl_lights(image))

        cropped_imgs = utils.crop_img_by_indx_list(image, candidates)

        loaded_model = load_model("./part_2/model.h5")
        cropped_imgs_predicts = loaded_model.predict(np.array(cropped_imgs))

        candidates_list = adapter_project_parts.adapter_part_2_to_part_3(cropped_imgs_predicts, candidates)

        data.curr.traffic_light = np.array(candidates_list)

        if data.prev is not None:  # and percents.shape[0] <= len(cropped_imgs):

            data.curr = calc_TFL_dist(data.prev, data.curr, data.focal, data.principle_point)

            # print(type(data.curr.traffic_lights_3d_location[0]))
            # data.curr.traffic_lights_3d_location = \
            #     np.array(
            #         [np.array(c) for c in data.curr.traffic_lights_3d_location if c[0] >= 0 and c[1] >= 0 and c[2] >= 0]
            #     )
            # np.array(filter(lambda c: c[0] >= 0 and c[1] >= 0 and c[2] >= 0, data.curr.traffic_lights_3d_location))

            self.visualize(data.prev, data.curr, data.focal, data.principle_point, data.curr_frame_id,
                           data.prev_frame_id)
