class Data:
    def __init__(self):
        self.prev = None
        self.curr = None
        self.principle_point = None
        self.focal = None
        self.curr_frame_id = None
        self.prev_frame_id = None

    def update(self, curr, curr_id):
        self.prev = self.curr
        self.prev_frame_id = self.curr_frame_id
        self.curr = curr
        self.curr_frame_id = curr_id

    def init_pp_and_focal(self, pp, focal):
        self.principle_point = pp
        self.focal = focal
