import numpy as np
class House:
    def __init__(self, utm_coords,utm_edges, id):
        self.utm_coords = utm_coords
        self.utm_edges = utm_edges
        self.utm_mean = np.mean(utm_coords, axis=1)
        self.image_ids = list()
        self.image_coords = list()
        self.image_edges = list()

        self.id = id
