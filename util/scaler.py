class MinMaxScaler():
    def __init__(self, min_value, max_value):
        assert (max_value > min_value)
        self.min_value = min_value
        self.max_value = max_value

    def scale_value(self, val):
        return (val - self.min_value) / (self.max_value - self.min_value)

    def inv_scale_value(self, scaled_val):
        return self.min_value + scaled_val * (self.max_value - self.min_value)