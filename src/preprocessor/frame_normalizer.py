class FrameNormalizer(object):
    def __init__(self):
        pass

    def __call__(self, image):
        image = ((image / 255.0) - 0.5) * 2

        return image


class FrameDeNormalizer(object):
    def __init__(self):
        pass

    def __call__(self, image):
        image = ((image / 2) + 0.5) * 255.0

        return image
