class FrameNormalizer(object):
    def __init__(self):
        pass

    def __call__(self, image):
        image = (image / 255.0) * 2.0 - 1.0

        return image


class FrameDeNormalizer(object):
    def __init__(self):
        pass

    def __call__(self, image):
        image = ((image + 1) / 2) * 255.0

        return image
