import sys
sys.path.append('../helpers/')
import cv2
import numpy as np
from torch import tensor
from torch import uint8 as torch_uint8
from src.helpers import factory
from torchvision.io import encode_jpeg

try:
    from torchvision.io import decode_jpeg
except ImportError:
    from torchvision.io import decode_image


class ScaleToNetInputSize:
    def __init__(self, sample_shape, aug_params):
        self.sample_shape = sample_shape
        self.interpolation = aug_params.Resolution.interpolation
        print("INFO -- Upscaler uses Interpolation Method: {}".format(self.interpolation))

    def __call__(self,x):
        # skip up-scaling if sample was not downscaled in the first place
        if x.shape[0] == self.sample_shape[0] and x.shape[1] == self.sample_shape[1]:
            return x
        method = factory.create(f"cv2.{np.random.choice(self.interpolation)}")

        return cv2.resize(x, (self.sample_shape[1], self.sample_shape[0]), interpolation=method)


class Resolution:

    def __init__(self, aug_params, sample_shape):
        self.sample_shape = sample_shape
        self.probability = aug_params.Resolution.probability
        self.interpolation = aug_params.Resolution.interpolation
        self.min_res = aug_params.Resolution.min_res
        self.max_res = aug_params.Resolution.max_res
        self.keep_aspect_ratio = aug_params.Resolution.keep_aspect_ratio
        print("INFO -- Resolution uses Interpolation Method: {}".format(self.interpolation))

    def __call__(self, x):
        if np.random.uniform(0, 1) < self.probability:
            method = factory.create(f"cv2.{np.random.choice(self.interpolation)}")
            new_width = np.random.randint(self.min_res, self.max_res + 1)
            if self.keep_aspect_ratio:
                aspr = self.sample_shape[1] / self.sample_shape[0]
                new_height = int(np.round(new_width / aspr))
            rescaled = cv2.resize(x, (int(new_width),int(new_height)), interpolation=method)
            return rescaled
        return x


class PerImageStandardizationTransform:
    """implements same procedure as tf.image.per_image_standardization:
    Linearly scales each image in image to have mean 0 and variance 1."""

    def __call__(self, x):
        image = x.astype(np.float32)
        mean = image.mean()
        stddev = np.std(image)
        adjusted_stddev = max(stddev, 1.0 / np.sqrt(image.size))

        image -= mean
        image /= adjusted_stddev
        return image


class Normalize:

    def __init__(self):
        pass

    def __call__(self, x):
        div = x.max() - x.min()
        x = x - x.min()
        x = x / div
        return x


# class JpegCompression:
#     """Jpeg compression augmentation
#
#     Args:
#         x: Image to jpeg compress
#
#     Returns:
#         Augmented image
#     """
#
#     def __init__(self, aug_params, inf_injection=None):
#         self.probability = aug_params.Jpeg.probability
#         self.QF_min = aug_params.Jpeg.QF_min
#         self.QF_max = aug_params.Jpeg.QF_max
#         self.inf_injection = inf_injection
#         if self.inf_injection is not None:
#             print("INFO -- Initializing JPEG Augmentation with inf_injection class {}".format(self.inf_injection))
#
#     def __call__(self, x):
#         qf = None
#         if np.random.uniform(0, 1) < self.probability:
#             qf = np.random.randint(self.QF_min, self.QF_max + 1)
#
#             # torch wants img in CHW order
#             x = np.expand_dims(x, axis=0)
#             x_enc = encode_jpeg(tensor(x * 255.0, dtype=torch_uint8), qf)
#             try:
#                 x_dec = decode_jpeg(x_enc)  # returns uint8 in CHW order
#             except NameError:
#                 x_dec = decode_image(x_enc)
#
#             x = np.squeeze(x_dec)
#             x = x.numpy() / 255.0
#             x = np.clip(x, 0, 1)
#
#         # try binning
#         #qf_emb = (qf-1) // 10  # 10 classes
#         qf_emb = qf -1
#
#         qf_emb = self.inf_injection if self.inf_injection is not None else qf_emb
#
#         return qf_emb, x  # for better handling scale jpeg quality between [0, max_classes-1]


class JpegCompression:

    def __init__(self, aug_params, inf_injection=None, bin_map=None, linear_map_factor=None):
        self.probability = aug_params.Jpeg.probability
        self.QF_min = aug_params.Jpeg.QF_min
        self.QF_max = aug_params.Jpeg.QF_max
        self.n = Normalize()
        self.inf_injection = inf_injection
        assert not (bin_map is not None and linear_map_factor is not None), "bin_map or linear_map_factor has to be " \
                                                                            "None! Decide for one JPEG quality class" \
                                                                            " computation!"
        self.bin_map = bin_map
        self.linear_map_factor = linear_map_factor
        if self.linear_map_factor is not None:
            print("INFO -- Initializing JPEG Augmentation with linear class scaling: (qf-1) // {}".format(
                self.linear_map_factor))
        if self.bin_map is not None:
            print("INFO -- Initializing JPEG Augmentation with custom jpeg class binning map: {}".format(self.bin_map))
        if self.inf_injection is not None:
            print("INFO -- Initializing JPEG Augmentation with inf_injection class {}".format(self.inf_injection))
        if self.linear_map_factor is None and self.bin_map is None:
            print("INFO -- Initializing JPEG Augmentation with unscaled classes: qf - 1")

    def __call__(self, x):
        qf = np.random.randint(self.QF_min, self.QF_max + 1)
        qf_emb = qf - 1
        if np.random.uniform(0, 1) < self.probability:
            # torch wants img in CHW order
            x = np.expand_dims(x, axis=0)
            x_enc = encode_jpeg(tensor(x * 255.0, dtype=torch_uint8), qf)
            try:
                x_dec = decode_jpeg(x_enc)  # returns uint8 in CHW order
            except NameError:
                x_dec = decode_image(x_enc)

            x = np.squeeze(x_dec.numpy())
            x = self.n(x)

            if self.inf_injection is not None:
                qf_emb = self.inf_injection
            else:
                # bin_map or linear mapping (with or without scaling)
                if self.bin_map is not None:
                    qf_emb = self.bin_map[qf - 1]
                elif self.linear_map_factor is not None:
                    qf_emb = (qf - 1) // self.linear_map_factor  # qf \in [0, num_degradation_classes-1]
                else:
                    qf_emb = (qf - 1)  # no scaling
        return qf_emb, x  # for better handling scale jpeg quality between [0, max_classes-1]

