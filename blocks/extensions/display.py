import numpy
import theano


def get_grid_shape(num_examples):
    """Get the best grid shape given a number of examples.

    Parameters
    ----------
    num_examples : int
        Number of examples considered.


    Returns
    -------
    (height, width) : tuple
        The grid shape.

    """
    height = int(numpy.floor(numpy.sqrt(num_examples)))
    width = int(numpy.ceil(num_examples * 1. / height))

    return (height, width)


def gather_patches(raw_data, grid_shape=None):
    """Gather patches in one image.

    Parameters
    ----------
    raw_data : numpy.array
        The image 4D tensor with axes ('b', 0, 1, 'c').

    Returns
    -------
    image : numpy.array
        The image 3D tensor with axes (0, 1, 'c')

    """
    num_examples = raw_data.shape[0]
    img_h = raw_data.shape[1]
    img_w = raw_data.shape[2]
    img_c = raw_data.shape[3]

    if grid_shape is None:
        grid_shape = get_grid_shape(num_examples)

    expected_examples = grid_shape[0] * grid_shape[1]
    padding_pattern = (((0, expected_examples
                         - num_examples),)
                       + ((0, 0),) * 3)
    padded_data = numpy.pad(
        raw_data,
        pad_width=padding_pattern,
        mode='constant',
        constant_values=0
    )

    image = padded_data.view().reshape((
        grid_shape[1], grid_shape[0] * img_h, img_w, img_c)
    ).transpose(
        (1, 0, 2, 3)
    ).reshape(
        (grid_shape[0] * img_h,
         grid_shape[1] * img_w,
         img_c)
    ).copy()

    image *= 0.5
    image += 0.5
    image *= 255

    return image


def reshape_image(raw_data, image_shape,
                  orig_axes=(0, 1, 'c'), dest_axes=(0, 1, 'c')):
    """Reshape an data tensor into a 4D image tensor.

    Parameters
    ----------
    raw_data : numpy.array
        The data tensor.
    image_shape : tuple
        The original image shape. To match with the original axes.
        See orig_axes.
    orig_axes : tuple, optional
        The original axes of the image. Default is (0, 1, 'c').
    dest_axes : tuple, optional
        The desired axes. Default is (0, 1, 'c').

    Returns
    -------
    image : numpy.array
        The image 4D tensor with axes corresponding to dest_axes.

    """

    if raw_data.ndim != len(image_shape) + 1:
        image_tensor = raw_data.reshape((-1,) + image_shape)
    else:
        image_tensor = raw_data

    rval = image_tensor.transpose(
        (0,) + tuple(orig_axes.index(x) + 1 for x in dest_axes)
    )

    return rval[::-1, ::-1]


class ImageGetter(object):
    """A base class for getting image tensor to display.

    Parameters
    ----------
    image_shape : tuple of int
        The shape of one image.
    axes : tuple, optional
        The original axes of the image.
    shift : float, optional
        The shift necessary so that the images are between -1 and 1.
        Default is 0.
    rescale : float, optional
        The rescale necessary so that the images are between -1 and 1.
        The output images follow the formula:
        images = rescale * (init_images + shift)
        Default is 1.
    grid_shape : tuple, optional
        The shape of the grid that will be shown. Default is inferred.

    """
    default_axes = (0, 1, 'c')

    def __init__(self, image_shape, axes=(0, 1, 'c'),
                 shift=0, rescale=1, grid_shape=None):
        self.image_shape = image_shape
        self.axes = axes
        self.shift = shift
        self.rescale = rescale
        self.grid_shape = grid_shape

    def get_image_data(self):
        """Get raw image data.

        The data will later be reshaped, transposed and rescaled to be
        between -1 and 1.

        """
        raise NotImplementedError(str(type(self)) + 'does not'
                                  'implement get_image.')

    def __call__(self):
        raw_data = self.get_image_data()

        image_tensor = reshape_image(
            raw_data=raw_data,
            image_shape=self.image_shape,
            orig_axes=self.axes,
            dest_axes=self.default_axes
        )

        image = gather_patches(
            numpy.clip(
                self.rescale * (image_tensor + self.shift),
                -1, 1
            ),
            self.grid_shape
        )

        return image


class IndexableImageDatasetDisplay(ImageGetter):
    """Return image of image examples of an indexable dataset.

    Parameters
    ----------
    indexable_dataset : fuel.datasets.IndexableDataset
        The indexable image dataset you show image from.
    num_examples : int, optional
        The number of image examples shown. Default is 100.
    source : string, optional
        The source from which the image is obtained. Default is 'features'.

    """

    def __init__(self, dataset, num_examples=100,
                 source='features', ** kwargs):
        self.dataset = dataset
        self.num_examples = num_examples
        self.source = source

        super(IndexableImageDatasetDisplay, self).__init__(** kwargs)

    def get_image_data(self):
        raw_data = self.dataset.get_data(
            request=list(numpy.random.randint(
                self.dataset.num_examples,
                size=self.num_examples
            ))
        )[self.dataset.sources.index(self.source)]

        return raw_data


class WeightDisplay(ImageGetter):
    """Return image of image examples of an indexable dataset.

    Parameters
    ----------
    weights : theano shared variable
        The Theano shared variable that represent the weights.
    n_weights : int, optional

    """

    def __init__(self, weights, n_weights=None, transpose=None, ** kwargs):
        self.weights = weights
        self.n_weights = n_weights
        self.transpose = transpose

        super(WeightDisplay, self).__init__(** kwargs)

    def get_image_data(self):
        raw_data = self.weights.get_value()

        if self.transpose is not None:
            raw_data = raw_data.transpose(self.transpose)

        if self.n_weights is not None:
            raw_data = raw_data[:self.n_weights]

        raw_data /= max(raw_data.max(), 1e-8)
        raw_data *= .5
        raw_data += .5

        return raw_data


class ImageSamplesDisplay(ImageGetter):
    def __init__(self, model, num_samples=100, ** kwargs):
        self.sample = theano.function([], model.sample(num_samples))

        super(ImageSamplesDisplay, self).__init__(** kwargs)

    def get_image_data(self):
        raw_data = self.sample()

        return raw_data
