"""
Line integral convolution algorithm.
"""

import numpy as np
from scipy.ndimage import map_coordinates


def line_integral_convolution(
    input, velocity, kernel, origin=0, order=3, weighted="average", step_size="unit_length", maximum_velocity=None
):
    """
    Line integral convolution of an input image with an arbitrary kernel. Lines are defined by a velocity field.

    Args:
      input (array_like): Random image to be convolved (can be created for instance
        with``1. * (np.random.random(shape) > 0.5)``).
      velocity (array_like): One velocity vector for each pixel. Must have shape
        ``input.shape + (input.ndim,)``. First dimensions are identical to the shape
        of the random input image, the last dimension defines the coordinate of the velocity.
      kernel (array_like): 1-D array with at least on element. Defines the convolution kernel
        (e.g. a Gaussian kernel constructed with
        scipy.stats.norm.norm.pdf(np.linspace(-3,3,50))).
        The flow direction can be visualized by using an asymmetric kernel
        in combination with an ``origin`` parameter equal to ``None`` or
        equivalently ``-(len(kernel) // 2)``.
      origin (int, optional): Placement of the filter, by default 0 (which is correct for
        symmetric kernels). The flow direction can be visualized by using an
        asymmetric kernel in combination with an ``origin`` parameter equal to
        ``None`` or equivalently ``-(len(kernel) // 2)``.
      order (int, optional): The order of the spline interpolation used for interpolating the input
        image and the velocity field, by default 3.
        See the documentation of
        ``scipy.ndimage.interpolation.map_coordinates`` for more details.
      weighted (str, optional):Can be either ``'average'`` or ``'integral'``, by default
        ``'average'``. If set to ``'average'``, the weighted average is
        computed. If set to ``'integral'``, the weighted integral is computed.
        See the examples to see which parameter is appropriate each use case.
      step_size (str, optional): Can be either ``'unit_length'`` or ``'unit_time'``, by default
        ``'unit_length'``. If set to ``'unit_length'``, the integration step is
        the velocity scaled to unit length. If set to ``'unit_time'``, the step
        equals the velocity.
      maximum_velocity (float, optional): Is ``None`` by default. If it is not ``None``, the
        velocity field is mutiplied with a scalar variable s.t. the maximum velocity after
        multiplication equals ``maximum_velocity``.

    Returns:
      ndarray: Returned array of same shape as ``input``.

    See Also:
      * scipy.ndimage.filters.convolve1d : Calculate a one-dimensional convolution
        with a kernel.
      * scipy.integrate.ode : Integrate an ordinary differential equation.
      * http://dl.acm.org/citation.cfm?id=166151
    """
    input = np.asarray(input)
    kernel = np.asarray(kernel)
    velocity = np.asarray(velocity)
    if (kernel.ndim != 1) or (len(kernel) < 1):
        raise ValueError("Kernel must be 1-D array of length at least 1")
    if velocity.shape != (input.shape + (input.ndim,)):
        raise ValueError("Shape of velocity not compatible with shape of input image")
    float_type = np.result_type(velocity.dtype, kernel.dtype, input.dtype, np.float32)
    if origin is None:
        center = 0
    else:
        center = (len(kernel) // 2) + origin
    if center < 0:
        raise ValueError("Origin parameter is too low for the chosen kernel length")
    if center >= len(kernel):
        raise ValueError("Origin paremeter is too large for the kernel length")
    if maximum_velocity is not None:
        velocity = (
            maximum_velocity * velocity / np.sqrt(np.max(np.sum(np.square(velocity.astype(float_type)), axis=-1)))
        )
    if weighted not in ("integral", "average"):
        raise ValueError("Weighted must be either 'intergral' or 'average'")
    if step_size not in ("unit_length", "unit_time"):
        raise ValueError("Step_size must be either 'unit_length' or 'unit_time'")

    if weighted == "average":
        weight = np.zeros(input.shape, float_type)
    result = np.zeros(input.shape, float_type)

    # the kernel is divided into a part left of "center" and a part right of
    # "center".
    # The part on the "right" lies in positive flow direction (sign=1),
    # the part on the "left" lies in negative flow direction (sign=-1)
    for sign, ids in [(+1, range(center, len(kernel))), (-1, reversed(range(0, center + 1)))]:
        # workaround to prevent pyflakes from complaining about undefined
        # variables
        v2, m1, v2l2 = None, None, None
        for i in ids:
            if i == center:
                # initialize position and mask
                pos = np.mgrid[[slice(None, s) for s in input.shape]].astype(float_type)
                m = np.ones(input.shape, dtype=bool)
            else:
                # advance position using velocity
                if step_size == "unit_length":
                    denominator = v2l2[m1][None, :]
                elif step_size == "unit_time":
                    denominator = 1
                pos[:, m] += sign * v2[:, m1] / denominator
            # velocity at current position
            v2 = np.array(
                [
                    map_coordinates(velocity[..., axis], pos[:, m], order=order, output=float_type)
                    for axis in range(input.ndim)
                ]
            )
            # l2-norm of velocity
            v2l2 = np.sqrt(np.sum(np.square(v2), axis=0))
            # update mask
            m1 = v2l2 > 0
            m[m] = m1
            # update result
            if (sign == 1) or (i != center):
                if step_size == "unit_length":
                    ki = kernel[i]
                elif step_size == "unit_time":
                    ki = kernel[i] * v2l2[m1]
                if i == center:
                    result[m] += ki * input[m]
                else:
                    result[m] += ki * map_coordinates(input, pos[:, m], order=order, output=float_type)
                if weighted == "average":
                    weight[m] += ki
    if weighted == "average":
        return result / weight
    else:
        return result


def lic_test_plot():
    """Visualize a 2-D vortex with different configurations of the ``line_integral_convolution`` algorithm.

    Use the 2-D vortex to add motion blur to a sample image.
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from scipy.stats import norm

    if rcParams["savefig.dpi"] == "figure":
        dpi = rcParams["figure.dpi"]
    else:
        dpi = rcParams["savefig.dpi"]

    # The 2-D vortex that we want to visualize is described by the ``velocity`` array.
    position = np.mgrid[:150, :180].astype(float)
    velocity = np.tensordot(
        (position - np.array(position.shape[1:])[:, None, None] / 2), [[0, 1], [-1, 0]], axes=(0, 0)
    )
    # squared length of the velocity
    velocity_ssq = np.sum(np.square(velocity), axis=-1)

    # The following lines create three random boolean images. ``image5`` contains
    # an equal amount of black an white pixels. ``image01`` is mostly black and
    # contains only a few white pixels. ``image004`` is darker at pixels with
    # higher velocity, with a minimum brightness defined by the ``0.004``
    # constant and a maximum brightness controlled by the ``0.2`` constant.
    np.random.seed(0)
    image5 = np.random.random(position.shape[1:]) < 0.5
    np.random.seed(0)
    image01 = np.random.random(position.shape[1:]) < 0.01
    np.random.seed(0)
    image004 = (
        np.maximum(np.sqrt(velocity_ssq / np.max(velocity_ssq)), 0.2) * np.random.random(position.shape[1:])
    ) < 0.004

    # A symmetric Gaussian kernel (``gauss_kernel``) and an asymmetric exponential
    # kernel (``exp_kernel``) are created.
    gauss_kernel = norm.pdf(np.linspace(-3, 3, 25))
    exp_kernel = np.exp(-np.linspace(0, 3, 25))

    # The symmetric kernel in combination with the ``image5`` input
    # image can be used to visualize the flow, ignoring velocity magnitude and
    # direction.
    fig = plt.figure(figsize=(np.array(image5.T.shape) / dpi))
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(
        np.clip(line_integral_convolution(image5, velocity, gauss_kernel), 0, 1),
        cmap="Greys_r",
        interpolation="nearest",
    )
    plt.figtext(0.5, 0.05, "Standard LIC", fontsize=1000 // dpi, color="w", backgroundcolor="k", ha="center")
    fig.savefig("test1.pdf")

    # The asymmetric kernel can be used to visualize the flow direction. The
    # ``image01`` array is used instead of the ``image5`` image, and the origin is
    # set to ``None``. Also, ``weighted`` is set to ``'integral'`` to treat the
    # white pixels in the input image as single particles.
    fig = plt.figure(figsize=(np.array(image01.T.shape) / dpi))
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(
        np.clip(line_integral_convolution(image01, velocity, exp_kernel, origin=None, weighted="integral"), 0, 1),
        cmap="Greys_r",
        interpolation="nearest",
    )
    plt.figtext(0.5, 0.05, "Direction", fontsize=1000 // dpi, color="w", backgroundcolor="k", ha="center")
    fig.savefig("test2.pdf")

    # The symmetric kernel in combination with ``step_size='unit_time'`` and
    # ``maximum_velocity=2.`` visualizes the velocity magnitude and ignores the
    # direction. The largest element in the kernel is set to 1 in this case
    # (using ``gauss_kernel / np.max(gauss_kernel)``). To ensure that the resulting
    # image has the same line density everywhere, we adjust the number of white
    # pixels to the velocity by using ``image004``.
    fig = plt.figure(figsize=(np.array(image004.T.shape) / dpi))
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(
        np.clip(
            line_integral_convolution(
                image004,
                velocity,
                (gauss_kernel / np.max(gauss_kernel)),
                weighted="integral",
                step_size="unit_time",
                maximum_velocity=2.0,
            ),
            0,
            1,
        ),
        cmap="Greys_r",
        interpolation="nearest",
    )
    plt.figtext(0.5, 0.05, "Magnitude", fontsize=1000 // dpi, color="w", backgroundcolor="k", ha="center")
    fig.savefig("test3.pdf")

    # Velocity direction and magnitude can be visualized by using the same
    # configuration with the asymmetric kernel.
    fig = plt.figure(figsize=(np.array(image004.T.shape) / dpi))
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(
        np.clip(
            line_integral_convolution(
                image004,
                velocity,
                exp_kernel,
                origin=None,
                weighted="integral",
                step_size="unit_time",
                maximum_velocity=2.0,
            ),
            0,
            1,
        ),
        cmap="Greys_r",
        interpolation="nearest",
    )
    plt.figtext(0.5, 0.05, "Direction and magnitude", fontsize=1000 // dpi, color="w", backgroundcolor="k", ha="center")
    fig.savefig("test4.pdf")
