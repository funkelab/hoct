"""Dataset preparation shared by scoring and tracking."""

import tracksdata as td
from tracksdata.functional import TilingScheme

from hoct._logging import LOG
from hoct.data import FrameDataset, GraphConcatDataset, TiledRoiDataset
from hoct.data._transforms import Affine, Flip, Standardize
from hoct.features.constants import REGIONPROPS

_MEAN = [
    4.6326e02,
    2.9380e00,
    3.5649e02,
    3.4491e02,
    1.1521e01,
    2.7600e-01,
    9.6600e-01,
    5.7400e-01,
    1.6200e-01,
    1.6781e02,
    -2.7000e-02,
    5.0000e-02,
    -2.7000e-02,
    8.7012e01,
    -1.4010e00,
    5.0000e-02,
    -1.4010e00,
    8.3695e01,
    9.0000e-03,
]

_STD = [
    5.5578e02,
    7.6000e00,
    1.9588e02,
    2.2610e02,
    8.1990e00,
    2.1600e-01,
    2.8100e-01,
    1.9300e-01,
    6.9000e-02,
    6.7845e02,
    3.1670e00,
    2.8750e00,
    3.1670e00,
    5.1292e02,
    1.8274e02,
    2.8750e00,
    1.8274e02,
    3.0608e02,
    7.8000e-02,
]


def _create_dataset(
    graph: td.graph.BaseGraph,
    tiling_scheme: TilingScheme | None = None,
    window_size: int = 5,
    test_time_augs: int = 0,
    scale: tuple[float, ...] | None = None,
) -> FrameDataset | TiledRoiDataset | GraphConcatDataset:
    """
    Create a dataset from a graph.

    Parameters
    ----------
    graph : td.graph.BaseGraph
        The graph to create the dataset from.
    tiling_scheme : TilingScheme | None, default=None
        The tiling scheme to use for the dataset.
    window_size : int, default=5
        The window size to use for the dataset.
    test_time_augs : int, default=0
        The number of test time augmentations to use for the dataset.
    scale : tuple[float, ...] | None, default=None
        Physical spacing (t, [z,] y, x) to apply to dataset spatial features.
        The scaling is deterministic; ``t`` is used for graph construction and
        is not applied to spatial dataframe features.

    Returns
    -------
    FrameDataset | TiledRoiDataset | GraphConcatDataset
        The created dataset.
    """
    df_transforms = []
    if scale is not None:
        spatial_scale: tuple[float, ...] = scale[1:]
        if len(spatial_scale) == 2:
            # 2D+t inputs have a singleton z axis in the graph.
            spatial_scale = (1.0, *spatial_scale)
        elif len(spatial_scale) != 3:
            raise ValueError(f"Scale must have 3 or 4 elements (t, [z,] y, x), got {len(scale)}")

        # Scaling is a data transform rather than just an edge-construction
        # parameter.  Use fixed ranges so every dataset item receives the same
        # physical scaling before any optional random test-time augmentation.
        df_transforms.append(
            Affine(
                degree_range=(0, 0),
                scale_range=[(value, value) for value in spatial_scale],
                shear_range=((0, 0), (0, 0)),
            )
        )

    if test_time_augs > 0:
        df_transforms.extend(
            [
                Flip(columns=["z", "y", "x"], p=0.5),
                Affine(
                    degree_range=(-180, 180),
                    scale_range=[(1, 1), (1, 1), (1, 1)],
                    shear_range=((0, 0), (0, 0)),
                ),
            ]
        )

    if tiling_scheme is not None:
        LOG.info("Creating tiled ROI dataset")
        dataset = TiledRoiDataset(
            graph=graph,
            properties=REGIONPROPS,
            tiling_scheme=tiling_scheme,
            df_transforms=df_transforms,
            dict_transforms=[Standardize(mean=_MEAN, std=_STD)],
        )
    else:
        LOG.info("Creating frame dataset with window_size=%d", window_size)
        dataset = FrameDataset(
            graph=graph,
            min_window_size=window_size,
            properties=REGIONPROPS,
            df_transforms=df_transforms,
            dict_transforms=[Standardize(mean=_MEAN, std=_STD)],
        )

    if test_time_augs > 0:
        dataset = GraphConcatDataset([dataset] * test_time_augs)

    return dataset
