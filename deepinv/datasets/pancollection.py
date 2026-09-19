from __future__ import annotations
from typing import Callable
from types import MappingProxyType
from pathlib import Path

import numpy as np
import torch

from deepinv.datasets.base import ImageDataset
from deepinv.utils.tensorlist import TensorList
from .utils import resolve_root


class PanCollectionDataset(ImageDataset):
    """PanCollection pansharpening dataset.

    Returns multispectral (MS) and panchromatic (PAN) satellite image pairs for pansharpening,
    compiled in the `PanCollection <https://github.com/liangjiandeng/PanCollection>`_ benchmark
    :footcite:p:`deng2022machine` and downloaded from
    `HuggingFace <https://huggingface.co/datasets/elsting/PanCollection>`_.

    Each raw ``h5`` file contains, for every example, arrays following the Wald protocol used to
    simulate reduced-resolution pansharpening data, as described in :footcite:t:`deng2022machine`:
    the original MS and PAN images are each filtered with the sensor's modulation transfer function (MTF)
    (an almost-ideal filter for PAN) then downsampled by nearest-neighbor interpolation at a scale factor
    of 4, giving ``ms`` and ``pan``; the original, non-degraded MS image is kept as ``gt``:

    - ``gt``: original, non-degraded high-resolution MS image (not available for the ``"test_full"`` split,
      which consists of real, full-resolution measurements without a reference);
    - ``ms``: MTF-filtered and downsampled (factor 4) low-resolution MS image;
    - ``pan``: filtered and downsampled high-resolution panchromatic image.

    .. note::
        The raw files also contain an ``lms`` array (``ms`` upsampled back to ``gt``'s resolution via a
        23-tap polynomial interpolator, called ``UMS`` in the paper). It carries no information beyond
        ``ms`` and is only a convenience input for some network architectures, so it is not loaded here.

    Following :class:`deepinv.physics.Pansharpen`, this dataset returns ``x = gt`` (``nan`` for the
    ``"test_full"`` split, which has no reference) and ``y = TensorList([ms, pan])`` as the measurement.

    The dataset is split by ``satellite`` (``"gf2"``, ``"qb"``, ``"wv2"`` or ``"wv3"``, with ``C=4``
    bands except for ``"wv3"`` which has ``C=8``) and by ``split``:

    - ``"train"``, ``"valid"``: reduced-resolution patches used for training/validation;
    - ``"test"``: reduced-resolution examples (with ``gt``) used for quantitative evaluation;
    - ``"test_full"``: real, full-resolution examples (without ``gt``) used for qualitative evaluation.

    Not all combinations of ``satellite`` and ``split`` are available, see :attr:`_files` for the full list.

    This is a basic, exploratory dataset loader: values are simply normalized by a fixed per-satellite
    digital number (e.g. ``2047`` for ``"wv3"``), and no other processing (e.g. channel reordering) is done.

    |sep|

    :Examples:

        Instantiate dataset and download raw data from HuggingFace ::

            from deepinv.datasets import PanCollectionDataset
            dataset = PanCollectionDataset(
                root_dir=".",       # root directory
                satellite="gf2",    # choose satellite
                split="test",       # choose split
                download=True,      # download dataset
                use_pan=True        # return pan image too as TensorList (MS, PAN)
            )
            print(len(dataset))

    :param str, pathlib.Path root_dir: dataset root directory.
    :param str satellite: satellite name, one of ``"gf2"``, ``"qb"``, ``"wv2"``, ``"wv3"``, defaults to ``"gf2"``.
    :param str split: dataset split, one of ``"train"``, ``"valid"``, ``"test"``, ``"test_full"``, defaults to ``"test"``.
    :param bool use_pan: if ``True``, the measurement ``y`` is a :class:`deepinv.utils.TensorList` of
        ``(MS, PAN)``, if ``False``, ``y`` is just the MS image.
    :param Callable transform_ms: optional transform for multispectral images (applied to both ``x`` and the MS component of ``y``).
    :param Callable transform_pan: optional transform for panchromatic images.
    :param bool download: whether to download the dataset from HuggingFace.
    :param bool use_dict_output: whether to return output as dict with keys "x", "y", "params" instead of tuple (default `False`).
    """

    _repo_id = "elsting/PanCollection"

    _files = MappingProxyType(
        {
            ("train", "gf2"): "training_data/train_gf2_19809.h5",
            ("train", "qb"): "training_data/train_qb_17139.h5",
            ("train", "wv2"): "training_data/train_wv2_15084.h5",
            ("train", "wv3"): "training_data/train_wv3_9714.h5",
            ("valid", "gf2"): "training_data/valid_gf2_19809.h5",
            ("valid", "qb"): "training_data/valid_qb_17139.h5",
            ("valid", "wv3"): "training_data/valid_wv3_9714.h5",
            ("test", "gf2"): "test_data/test_gf2_multiExm1.h5",
            ("test", "qb"): "test_data/test_qb_multiExm1.h5",
            ("test", "wv3"): "test_data/test_wv3_multiExm1.h5",
            ("test_full", "gf2"): "test_data/test_gf2_OrigScale_multiExm1.h5",
            ("test_full", "qb"): "test_data/test_qb_OrigScale_multiExm1.h5",
            ("test_full", "wv3"): "test_data/test_wv3_OrigScale_multiExm1.h5",
        }
    )

    # digital number used to normalize each satellite's images to roughly [0, 1]
    _max_values = MappingProxyType({"gf2": 1023, "qb": 2047, "wv2": 2047, "wv3": 2047})

    def __init__(
        self,
        root_dir: str | Path = None,
        satellite: str = "gf2",
        split: str = "test",
        use_pan: bool = True,
        transform_ms: Callable = None,
        transform_pan: Callable = None,
        download: bool = False,
        use_dict_output: bool = False,
    ):
        super().__init__(use_dict_output=use_dict_output)

        key = (split, satellite)
        if key not in self._files:
            raise ValueError(
                f"No data available for split={split!r} and satellite={satellite!r}. "
                f"Available (split, satellite) combinations: {sorted(self._files.keys())}."
            )

        self.root_dir = resolve_root(root_dir, "PanCollection")
        self.relative_path = self._files[key]
        self.file_path = self.root_dir / self.relative_path
        self.satellite = satellite
        self.split = split
        self.has_gt = split != "test_full"
        self.use_pan = use_pan
        self.transform_ms = transform_ms
        self.transform_pan = transform_pan
        self.normalize = lambda x: (x / self._max_values[satellite]).astype(np.float32)

        if not self.check_dataset_exists():
            if download:
                from huggingface_hub import hf_hub_download

                print(f"Downloading {self.relative_path}")
                hf_hub_download(
                    repo_id=self._repo_id,
                    repo_type="dataset",
                    filename=self.relative_path,
                    local_dir=self.root_dir,
                )

                if self.check_dataset_exists():
                    print("Dataset has been successfully downloaded.")
                else:
                    raise ValueError("There is an issue with the data downloaded.")
            else:
                raise FileNotFoundError(
                    "Local dataset not downloaded or root_dir set incorrectly. Download by setting download=True."
                )

        import h5py

        with h5py.File(self.file_path, "r") as f:
            self.length = f["ms"].shape[0]

    def check_dataset_exists(self) -> bool:
        """Verify that the raw ``h5`` file for the chosen satellite/split exists."""
        return self.file_path.is_file()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Load ground truth ``x`` (if available) and MS (+ optionally PAN) measurement ``y``.

        :param int idx: image index
        :return: ``(x, y)`` tuple, or ``{"x": x, "y": y}`` dict if ``use_dict_output=True``
            (``x`` is omitted/``nan`` for the ``"test_full"`` split, which has no ground truth).
        """
        import h5py

        with h5py.File(self.file_path, "r") as f:
            ms = f["ms"][idx]
            pan = f["pan"][idx] if self.use_pan else None
            gt = f["gt"][idx] if self.has_gt else None

        ms = torch.from_numpy(self.normalize(ms))
        if self.transform_ms is not None:
            ms = self.transform_ms(ms)

        if self.use_pan:
            pan = torch.from_numpy(self.normalize(pan))
            if self.transform_pan is not None:
                pan = self.transform_pan(pan)
            y = TensorList([ms, pan])
        else:
            y = ms

        if self.has_gt:
            x = torch.from_numpy(self.normalize(gt))
            if self.transform_ms is not None:
                x = self.transform_ms(x)
        else:
            x = float("nan")

        if self.use_dict_output:
            out = {"y": y}
            if self.has_gt:
                out["x"] = x
            return out

        return x, y
