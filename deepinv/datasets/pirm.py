from typing import Callable
from types import MappingProxyType
import os

from deepinv.datasets.utils import (
    calculate_md5_for_folder,
    download_archive,
    extract_tarball,
)
from deepinv.datasets.base import ImageFolder
from .utils import resolve_root


class PIRMHR(ImageFolder):
    """Dataset for `PIRM <https://pirm.github.io/>`_.

    The PIRM dataset :footcite:p:`blau20182018` is a dataset consisting of 200 images commonly used for testing performance
    of perceptual image super-resolution algorithms, split into two equal sets of 100 images for validation and testing.
    Images have variable sizes, typically around 300,000 pixels in total (e.g. 412×620 to 704×648 pixels).

    **Raw data file structure:** ::

        self.root --- PIRM_valid_HR.tar.gz
                |
                --- PIRM_valid_HR --- 1.png
                |                  |
                |                  --- 2.png
                |                  --- ...
                |
                --- PIRM_test_HR.tar.gz
                |
                --- PIRM_test_HR --- 201.png
                |                 |
                |                 --- 202.png
                |                 --- ...
                |
                --- xxx

    Raw dataset source : https://huggingface.co/datasets/eugenesiow/PIRM

    :param str root: Root directory of dataset. Directory path from where we load and save the dataset.
    :param str mode: Select a split of the dataset between ``'val'`` (100 validation images) or ``'test'`` (100 test images).
        Default at ``'val'``.
    :param bool download: If ``True``, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again. Default at False.
    :param Callable transform:: (optional)  A function/transform that takes in a PIL image
        and returns a transformed version. E.g, ``torchvision.transforms.RandomCrop``
    :param bool verbose: Print a message if the dataset has been correctly downloaded. Default ``True``.
    :param bool use_dict_output: whether to return output as dict with keys "x", "y", "params" instead of tuple (default `False`).

    """

    _archive_urls = MappingProxyType(
        {
            "PIRM_valid_HR.tar.gz": "https://huggingface.co/datasets/eugenesiow/PIRM/resolve/main/data/PIRM_valid_HR.tar.gz",
            "PIRM_test_HR.tar.gz": "https://huggingface.co/datasets/eugenesiow/PIRM/resolve/main/data/PIRM_test_HR.tar.gz",
        }
    )
    _checksums = MappingProxyType(
        {
            "PIRM_valid_HR": "5aa08bad3de701064350ead668f98d43",
            "PIRM_test_HR": "9dfe66fabb33e68a0a8d134696b08eae",
        }
    )
    # for integrity of downloaded data

    def __init__(
        self,
        root: str = None,
        mode: str = "val",
        download: bool = False,
        transform: Callable = None,
        verbose: bool = True,
        use_dict_output: bool = False,
    ) -> None:
        self.root = resolve_root(root, "PIRM")

        if mode == "val":
            self.folder_name = "PIRM_valid_HR"
        elif mode == "test":
            self.folder_name = "PIRM_test_HR"
        else:
            raise ValueError(
                f"Expected `val` or `test` values for `mode` argument, instead got `{mode}`"
            )
        self.mode = mode
        self.img_dir = os.path.join(self.root, self.folder_name)

        # download dataset, we check first that dataset isn't already downloaded
        if not self.check_dataset_exists():
            if download:
                if not os.path.isdir(self.root):
                    os.makedirs(self.root)
                if os.path.exists(self.img_dir):
                    raise ValueError(
                        f"The image folder already exists, thus the download is aborted. Please set `download=False` OR remove `{self.img_dir}`."
                    )

                archive_filename = f"{self.folder_name}.tar.gz"
                download_archive(
                    url=self._archive_urls[archive_filename],
                    save_path=os.path.join(self.root, archive_filename),
                )
                extract_tarball(os.path.join(self.root, archive_filename), self.root)

                if self.check_dataset_exists() and verbose:
                    print("Dataset has been successfully downloaded.")
                else:
                    raise ValueError("There is an issue with the data downloaded.")
            # stop the execution since the dataset is not available and we didn't download it
            else:
                raise RuntimeError(
                    f"Dataset not found at `{self.root}`. Please set `root` correctly (currently `root={self.root}`) OR set `download=True` (currently `download={download}`)."
                )

        super().__init__(
            self.img_dir, transform=transform, use_dict_output=use_dict_output
        )

    def check_dataset_exists(self) -> bool:
        """Verify that the image folder for the selected split exists and contains all the images.

        ``self.root`` should have the following structure: ::

            self.root --- PIRM_valid_HR --- 1.png
                    |                    |
                    |                    --- 2.png
                    |                    --- ...
                    |
                    --- PIRM_test_HR --- 201.png
                    |                 |
                    |                 --- 202.png
                    |                 --- ...
                    |
                    --- xxx
        """
        if not os.path.isdir(self.img_dir):
            return False
        return (
            calculate_md5_for_folder(self.img_dir) == self._checksums[self.folder_name]
        )
