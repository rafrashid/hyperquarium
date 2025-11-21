import logging
import typing as T
from pathlib import Path

import xarray as xr

logger = logging.getLogger(__name__)


class CubeLoader:

    def __init__(self, data_root: str, filename: str):
        self.data_root = Path(data_root)
        assert self.data_root.is_dir()
        self.filename = filename

    def __str__(self):
        return f'{self.__class__.__name__}(' \
               f'{self.data_root}, {self.filename})'

    def __getitem__(self, netcdf_file):
        return self.load(netcdf_file)

    def load(self, netcdf_file) -> xr.DataArray:
        logger.info(f'Loading {netcdf_file}')
        cube = xr.open_dataarray(netcdf_file)
        return cube


class MultiCubeLoader(CubeLoader):
    """
    Loads multiple data cubes. Returns a dataset. Coordinates and dimensions of loaded cubes must be compatible.
    """

    def __init__(self, loaders: T.Dict[str, CubeLoader],
                 method='load'):
        self._loaders = loaders
        self.method_name = str(method)

    def __str__(self):
        return f'MultiCubeLoader({self._loaders})'

    def load_from(self, netcdf_file, *args, **kwargs) -> xr.Dataset:
        cubes = {}

        for name, loader in self._loaders.items():
            method = getattr(loader, self.method_name)
            logger.debug(f'Loading {name}: {netcdf_file} with {method}')
            cube = method(netcdf_file, *args, **kwargs)
            cubes[name] = cube

        ds = xr.Dataset(cubes)
        return ds

    def __getitem__(self, netcdf_file):
        return self.load_from(netcdf_file)
