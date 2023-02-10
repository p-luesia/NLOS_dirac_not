
import tal
import numpy as np


def read_observations(data_file: str):
    """
    Read the observations from the file data_file
    @param data_file    : Filepath to the data source file
    @return (Tmn, ks, kl, offset_ks, offset_kl, kt) :
        Tmn: 3D observations of m sensor point and n laser points in time t
        ks: positions of the m sensor points in the relay wall
        kl: positions of the n laser points in the relay wall
        offset_ks: the distance from the relay wall sensor points to the sensor
        offset_kl: the distance from the relay wall laser points to the laser
        kt: time stamps of the signal at the index i
    """
    data = tal.io.read_capture(data_file)

    # Create all the points of the laser and sensor grids into a vector
    kl = data.laser_grid_xyz.reshape(-1, 3)
    ks = data.sensor_grid_xyz.reshape(-1, 3)

    # Offset between the grid and the emitter, sensing points
    offset_kl = np.linalg.norm(kl - data.laser_xyz, axis=-1)
    offset_ks = np.linalg.norm(ks - data.sensor_xyz, axis=-1)

    # Vector of timestamps
    kt = data.delta_t * np.arange(data.H.shape[0])

    # Data in a 3d matrix of observations
    planar_H = data.H.reshape((kt.shape[0], ks.shape[0], kl.shape[0]))
    # Reshape to match the coordinates
    T = np.moveaxis(planar_H, 0,-1)
    # Return the data
    return (T, ks, kl, offset_ks, offset_kl, kt)
