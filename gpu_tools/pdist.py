from math import sqrt

import numpy as np
import tqdm
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from scipy.spatial.distance import cdist


def gpu_bloc_matrix_pdist(
    X: np.ndarray,
    gpu_devices_to_use: list,
    chunk_size: int = None,
    force_type=np.float32,
) -> dict:
    """gpu_bloc_matrix_pdist computes the pairwise distance of observations in X
    Splits the matrix into blocs and compute each bloc's pairwise distance in order
    to ease the full pairwise computation

    Parameters
    ----------
    X : np.ndarray
        The matrix of n observation across p features on which to compute the distance
    gpu_devices_to_use : list
        List of GPU IDs to use
    chunk_size : int, optional
        Overrides the size of the blocs of X to be sent to each GPU. Be aware that
        chunk_size**2 float will have to fit into each GPU's memory. Automatically
        computed wrt the cluster's memory if None, by default None
    force_type : _type_, optional
        to force a dtype during computation, by default np.float32

    Returns
    -------
    dict
        a dict holding (i, j) => cdist(Xi, Xj) for every (i, j)
    """
    # prepare cluster
    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=gpu_devices_to_use)
    client = Client(cluster)

    # get some infos
    n_gpu = len(gpu_devices_to_use)
    if not chunk_size:
        # approx 4 byte per observation inside an NxN matrix
        chunk_size = sqrt(cluster.memory_limit / 4) / n_gpu
        print(f"Setting {chunk_size=}")

    # prepare chunks
    n_chunks = len(X) // chunk_size
    limit = chunk_size * (len(X) // chunk_size)
    X_chunks = np.split(X[:limit], n_chunks)
    if limit < len(X):
        # don't forget the tail we may have cutted
        X_chunks.append(X[limit:])
        n_chunks += 1

    # Prepare the full computation path along the n_chunk x n_chunk bloc-matrix
    ii, jj = np.meshgrid(np.arange(n_chunks), np.arange(n_chunks))
    ij_path = list(zip(ii.ravel(), jj.ravel()))
    ij_path_by_n_gpu = [
        ij_path[(i * n_gpu) : ((i + 1) * n_gpu)]
        for i in range(1 + len(ij_path) // n_gpu)
    ]

    # now loop over every Xi, Xj bloc matrix
    dists_by_coords = {}
    for path in tqdm(ij_path_by_n_gpu):
        # scatter the Xi and the Xj across the available GPUs
        As = [
            client.scatter(X_chunks[i], workers=gpu_id)
            for (i, _), gpu_id in zip(path, gpu_devices_to_use)
        ]
        Bs = [
            client.scatter(X_chunks[j], workers=gpu_id)
            for (_, j), gpu_id in zip(path, gpu_devices_to_use)
        ]
        # Proceed to bloc matrix distance on each GPU
        Ds = [
            client.submit(
                lambda XA, XB: cdist(XA, XB).astype(force_type), A, B, workers=gpu_id
            )
            for A, B, gpu_id in zip(As, Bs, gpu_devices_to_use)
        ]
        # Gather the blocs
        dists_by_coords.update(
            {coords: client.gather(D) for coords, D in zip(path, Ds)}
        )
        # Release the memory
        for gpu_element in Ds + As + Bs:
            gpu_element.release()

    # return dict bloc-coordinate => bloc-distance
    client.close()
    return dists_by_coords
