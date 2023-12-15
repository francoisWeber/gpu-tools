from math import sqrt

import numpy as np
import tqdm
from dask.distributed import Client as DaskClient
from dask_cuda import LocalCUDACluster
from scipy.spatial.distance import cdist


def gpu_bloc_matrix_pdist(
    X: np.ndarray,
    client: DaskClient,
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
    # get some infos
    gpu_devices_to_use = client.cluster.cuda_visible_devices
    n_gpu = len(gpu_devices_to_use)
    if not chunk_size:
        # approx 4 byte per observation inside an NxN matrix
        chunk_size = sqrt(client.cluster.memory_limit / 4) / n_gpu
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
        # Release the memory
        for gpu_element in As + Bs:
            gpu_element.release()
        # Gather the blocs
        dists_by_coords.update(
            {coords: client.gather(D) for coords, D in zip(path, Ds)}
        )
        # Release the memory
        for gpu_element in Ds:
            gpu_element.release()

    # return dict bloc-coordinate => bloc-distance
    return dists_by_coords


def gather_bloc_matrix_pdist(dists_by_coords: dict):
    return np.block(
        [
            [
                dists_by_coords.get((i, j))
                for j in range(int(sqrt(len(dists_by_coords))))
            ]
            for i in range(int(sqrt(len(dists_by_coords))))
        ]
    )


def gather_rowwise_bloc_matrix_pdist(dists_by_coords: dict, n_obs):
    full_dist = np.empty((n_obs, n_obs), dtype=np.float32)
    last_index = 0
    n_blocs = int(sqrt(len(dists_by_coords)))
    for i in range(n_blocs):
        bloc_row = np.concatenate(
            [dists_by_coords[(i, j)] for j in range(n_blocs)], axis=1
        )
        slice_start = last_index
        last_index = last_index + len(bloc_row)
        full_dist[slice_start:last_index, :] = bloc_row
    return full_dist


def gather_rowwise_bloc_matrix_pdist_from_files(
    serialized_paths_by_coords: dict, n_obs=None
):
    # infos
    n_blocs = int(sqrt(len(serialized_paths_by_coords)))
    if n_obs is None:
        # infer n_obs by loading the first bloc and last bloc and measure their shape
        sample_bloc = np.load(serialized_paths_by_coords[(0, 0)])
        first_bloc_len = len(sample_bloc)
        sample_bloc = np.load(serialized_paths_by_coords[(n_blocs - 1, 0)])
        last_bloc_len = len(sample_bloc)
        del sample_bloc
        n_obs = first_bloc_len * (n_blocs - 1) + last_bloc_len
    # gather files to rebuild full matrix
    full_dist = np.empty((n_obs, n_obs), dtype=np.float32)
    last_index = 0
    for i in tqdm(range(n_blocs)):
        bloc_row = np.concatenate(
            [np.load(serialized_paths_by_coords[(i, j)]) for j in range(n_blocs)],
            axis=1,
        )
        slice_start = last_index
        last_index = last_index + len(bloc_row)
        full_dist[slice_start:last_index, :] = bloc_row
    return full_dist


from scipy.sparse import hstack, vstack


def _hstack_serialized_matrix_from_files(
    serialized_paths_to_stack: dict, nblocs, kernel: callable = None
):
    if not kernel:
        kernel = lambda x: x
    if len(serialized_paths_to_stack) == 1:
        return vstack(
            [kernel(np.load(serialized_paths_to_stack[0][i])) for i in range(nblocs)]
        )
    elif len(serialized_paths_to_stack) == 2:
        row1 = _hstack_serialized_matrix_from_files(
            [serialized_paths_to_stack[0]], nblocs=nblocs, kernel=kernel
        )
        row2 = _hstack_serialized_matrix_from_files(
            [serialized_paths_to_stack[1]], nblocs=nblocs, kernel=kernel
        )
        return hstack([row1, row2])
    else:
        half_blocs = len(serialized_paths_to_stack) // 2
        row1 = _hstack_serialized_matrix_from_files(
            serialized_paths_to_stack[:half_blocs], nblocs=nblocs, kernel=kernel
        )
        row2 = _hstack_serialized_matrix_from_files(
            serialized_paths_to_stack[half_blocs:], nblocs=nblocs, kernel=kernel
        )
        return hstack([row1, row2])


def hstack_sparse_blocs_from_files(serialized_paths_by_coords, kernel=None):
    nblocs = int(sqrt(len(serialized_paths_by_coords)))
    serialized_paths_to_stack = [
        [serialized_paths_by_coords[(i, j)] for j in range(nblocs)]
        for i in range(nblocs)
    ]
    return _hstack_serialized_matrix_from_files(
        serialized_paths_to_stack, nblocs, kernel
    )
