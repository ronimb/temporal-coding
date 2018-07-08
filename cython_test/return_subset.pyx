import numpy as np
cimport numpy as np

DTI = np.intc
DTF = np.float32
DTB = np.bool

ctypedef np.int_t  DTI_t
ctypedef np.float_t  DTF_t
# ctypedef np.bool_t DTI_t

cpdef return_subset(DTI_t batch_size,
                  np.ndarray[DTF_t, ndim=2] samples,
                  np.ndarray[DTI_t] labels,
                  DTI_t num_samples,
                  DTI_t num_neurons):
    cdef np.ndarray[DTI_t] selected_inds
    cdef np.ndarray[DTI_t] batch_inds
    cdef np.ndarray[np.uint8_t, cast=True] ind_locs
    cdef np.ndarray[DTF_t, ndim=2] subset
    cdef np.ndarray[DTI_t] u, inv

    cdef dict samp_map

    selected_inds = np.sort(np.random.choice(range(num_samples), batch_size, replace=False))
    batch_inds = (selected_inds * num_neurons + np.arange(num_neurons)[:, np.newaxis]).flatten()

    ind_locs = np.in1d(samples[0, :], batch_inds)
    subset = np.zeros((3, ind_locs.sum()))
    subset[0, :] = samples[0, ind_locs]

    samp_map = {v: i for i, v in enumerate(selected_inds)}
    subset[0, :], neur = np.divmod(subset[0, :], num_neurons)
    u, inv = np.unique(subset[0, :].astype(int), return_inverse=True)
    subset[0, :] = np.array([samp_map[x] for x in u])[inv] * num_neurons + neur
    subset[1, :] = samples[1, ind_locs]
    subset[2, :] = samples[2, ind_locs]

    return subset, labels[selected_inds]