# -*- coding: utf-8 -*-
"""
Class to get, save and load connection matrix and sample submatrices from it
authors: Michael Reimann, András Ecker
last modified: 12.2020
"""

import h5py
from tqdm import tqdm
from scipy import sparse
import numpy as np
import pandas


class _MatrixNodeIndexer(object):
    def __init__(self, parent, prop_name):
        self._parent = parent
        self._prop = parent._vertex_properties[prop_name]

    def eq(self, other):
        pop = self._parent._vertex_properties.index.values[self._prop == other]
        return self._parent.subpopulation(pop)

    def isin(self, other):
        pop = self._parent._vertex_properties.index.values[np.in1d(self._prop == other)]
        return self._parent.subpopulation(pop)

    def le(self, other):
        pop = self._parent._vertex_properties.index.values[self._prop <= other]
        return self._parent.subpopulation(pop)

    def lt(self, other):
        pop = self._parent._vertex_properties.index.values[self._prop < other]
        return self._parent.subpopulation(pop)

    def ge(self, other):
        pop = self._parent._vertex_properties.index.values[self._prop >= other]
        return self._parent.subpopulation(pop)

    def gt(self, other):
        pop = self._parent._vertex_properties.index.values[self._prop > other]
        return self._parent.subpopulation(pop)

    def random_numerical(self, ref, n_bins=50):
        all_gids = self._prop.index.values
        ref_gids = self._parent.__extract_vertex_ids__(ref)
        assert np.isin(ref_gids, all_gids).all(), "Reference gids are not part of the connectivity matrix"

        ref_values = self._prop[ref_gids]
        hist, bin_edges = np.histogram(ref_values.values, bins=n_bins)
        value_bins = np.digitize(self._prop.values, bins=bin_edges)
        assert len(hist == len(value_bins[1:-1]))  # `digitize` returns values below and above the spec. bin_edges
        sample_gids = []
        for i in range(n_bins):
            idx = np.where(value_bins == i+1)[0]
            assert idx.shape[0] >= hist[i], "Not enough neurons at this depths to sample from"
            sample_gids.extend(np.random.choice(all_gids[idx], hist[i], replace=False).tolist())
        return self._parent.subpopulation(sample_gids)

    def random_categorical(self, ref):
        all_gids = self._prop.index.values
        ref_gids = self._parent.__extract_vertex_ids__(ref)
        assert np.isin(ref_gids, all_gids).all(), "Reference gids are not part of the connectivity matrix"

        ref_values = self._prop[ref_gids].values
        value_lst, counts = np.unique(ref_values, return_counts=True)
        sample_gids = []
        for i, mtype in enumerate(value_lst):
            idx = np.where(self._prop == mtype)[0]
            assert idx.shape[0] >= counts[i], "Not enough %s to sample from" % mtype
            sample_gids.extend(np.random.choice(all_gids[idx], counts[i], replace=False).tolist())
        return self._parent.subpopulation(sample_gids)


class ConnectivityMatrix(object):
    """Small utility class to hold a connections matrix and generate submatrices"""
    def __init__(self, *args, vertex_labels=None, vertex_properties=None,
                 edge_properties=None, default_edge_property='data', shape=None):
        """Not too intuitive init - please see `from_bluepy()` below"""
        """Initialization 1: By adjacency matrix"""
        if isinstance(args[0], np.ndarray) or isinstance(args[0], sparse.spmatrix):
            m = args[0]
            assert m.ndim == 2
            m = sparse.coo_matrix(m)
            self._edges = pandas.DataFrame({
                'row': m.row,
                'col': m.col,
                'data': m.data
            })
            if shape is None:
                shape = m.shape
        """Initialization 2: By edge-specific DataFrames"""
        if isinstance(args[0], pandas.DataFrame):
            assert 'row' in args[0] and 'col' in args[0]
            self._edges = args[0]
            if shape is None:
                shape = (np.max(self._edges['row']), np.max(self._edges['col']))

        # In the future: implement the ability to represent connectivity from population A to B.
        # For now only connectivity within one and the same population
        assert shape[0] == shape[1]
        self._shape = shape

        """Initialize vertex property DataFrame"""
        if vertex_properties is None:
            if vertex_labels is None:
                vertex_labels = np.arange(shape[0])
            self._vertex_properties = pandas.DataFrame({}, index=vertex_labels)
        elif isinstance(vertex_properties, dict):
            if vertex_labels is None:
                vertex_labels = np.arange(shape[0])
            self._vertex_properties = pandas.DataFrame(vertex_properties, index=vertex_labels)
        elif isinstance(vertex_properties, pandas.DataFrame):
            if vertex_labels is not None:
                raise ValueError("""Cannot specify vertex labels separately
                                 when instantiating vertex_properties explicitly""")
            self._vertex_properties = vertex_properties
        else:
            raise ValueError("""When specifying vertex properties it must be a DataFrame or dict""")
        assert len(self._vertex_properties) == shape[0]

        """Adding additional edge properties"""
        if edge_properties is not None:
            for prop_name, prop_mat in edge_properties.items():
                if isinstance(prop_mat, np.ndarray) or isinstance(prop_mat, sparse.spmatrix):
                    prop_mat = sparse.coo_matrix(prop_mat)
                    assert np.all(prop_mat.row == self._edges['row']) and np.all(prop_mat.col == self._edges['col'])
                    self._edges[prop_name] = prop_mat.data
                else:
                    assert len(prop_mat) == len(self._edges)
                    self._edges[prop_name] = prop_mat

        self._default_edge = default_edge_property

        self._lookup = self.__make_lookup__()
        #  NOTE: This part implements the .gids and .depth properties
        for colname in self._vertex_properties.columns:
            #  TODO: Check colname against existing properties
            setattr(self, colname, self._vertex_properties[colname].values)

        # TODO: calling it "gids" might be too BlueBrain-specific! Change name?
        self.gids = self._vertex_properties.index.values

    def __make_lookup__(self):
        return pandas.Series(np.arange(self._shape[0]), index=self._vertex_properties.index)

    def matrix_(self, edge_property=None):
        if edge_property is None:
            edge_property = self._default_edge
        return sparse.coo_matrix((self._edges[edge_property], (self._edges['row'], self._edges['col'])))

    @property
    def matrix(self):
        return self.matrix_(self._default_edge)

    def dense_matrix_(self, edge_property=None):
        return self.matrix_(edge_property=edge_property).todense()

    @property
    def dense_matrix(self):
        return self.dense_matrix_()

    def array_(self, edge_property=None):
        return np.array(self.dense_matrix_(edge_property=edge_property))

    @property
    def array(self):
        return self.array_()

    def index(self, prop_name):
        return _MatrixNodeIndexer(self, prop_name)

    @staticmethod
    def __extract_vertex_ids__(an_obj):
        from assemblyfire.assemblies import Assembly
        if isinstance(an_obj, Assembly):
            return an_obj.gids
        return an_obj

    @classmethod
    def from_bluepy(cls, blueconfig_path, gids=None):
        """
        BlueConfig based constructor
        :paramfig_path: path to BlueConfig
        :param gids: array of gids aka. nodes of the graph, if None - all excitatory gids from the circuit are used
        """
        from scipy import sparse
        from .spikes import get_bluepy_simulation
        from .utils import get_depths, get_mtypes

        sim = get_bluepy_simulation(blueconfig_path)
        if gids is None:
            from assemblyfire.utils import get_E_gids
            gids = get_E_gids(sim.circuit, sim.target)
        depths = np.asarray(get_depths(sim.circuit, gids))
        mtypes = np.asarray(get_mtypes(sim.circuit, gids))
        conv = pandas.Series(np.arange(len(gids)), index=gids)
        indptr = [0]
        indices = []
        for gid in tqdm(gids, desc="Building connectivity matrix", miniters=len(gids) / 100):
            aff = conv[np.intersect1d(sim.circuit.connectome.afferent_gids(gid), gids)]
            indices.extend(aff)
            indptr.append(len(indices))
        data = np.ones_like(indices, dtype=bool)
        adj_mat = sparse.csc_matrix((data, indices, indptr), shape=(len(gids), len(gids)))
        vertex_props = pandas.DataFrame({'depths': depths,
                                         'mtypes': mtypes},
                                        index=gids)
        return cls(adj_mat, vertex_properties=vertex_props)

    def submatrix(self, sub_gids, edge_property=None, sub_gids_post=None):
        """
        Return a submatrix specified by `sub_gids`
        :param sub_gids: Subpopulation to get the submatrix for. Can be either a list of gids, or an Assembly object
        :param sub_gids_post: (optiona) if specified, defines the postsynaptic population. Else pre- equals postsynaptic
        population
        :return: the adjacency submatrix of the specified population(s).
        """
        m = self.matrix_(edge_property=edge_property)
        if sub_gids_post is not None:
            return m[np.ix_(self._lookup[self.__extract_vertex_ids__(sub_gids)],
                            self._lookup[self.__extract_vertex_ids__(sub_gids_post)])]
        idx = self._lookup[self.__extract_vertex_ids__(sub_gids)]
        return m[np.ix_(idx, idx)]

    def dense_submatrix(self, sub_gids, edge_property=None, sub_gids_post=None):
        return self.submatrix(sub_gids, edge_property=edge_property, sub_gids_post=sub_gids_post).todense()

    def subarray(self, sub_gids, edge_property=None, sub_gids_post=None):
        return np.array(self.dense_submatrix(sub_gids, edge_property=edge_property, sub_gids_post=sub_gids_post))

    def subpopulation(self, subpop_ids, copy=True):
        """A ConnectivityMatrix object representing the specified subpopulation"""
        if not copy:
            #  TODO: Return a view on this object
            raise NotImplementedError()
        assert np.all(np.in1d(subpop_ids, self._vertex_properties.index.values))

        out_edges = pandas.DataFrame(dict([(prop,
                                            self.submatrix(subpop_ids, edge_property=prop).data)
                                           for prop in self._edges.columns]
                                          )
                                     )
        out_vertices = self._vertex_properties[subpop_ids]
        return ConnectivityMatrix(out_edges, vertex_properties=out_vertices, shape=(len(subpop_ids), len(subpop_ids)),
                                  default_edge_property=self._default_edge)
        
    def sample_vertices_n_neurons(self, ref_gids, sub_gids=None):
        """
        Return n gids sampled at random where n is the number of neurons in `ref_gids`
        :param ref_gids: Subpopulation to use as reference for sampling.
            Can be either a list of gids, or an Assembly object. Or an int, in that case it specifies the number
            of vertices to sample.
        :param sub_gids: (optional) if specified, subpopulation to sample from
            Can be either a list of gids, or an Assembly object as above
        """
        if sub_gids is not None:
            sub_gids = self.__extract_vertex_ids__(sub_gids)
            assert np.isin(sub_gids, self._vertex_properties.index.values).all(), "Sub gids are not part of the connectivity matrix"
        else:
            sub_gids = self._vertex_properties.index.values
        if hasattr(ref_gids, "__len__"):
            N = len(ref_gids)
            assert np.isin(self.__extract_vertex_ids__(ref_gids),
                           sub_gids).all(), "Reference gids are not part of sub gids"
        elif isinstance(ref_gids, int):  # Just specify the number
            N = ref_gids
        else:
            raise ValueError()

        return np.random.choice(sub_gids, N, replace=False)
        
    def sample_matrix_n_neurons(self, ref_gids, sub_gids=None):
        idx = self._lookup[self.sample_vertices_n_neurons(ref_gids, sub_gids)]
        return self.matrix[np.ix_(idx, idx)]

    def dense_sample_n_neurons(self, ref_gids, sub_gids=None):
        return self.sample_matrix_n_neurons(ref_gids, sub_gids).todense()

    def sample_population_n_neurons(self, ref_gids, sub_gids=None):
        return self.subpopulation(self.sample_vertices_n_neurons(ref_gids, sub_gids))

    def sample_n_neurons(self, ref_gids, sub_gids=None):
        return np.array(self.dense_sample_n_neurons(ref_gids, sub_gids))

    def sample_vertices_from_numerical_property(self, ref_gids, property_name='depth', sub_gids=None, n_bins=50):
        """
        Return gids with the same (binned) depth profile as `ref_gids`
        :param ref_gids: Subpopulation to use as reference for sampling.
            Can be either a list of gids, or an Assembly object
        :param sub_gids: (optional) if specified, subpopulation to sample from
            Can be either a list of gids, or an Assembly object as above
        :param n_bins: number of bins to be used to bin depth values
        """
        print("DEPRECATED! Use the .index(property_name).random_numerical functionality instead!")
        ref_gids = self.__extract_vertex_ids__(ref_gids)
        if sub_gids is not None:
            sub_gids = self.__extract_vertex_ids__(sub_gids)
            assert np.isin(sub_gids, self._vertex_properties.index.values).all(), "Sub gids are not part of the connectivity matrix"
            assert np.isin(ref_gids, sub_gids).all(), "Reference gids are not part of sub gids"
            depths = self._vertex_properties[property_name][sorted(sub_gids)]
        else:
            sub_gids = self._vertex_properties.index.values
            assert np.isin(ref_gids, sub_gids).all(), "Reference gids are not part of the connectivity matrix"
            depths = self._vertex_properties[property_name]

        ref_depths = depths[ref_gids]
        hist, bin_edges = np.histogram(ref_depths.values, bins=n_bins)
        depths_bins = np.digitize(depths.values, bins=bin_edges)
        assert len(hist == len(depths_bins[1:-1]))  # `digitize` returns values below and above the spec. bin_edges
        sample_gids = []
        for i in range(n_bins):
            idx = np.where(depths_bins == i+1)[0]
            assert idx.shape[0] >= hist[i], "Not enough neurons at this depths to sample from"
            sample_gids.extend(np.random.choice(sub_gids[idx], hist[i], replace=False).tolist())
        return sample_gids

    def sample_matrix_from_numerical_property(self, ref_gids, property_name='depth', sub_gids=None, n_bins=50):
        idx = self._lookup[self.sample_vertices_from_numerical_property(ref_gids, sub_gids=sub_gids,
                                                                        property_name=property_name, n_bins=n_bins)]
        return self.matrix[np.ix_(idx, idx)]

    def dense_sample_from_numerical_property(self, ref_gids, property_name='depth', sub_gids=None, n_bins=50):
        return self.sample_matrix_from_numerical_property(ref_gids, sub_gids=sub_gids,
                                                          property_name=property_name, n_bins=n_bins).todense()

    def sample_from_numerical_property(self, ref_gids, property_name='depth', sub_gids=None, n_bins=50):
        return np.array(self.dense_sample_from_numerical_property(ref_gids, sub_gids=sub_gids,
                                                                  property_name=property_name, n_bins=n_bins))

    def sample_vertices_from_categorical_property(self, ref_gids, property_name='mtype', sub_gids=None):
        """
        Return gids with the same mtype composition as `ref_gids`
        :param ref_gids: Subpopulation to use as reference for sampling.
            Can be either a list of gids, or an Assembly object
        :param sub_gids: (optional) if specified, subpopulation to sample from
            Can be either a list of gids, or an Assembly object as above
        """
        print("DEPRECATED! Use the .index(property_name).random_categorical functionality instead!")
        ref_gids = self.__extract_vertex_ids__(ref_gids)
        if sub_gids is not None:
            sub_gids = self.__extract_vertex_ids__(sub_gids)
            assert np.isin(sub_gids, self._vertex_properties.index.values).all(), "Sub gids are not part of the connectivity matrix"
            assert np.isin(ref_gids, sub_gids).all(), "Reference gids are not part of sub gids"
            mtypes = self._vertex_properties[property_name][sub_gids]
        else:
            sub_gids = self._vertex_properties.index.values
            assert np.isin(ref_gids, sub_gids).all(), "Reference gids are not part of the connectivity matrix"
            mtypes = self._vertex_properties[property_name]

        ref_mtypes = mtypes[ref_gids].values
        mtypes_lst, counts = np.unique(ref_mtypes, return_counts=True)
        sample_gids = []
        for i, mtype in enumerate(mtypes_lst):
            idx = np.where(mtypes == mtype)[0]
            assert idx.shape[0] >= counts[i], "Not enough %s to sample from" % mtype
            sample_gids.extend(np.random.choice(sub_gids[idx], counts[i], replace=False).tolist())
        return sample_gids

    def sample_matrix_from_categorical_property(self, ref_gids, property_name='mtype', sub_gids=None):
        idx = self._lookup[self.sample_vertices_from_categorical_property(ref_gids, property_name=property_name,
                                                                          sub_gids=sub_gids)]
        return self.matrix[np.ix_(idx, idx)]

    def dense_sample_from_categorical_property(self, ref_gids, property_name='mtype', sub_gids=None):
        return self.sample_matrix_from_categorical_property(ref_gids, property_name=property_name,
                                                            sub_gids=sub_gids).todense()

    def sample_from_categorical_property(self, ref_gids, property_name='mtype', sub_gids=None):
        return np.array(self.dense_sample_from_categorical_property(ref_gids, property_name=property_name,
                                                                    sub_gids=sub_gids))

    @classmethod
    def from_h5(cls, fn, group_name=None, prefix=None):
        if prefix is None:
            prefix = "connectivity"
        if group_name is None:
            group_name = "full_matrix"
        full_prefix = prefix + "/" + group_name
        vertex_properties = pandas.read_hdf(fn, full_prefix + "/vertex_properties")
        edges = pandas.read_hdf(fn, full_prefix + "/edges")

        with h5py.File(fn, 'r') as h5:
            data_grp = h5[full_prefix]
            shape = tuple(data_grp.attrs["NEUROTOP_SHAPE"])
            def_edge = data_grp.attrs["NEUROTOP_DEFAULT_EDGE"]
        return cls(edges, vertex_properties=vertex_properties, default_edge_property=def_edge,
                   shape=shape)

    def to_h5(self, fn, group_name=None, prefix=None):
        if prefix is None:
            prefix = "connectivity"
        if group_name is None:
            prefix = "full_matrix"
        full_prefix = prefix + "/" + group_name
        self._vertex_properties.to_hdf(fn, key=full_prefix + "/vertex_properties")
        self._edges.to_hdf(fn, key=full_prefix + "/edges")

        with h5py.File(fn, "a") as h5:
            data_grp = h5[full_prefix]
            data_grp.attrs["NEUROTOP_SHAPE"] = self._shape
            data_grp.attrs["NEUROTOP_DEFAULT_EDGE"] = self._default_edge

