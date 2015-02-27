import numpy as np
import tables
import sys
import os

"""
Simple Python script for reading merger tree HDF5 files
in "database mode," which is optimized for extracting
quantities along the main branch of a subhalo. This code
requires PyTables (http://www.pytables.org).

For convenience, there are some built-in functions to do
simple tasks, such as:

    get_main_branch
    get_all_progenitors
    get_all_progenitors_of_root_descendant
    get_subhalos_between_root_and_given
    get_direct_progenitors
    get_all_fellow_progenitors
    get_future_branch

There is also a "linked-list mode" which allows for more flexibility,
implemented in readtreeHDF5_linkedlist.py. However, the linked-list
approach with Python is extremely memory-expensive, so there is a C++
version as well, which is recommended for the larger simulations.

Vicente Rodriguez-Gomez (vrodriguez-gomez@cfa.harvard.edu)

--------------- USAGE EXAMPLE: PRINT STELLAR MASS HISTORY ---------------

import readtreeHDF5
treedir = '/n/ghernquist/vrodrigu/MergerTrees/output/Subhalos/Illustris/L75n1820FP'
tree = readtreeHDF5.TreeDB(treedir)
snapnum = 135; subfind_id = 0
branch = tree.get_main_branch(snapnum, subfind_id, keysel=['SubhaloMassType'])
print branch.SubhaloMassType[:, 4]

-------------------------------------------------------------------------

"""

##########################################################################
# --------------------------- DATABASE MODE ------------------------------
##########################################################################

class _Subset(object):
    """
    Used to represent a subset of _AdjacentRows.
    Can be initialized with an integer or boolean array.
    """
    def __init__(self, adj_rows, indices):
        # Copy fields
        for field_name in adj_rows._fields:
            setattr(self, field_name, getattr(adj_rows, field_name)[indices])

class _AdjacentRows(object):
    """
    Used by the TreeDB class. Consists of 
    a set of adjacent rows from the merger tree file.
    Since subhalo IDs are assigned in a depth-first fashion,
    a "chunk" of adjacent rows can represent, e.g., a main branch
    or a "subtree."
    For a given file number and range of row numbers,
    create arrays containing information from the merger tree
    for the specified rows.
    """
    def __init__(self, treedir, name, row_start, row_end, row_original=None, filenum=-1, keysel=None):
        # Public attributes
        self._row_start = row_start
        self._row_end = row_end
        if row_original == None:
            self._index_given_sub = 0
        else:
            self._index_given_sub = row_original - row_start

        # Only interested in these row numbers:
        self.nrows = row_end - row_start + 1
        locs = slice(row_start, row_end+1)

        # Tree filename
        if filenum == -1:
            filename = '%s/%s.hdf5' % (treedir, name)
        else:
            filename = '%s/%s.%d.hdf5' % (treedir, name, filenum)

        # Find out which fields to add
        f = tables.openFile(filename)
        if keysel == None:
            self._fields = f.root.__members__
        else:
            self._fields = keysel

        # Add them
        for field_name in self._fields:
            setattr(self, field_name, f.root._f_getChild(field_name)[locs])
        f.close()

    def _get_subset(self, indices):
        return _Subset(self, indices)

class TreeDB(object):
    """
    Python class to extract information from merger tree files
    in "database mode."
    
    --------------- USAGE EXAMPLE: PRINT STELLAR MASS HISTORY ---------------
    import readtreeHDF5
    treedir = '/n/ghernquist/vrodrigu/MergerTrees/output/Subhalos/Illustris/L75n1820FP'
    tree = readtreeHDF5.TreeDB(treedir)
    snapnum = 135; subfind_id = 0
    branch = tree.get_main_branch(snapnum, subfind_id, keysel=['SubhaloMassType'])
    print branch.SubhaloMassType[:, 4]
    -----------------------------------------------------------------------
    """

    def __init__(self, treedir, name='tree_extended'):
        """
        Parameters
        ----------
        treedir : string
                  Directory where the merger tree files are located.
        name : string, optional
              Base name of the HDF5 files:
                  'tree' : Contains only the minimal fields which are
                           essential to the merger tree structure.
                  'tree_extended' : Contains the minimal fields, plus
                                    (optionally) all subhalo fields from
                                    the catalogs. See 'keysel' parameter.
        """
        self._treedir = treedir
        self._name = name

        # Check that a few files/paths exist
        for rel_path in ['tree_extended.hdf5', 'offsets']:
            if not os.path.exists(self._treedir + '/' + rel_path):
                print 'Path not found: ' + self._treedir + '/' + rel_path
                sys.exit()

    def get_main_branch(self, snapnum, subfind_id, keysel=None):
        """
        For a subhalo specified by its snapshot number and Subfind ID,
        return the progenitors along its main branch, i.e. all subhalos
        with IDs between SubhaloID and MainLeafProgenitorID.
        
        Parameters
        ----------
        snapnum : int
        subfind_id : int
        keysel: list of strings or None, optional
                This argument specifies which fields from the Subfind catalog
                should be loaded. By default, all columns are loaded, which
                can be extremely memory-expensive.
        """
        
        # Get row number and other info from offset tables
        f = tables.openFile('%s/offsets/offsets_%s.hdf5' % (
                self._treedir, str(snapnum).zfill(3)))
        rownum = f.root.RowNum[subfind_id]
        subhalo_id = f.root.SubhaloID[subfind_id]
        main_leaf_progenitor_id = f.root.MainLeafProgenitorID[subfind_id]
        f.close()
        if rownum == -1:
            #print 'Subhalo not found: snapnum = %d, subfind_id = %d.' % (snapnum, subfind_id)
            return None

        # Create branch instance
        row_start = rownum
        row_end = rownum + (main_leaf_progenitor_id - subhalo_id)
        branch = _AdjacentRows(self._treedir, self._name, row_start, row_end, keysel=keysel)
        return branch

    def get_all_progenitors(self, snapnum, subfind_id, keysel=None):
        """
        For a subhalo specified by its snapshot number and Subfind ID,
        return all the objects in the subtree which is rooted on the
        subhalo of interest, i.e. all subhalos with IDs between SubhaloID
        and LastProgenitorID. Note that this includes the given subhalo itself.
        
        Parameters
        ----------
        snapnum : int
        subfind_id : int
        keysel: list of strings or None, optional
                This argument specifies which fields from the Subfind catalog
                should be loaded. By default, all columns are loaded, which
                can be extremely memory-expensive.
        """

        # Get row number and other info from offset tables
        f = tables.openFile('%s/offsets/offsets_%s.hdf5' % (
                self._treedir, str(snapnum).zfill(3)))
        rownum = f.root.RowNum[subfind_id]
        subhalo_id = f.root.SubhaloID[subfind_id]
        last_progenitor_id = f.root.LastProgenitorID[subfind_id]
        f.close()
        if rownum == -1:
            #print 'Subhalo not found: snapnum = %d, subfind_id = %d.' % (snapnum, subfind_id)
            return None

        # Create branch instance
        row_start = rownum
        row_end = rownum + (last_progenitor_id - subhalo_id)
        subtree = _AdjacentRows(self._treedir, self._name, row_start, row_end, keysel=keysel)
        
        return subtree

    def get_all_progenitors_of_root_descendant(self, snapnum, subfind_id, keysel=None):
        """
        Return the subtree rooted on the root descendant of the given subhalo,
        i.e. all subhalos with IDs between RootDescendantID and
        RootDescendant->LastProgenitorID. Note that this includes
        the given subhalo itself.
        
        Parameters
        ----------
        snapnum : int
        subfind_id : int
        keysel: list of strings or None, optional
                This argument specifies which fields from the Subfind catalog
                should be loaded. By default, all columns are loaded, which
                can be extremely memory-expensive.
        """
        # Get row number and other info from offset tables
        f = tables.openFile('%s/offsets/offsets_%s.hdf5' % (
                self._treedir, str(snapnum).zfill(3)))
        rownum = f.root.RowNum[subfind_id]
        subhalo_id = f.root.SubhaloID[subfind_id]
        f.close()
        if rownum == -1:
            #print 'Subhalo not found: snapnum = %d, subfind_id = %d.' % (snapnum, subfind_id)
            return None

        # Get root_descendant_id from merger tree
        f = tables.openFile('%s/tree_extended.hdf5' % (self._treedir))
        root_descendant_id = f.root.RootDescendantID[rownum]

        # We know the row number of the root descendant without searching for it
        row_start = rownum - (subhalo_id - root_descendant_id)
        row_end = f.root.LastProgenitorID[row_start]
        f.close()

        # Create branch instance
        branch = _AdjacentRows(self._treedir, self._name, row_start, row_end, row_original=rownum, keysel=keysel)
        return branch

    def get_subhalos_between_root_and_given(self, snapnum, subfind_id, keysel=None):
        """
        Return all subhalos with IDs between RootDescendantID and
        SubhaloID (of the given subhalo), in a depth-first fashion.
        This function is used by "get_forward_branch."
        
        Parameters
        ----------
        snapnum : int
        subfind_id : int
        keysel: list of strings or None, optional
                This argument specifies which fields from the Subfind catalog
                should be loaded. By default, all columns are loaded, which
                can be extremely memory-expensive.
        """
        # Get row number and other info from offset tables
        f = tables.openFile('%s/offsets/offsets_%s.hdf5' % (
                self._treedir, str(snapnum).zfill(3)))
        rownum = f.root.RowNum[subfind_id]
        subhalo_id = f.root.SubhaloID[subfind_id]
        f.close()
        if rownum == -1:
            #print 'Subhalo not found: snapnum = %d, subfind_id = %d.' % (snapnum, subfind_id)
            return None

        # Get root_descendant_id from merger tree
        f = tables.openFile('%s/tree_extended.hdf5' % (self._treedir))
        root_descendant_id = f.root.RootDescendantID[rownum]
        f.close()

        # We know the row number of the root descendant without searching for it
        row_start = rownum - (subhalo_id - root_descendant_id)
        row_end = rownum

        # Create branch instance
        branch = _AdjacentRows(self._treedir, self._name, row_start, row_end, row_original=rownum, keysel=keysel)
        return branch

    def get_direct_progenitors(self, snapnum, subfind_id, **kwargs):
        """
        Return the subhalos for which DescendantID corresponds to the
        current subhalo.

        Parameters
        ----------
        snapnum : int
        subfind_id : int
        keysel: list of strings or None, optional
                This argument specifies which fields from the Subfind catalog
                should be loaded. By default, all columns are loaded, which
                can be extremely memory-expensive.
        """

        # Make sure that some fields are included.
        include_fields = ['SubhaloID', 'DescendantID']
        if 'keysel' in kwargs:
            tmp_list = kwargs['keysel']
            for field_name in include_fields:
                if field_name not in tmp_list:
                    tmp_list.append(field_name)
            kwargs['keysel'] = tmp_list

        subtree = self.get_all_progenitors(snapnum, subfind_id, **kwargs)
        subhalo_id = subtree.SubhaloID[0]  # unique ID of given subhalo
        indices = subtree.DescendantID == subhalo_id
        return subtree._get_subset(indices)

    def get_all_fellow_progenitors(self, snapnum, subfind_id, **kwargs):
        """
        Return all the subhalos that will merge into the same object
        at any point in the future, i.e. those subhalos for which
        RootDescendantID equals RootDescendantID of the current subhalo.

        Parameters
        ----------
        snapnum : int
        subfind_id : int
        keysel: list of strings or None, optional
                This argument specifies which fields from the Subfind catalog
                should be loaded. By default, all columns are loaded, which
                can be extremely memory-expensive.
        """

        # Make sure that some fields are included.
        include_fields = ['RootDescendantID']
        if 'keysel' in kwargs:
            tmp_list = kwargs['keysel']
            for field_name in include_fields:
                if field_name not in tmp_list:
                    tmp_list.append(field_name)
            kwargs['keysel'] = tmp_list

        subtree = self.get_all_progenitors_of_root_descendant(snapnum, subfind_id, **kwargs)
        root_desc_id = subtree.RootDescendantID[subtree._index_given_sub]  # unique ID of root descendant
        indices = subtree.RootDescendantID == root_desc_id
        return subtree._get_subset(indices)

    def get_future_branch(self, snapnum, subfind_id, **kwargs):
        """
        Return the subhalos found in a sort of "forward" branch between
        SubhaloID and RootDescendantID. Note that these subhalos are not
        necessarily stored in adjacent rows, as is the case
        with a main branch (following FirstProgenitor links).

        Parameters
        ----------
        snapnum : int
        subfind_id : int
        keysel: list of strings or None, optional
                This argument specifies which fields from the Subfind catalog
                should be loaded. By default, all columns are loaded, which
                can be extremely memory-expensive.
        """

        # Make sure that some fields are included.
        include_fields = ['SubhaloID', 'DescendantID', 'RootDescendantID']
        if 'keysel' in kwargs:
            tmp_list = kwargs['keysel']
            for field_name in include_fields:
                if field_name not in tmp_list:
                    tmp_list.append(field_name)
            kwargs['keysel'] = tmp_list

        subtree = self.get_subhalos_between_root_and_given(snapnum, subfind_id, **kwargs)
        # Unfortunately, there are no shortcuts in this case and we must
        # proceed iteratively. This is almost at the limit of what one
        # can do when reading trees in "database mode."
        desc_id = subtree.DescendantID[subtree._index_given_sub]
        root_desc_id = subtree.RootDescendantID[subtree._index_given_sub]
        indices = [subtree._index_given_sub]
        while desc_id != root_desc_id:
            cur_index = np.where(subtree.SubhaloID == desc_id)[0][0]
            indices.append(cur_index)
            desc_id = subtree.DescendantID[cur_index]
        indices = indices[::-1]  # reverse
        return subtree._get_subset(indices)

