# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
structures.py
-------------------------------------------------------------------------------------------------------------------------
Created on Wed Apr  3 12:01:42 2024

@author: Andres Cremades Botella
Adapted for 2D PIV data (no z dimension, u and v only)

File to define the coherent structures. The file contains a class for the coherent structures:
    Class:
        - structures : 2D coherent structures base class (shapes are (shpy, shpx))
"""
import numpy as np


class structures():
    """
    2D version of the coherent structures base class.

    Shapes are (shpy, shpx). Volume becomes area, and there is no spanwise (z)
    dimension. `field_3` is kept in the init signature for backward compatibility
    but is not used; k123 becomes k12 (u^2 + v^2).

    * Functions:
        - __init__                      : initialization of the class
        - separate_structures           : connected-components labeling (2D)
        - physicalproperties_structures : physical properties (dim_x, dim_y,
                                          ymin, ymax, cg_x, cg_y, area, boxvol)
        - detect_quadrant               : detect the quadrant (u, v) of each structure
        - segmentation                  : build mat_segment / mat_segment_filtered
        - structure_u1u2                : per-structure share of |u*v|
        - structure_k123                : per-structure share of sqrt(u^2+v^2)
    """

    def __init__(self, data_in={"mat_struc":[], "field_1":[], "field_2":[], "field_3":[], "flag_sign":True,
                                "uvw_folder":"../../piv/", "uvw_file":"piv.$INDEX$.h5",
                                "dx":1, "dy":1, "L_x":319, "L_y":199, "rey":1.0,
                                "utau":1.0, "sym_quad":True, "filvol":0.0}):
        """
        Initialize the 2D structures class.

        Parameters
        ----------
        data_in : dict
            - mat_struc  : boolean mask (shpy, shpx) of grid-points in a structure
            - field_1    : u velocity fluctuation (shpy, shpx)
            - field_2    : v velocity fluctuation (shpy, shpx)
            - field_3    : unused (kept for backward-compatibility)
            - flag_sign  : if True, split structures by (sign(u), sign(v))
            - uvw_folder : folder of the flow fields
            - uvw_file   : file of the flow fields
            - dx, dy     : downsampling
            - L_x, L_y   : domain sizes
            - rey, utau  : reference Reynolds number / velocity
            - sym_quad   : if True, flip sign of v on the "inverted" side before
                           computing the quadrant. For PIV this is effectively
                           always False.
            - filvol     : area threshold for filtering small structures
        """
        from py_bin.py_class.flow_field import flow_field

        self.mat_struc = np.array(data_in["mat_struc"], dtype='bool')
        self.field_1   = np.array(data_in["field_1"])
        self.field_2   = np.array(data_in["field_2"])
        # field_3 kept for backward compat but unused in 2D
        try:
            self.field_3 = np.array(data_in["field_3"])
        except (KeyError, ValueError):
            self.field_3 = np.zeros_like(self.field_1)

        self.flag_sign = bool(data_in["flag_sign"])
        if self.flag_sign:
            self.sign_1 = np.array(np.sign(self.field_1), dtype='int')
            self.sign_2 = np.array(np.sign(self.field_2), dtype='int')

        self.uvw_folder = str(data_in["uvw_folder"])
        self.uvw_file   = str(data_in["uvw_file"])
        self.down_x     = int(data_in["dx"])
        self.down_y     = int(data_in["dy"])
        self.L_x        = float(data_in["L_x"])
        self.L_y        = float(data_in["L_y"])
        self.rey        = float(data_in["rey"])
        self.utau       = float(data_in["utau"])
        self.sym_quad   = bool(data_in["sym_quad"])
        self.filvol     = float(data_in["filvol"])

        Data_flow = {"folder":self.uvw_folder, "file":self.uvw_file,
                     "down_x":self.down_x, "down_y":self.down_y,
                     "L_x":self.L_x, "L_y":self.L_y,
                     "rey":self.rey, "utau":self.utau}
        flowfield = flow_field(data_in=Data_flow)
        flowfield.shape_tensor()
        flowfield.flow_grid()

        self.y_h_plus      = flowfield.y_h
        self.grid_dx_plus  = flowfield.delta_x
        self.grid_vol_plus = flowfield.vol_h   # (1, shpx) area per cell
        self.shpx          = flowfield.shpx
        self.shpy          = flowfield.shpy

    # ---------------------------------------------------------------------
    # Connected components labeling (2D, 4-connectivity)
    # ---------------------------------------------------------------------
    def separate_structures(self):
        """
        Populate `self.nodes` — a list where each entry is a (2, N) int array of
        [y_idx, x_idx] for the N nodes of one structure.

        When `flag_sign` is True, structures are split so that a single structure
        contains points with the same (sign(u), sign(v)) tuple.
        """
        from scipy.ndimage import label

        structure = np.ones((3, 3), dtype=int)  # 8-connectivity (matches 3D 6-conn loosely)

        self.nodes = []
        mask = self.mat_struc.astype(bool)

        if self.flag_sign:
            # 4 possible (sign_1, sign_2) combinations: (+,+), (-,+), (-,-), (+,-)
            combos = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
            for s1, s2 in combos:
                sub_mask = mask & (self.sign_1 == s1) & (self.sign_2 == s2)
                if not sub_mask.any():
                    continue
                labeled, nlab = label(sub_mask, structure=structure)
                for k in range(1, nlab + 1):
                    ys, xs = np.where(labeled == k)
                    if len(ys) == 0:
                        continue
                    self.nodes.append(np.array([ys, xs], dtype='int'))
        else:
            labeled, nlab = label(mask, structure=structure)
            for k in range(1, nlab + 1):
                ys, xs = np.where(labeled == k)
                if len(ys) == 0:
                    continue
                self.nodes.append(np.array([ys, xs], dtype='int'))

    # ---------------------------------------------------------------------
    # Physical properties of each structure
    # ---------------------------------------------------------------------
    def physicalproperties_structures(self):
        """
        Compute dim_x, dim_y, ymin, ymax, boxvol (box area), vol (area),
        cg_x, cg_y, cg_xbox, cg_ybox for each structure. 2D: no z-related
        quantities are computed, but dim_z/cg_z/cg_zbox are set to 0 for
        backward compatibility with the save/load interface.
        """
        n_struc = len(self.nodes)
        self.dim_x   = np.zeros((n_struc,), dtype="float")
        self.dim_y   = np.zeros((n_struc,), dtype="float")
        self.dim_z   = np.zeros((n_struc,), dtype="float")  # always 0 in 2D
        self.ymin    = np.zeros((n_struc,), dtype="float")
        self.ymax    = np.zeros((n_struc,), dtype="float")
        self.boxvol  = np.zeros((n_struc,), dtype="float")
        self.vol     = np.zeros((n_struc,), dtype="float")
        self.cg_x    = np.zeros((n_struc,), dtype="float")
        self.cg_y    = np.zeros((n_struc,), dtype="float")
        self.cg_z    = np.zeros((n_struc,), dtype="float")  # always 0 in 2D
        self.cg_xbox = np.zeros((n_struc,), dtype="float")
        self.cg_ybox = np.zeros((n_struc,), dtype="float")
        self.cg_zbox = np.zeros((n_struc,), dtype="float")  # always 0 in 2D
        self.inv_chn = np.zeros((n_struc,), dtype="bool")

        for nn in range(n_struc):
            pts = self.nodes[nn].astype('int')    # shape (2, N): row 0 = y, row 1 = x
            y_idx = pts[0, :]
            x_idx = pts[1, :]

            y_min_idx = int(np.min(y_idx))
            y_max_idx = int(np.max(y_idx))
            ymin = self.y_h_plus[y_min_idx]
            ymax = self.y_h_plus[y_max_idx]
            dim_y_val = np.abs(ymax - ymin)

            self.cg_xbox[nn] = np.floor(np.mean(x_idx))
            self.cg_ybox[nn] = np.floor(np.mean(y_idx))

            # Per-node cell area from grid_vol_plus (shape (1, shpx))
            cell_area = self.grid_vol_plus[0, x_idx]  # (N,)
            total_area = np.sum(cell_area)

            cg_x_val = np.sum(self.grid_dx_plus * x_idx * cell_area) / total_area
            cg_y_val = np.sum(self.y_h_plus[y_idx] * cell_area) / total_area
            self.cg_x[nn] = cg_x_val
            self.vol[nn]  = total_area

            dim_x_val = self.grid_dx_plus * (np.max(x_idx) - np.min(x_idx))

            # For non-periodic PIV we do not unwrap structures across x symmetry.
            # Keep the ymin/ymax/cg_y mapping in "plus" (wall-normal) units.
            self.ymin[nn]    = ymin
            self.ymax[nn]    = ymax
            self.cg_y[nn]    = cg_y_val
            self.inv_chn[nn] = False

            self.dim_x[nn]  = dim_x_val
            self.dim_y[nn]  = dim_y_val
            self.boxvol[nn] = dim_y_val * dim_x_val

    # ---------------------------------------------------------------------
    # Quadrant detection (Q1..Q4 based on signs of u and v)
    # ---------------------------------------------------------------------
    def detect_quadrant(self):
        """
        For each structure assign an event in {1,2,3,4} based on the dominant
        (u, v) quadrant, weighted by |(u,v)| * cell_area.

        Q1: u>0, v>0   Q2: u<0, v>0   Q3: u<0, v<0   Q4: u>0, v<0
        """
        self.mat_event = np.zeros((self.shpy, self.shpx))
        self.event     = np.zeros((len(self.nodes),))

        for nn in range(len(self.nodes)):
            pts = self.nodes[nn]
            voltot = np.zeros((4,))
            for k in range(pts.shape[1]):
                iy = pts[0, k]
                ix = pts[1, k]
                f1 = self.field_1[iy, ix]
                if self.sym_quad and self.inv_chn[nn]:
                    f2 = -self.field_2[iy, ix]
                else:
                    f2 = self.field_2[iy, ix]
                weight = np.sqrt(f1 ** 2 + f2 ** 2) * self.grid_vol_plus[0, ix]
                if f1 > 0 and f2 > 0:
                    voltot[0] += weight
                elif f1 < 0 and f2 > 0:
                    voltot[1] += weight
                elif f1 < 0 and f2 < 0:
                    voltot[2] += weight
                elif f1 > 0 and f2 < 0:
                    voltot[3] += weight

            max_event = int(np.argmax(voltot))
            self.event[nn] = max_event + 1
            for k in range(pts.shape[1]):
                self.mat_event[pts[0, k], pts[1, k]] = self.event[nn]

    # ---------------------------------------------------------------------
    # Segmentation matrix
    # ---------------------------------------------------------------------
    def segmentation(self):
        """
        Build `mat_segment` (label per node) and `mat_segment_filtered`
        (relabeled, only structures with area > filvol). Nodes not in any
        structure are 0.
        """
        self.mat_segment          = np.zeros((self.shpy, self.shpx))
        self.mat_segment_filtered = np.zeros((self.shpy, self.shpx))

        nn2 = 0
        nn3 = 0
        if len(self.nodes) > 0:
            for nn in range(len(self.nodes)):
                pts = self.nodes[nn]
                self.mat_segment[pts[0, :], pts[1, :]] = nn + 1
                if self.vol[nn] > self.filvol:
                    self.mat_segment_filtered[pts[0, :], pts[1, :]] = nn2 + 1
                    nn2 += 1
                else:
                    nn3 += 1
            total = nn2 + nn3
            self.filtstr_sum = nn3 / total if total > 0 else 0
        else:
            self.filtstr_sum = 0
        print('Percentage of filtered structures: ' + str(self.filtstr_sum * 100) + '%', flush=True)

    # ---------------------------------------------------------------------
    # Reynolds-stress share per structure
    # ---------------------------------------------------------------------
    def structure_u1u2(self):
        """
        Per-structure fraction of the total |u*v| carried by the structure.
        """
        max_struc = int(np.max(self.mat_segment))
        self.u1u2 = np.zeros((max_struc,))
        absu1u2   = np.abs(np.multiply(self.field_1, self.field_2))
        u1u2tot   = np.sum(absu1u2)
        if u1u2tot == 0:
            return
        for nn in range(max_struc - 1):
            mask = (self.mat_segment == nn + 1)
            self.u1u2[nn] = np.sum(absu1u2[mask]) / u1u2tot

    # ---------------------------------------------------------------------
    # Kinetic energy share per structure (2D: u^2 + v^2)
    # ---------------------------------------------------------------------
    def structure_k123(self):
        """
        Per-structure fraction of the total sqrt(u^2 + v^2) carried by the
        structure (name kept for backward compatibility; internally 2D).
        """
        max_struc = int(np.max(self.mat_segment))
        self.k123 = np.zeros((max_struc,))
        k_field   = np.sqrt(self.field_1 ** 2 + self.field_2 ** 2)
        k_tot     = np.sum(k_field)
        if k_tot == 0:
            return
        for nn in range(max_struc - 1):
            mask = (self.mat_segment == nn + 1)
            self.k123[nn] = np.sum(k_field[mask]) / k_tot