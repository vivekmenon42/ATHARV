"""
Solar wind and CME analysis utilities.

This module provides functions for:
- Coordinate transformations (RTN ↔ HGI, spherical ↔ cartesian)
- Time series interpolation
- Vector filtering and masking
- CME position calculation and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, PchipInterpolator
from matplotlib.animation import PillowWriter
from PIL import Image

# ============================================================================
# INTERPOLATION FUNCTIONS
# ============================================================================

def interpolate_vector_series(t_source, vectors, t_target, align=True):
    """
    Interpolates 3D vector time series to new timestamps.

    Parameters:
    - t_source: array of timestamps (datetime or float), original
    - vectors: array of shape (N, 3), vectors at t_source
    - t_target: array of timestamps to interpolate to
    - align: bool, if True, clip t_target to overlap with t_source to avoid extrapolation

    Returns:
    - vectors_interp: array of shape (len(t_target), 3)
    - t_target_clipped: clipped t_target array (only if align=True, else same as input)
    """
    t_source_dt = t_source if isinstance(t_source[0], float) else t_source
    t_target_dt = t_target if isinstance(t_target[0], float) else t_target
    
    # Convert to timestamps
    t_source = np.array([t.timestamp() if hasattr(t, 'timestamp') else t for t in t_source_dt])
    t_target_orig = np.array([t.timestamp() if hasattr(t, 'timestamp') else t for t in t_target_dt])
    
    vectors = np.asarray(vectors)
    
    # Align to avoid extrapolation if requested
    if align:
        t_start_common = max(t_source[0], t_target_orig[0])
        t_end_common = min(t_source[-1], t_target_orig[-1])
        
        # Clip t_target to common range
        mask = (t_target_orig >= t_start_common) & (t_target_orig <= t_end_common)
        t_target = t_target_orig[mask]
        
        if hasattr(t_target_dt[0], 'timestamp'):
            # Return clipped datetime array
            t_target_clipped = [t for t, ok in zip(t_target_dt, mask) if ok]
        else:
            t_target_clipped = t_target
    else:
        t_target = t_target_orig
        t_target_clipped = t_target_dt
    
    # Use 'linear' interpolation with boundary value fill instead of extrapolate
    interpolators = [interp1d(t_source, vectors[:, i], kind='linear', 
                              bounds_error=False, 
                              fill_value=(vectors[0, i], vectors[-1, i]))
                     for i in range(3)]
    
    vectors_interp = np.stack([interp(t_target) for interp in interpolators], axis=1)
    
    if align:
        return vectors_interp, t_target_clipped
    else:
        return vectors_interp
    

def pchip_vector_interp(t_source, vectors, t_target, align=True):
    """
    Interpolates 3D vector time series using PCHIP (shape-preserving).

    Parameters:
    - t_source: array of timestamps (datetime or float), original
    - vectors: array of shape (N, 3), vectors at t_source
    - t_target: array of timestamps to interpolate to
    - align: bool, if True, clip t_target to overlap with t_source to avoid extrapolation

    Returns:
    - vectors_interp: array of shape (len(t_target), 3)
    - t_target_clipped: clipped t_target array (only if align=True, else same as input)
    """
    t_source_dt = t_source if isinstance(t_source[0], float) else t_source
    t_target_dt = t_target if isinstance(t_target[0], float) else t_target
    
    t_source = np.array([t.timestamp() if hasattr(t, 'timestamp') else t for t in t_source_dt])
    t_target_orig = np.array([t.timestamp() if hasattr(t, 'timestamp') else t for t in t_target_dt])
    
    # Align to avoid extrapolation if requested
    if align:
        t_start_common = max(t_source[0], t_target_orig[0])
        t_end_common = min(t_source[-1], t_target_orig[-1])
        
        mask = (t_target_orig >= t_start_common) & (t_target_orig <= t_end_common)
        t_target = t_target_orig[mask]
        
        if hasattr(t_target_dt[0], 'timestamp'):
            t_target_clipped = [t for t, ok in zip(t_target_dt, mask) if ok]
        else:
            t_target_clipped = t_target
    else:
        t_target = t_target_orig
        t_target_clipped = t_target_dt

    interpolators = [PchipInterpolator(t_source, vectors[:, i], extrapolate=False) for i in range(3)]
    vectors_interp = np.stack([interp(t_target) for interp in interpolators], axis=1)

    if align:
        return vectors_interp, t_target_clipped
    else:
        return vectors_interp

# ============================================================================
# COORDINATE TRANSFORMATIONS
# ============================================================================

def rtn_to_hgi(velocity_rtn, position_hgi):
    """
    Transforms a velocity vector from RTN to HGI coordinates.

    Parameters
    ----------
    velocity_rtn : array-like, shape (3,)
        Velocity vector in RTN coordinates [v_R, v_T, v_N]
    position_hgi : array-like, shape (3,)
        Spacecraft position vector in HGI coordinates [x, y, z]

    Returns
    -------
    velocity_hgi : ndarray, shape (3,)
        Velocity vector in HGI coordinates
    """
    velocity_rtn = np.asarray(velocity_rtn)
    position_hgi = np.asarray(position_hgi)
    
    # Sun's rotation axis in HGI (unit vector along +Z)
    omega_sun = np.array([0.0, 0.0, 1.0])

    # Compute RTN basis vectors in HGI coordinates
    R_hat = position_hgi / np.linalg.norm(position_hgi)
    T_hat = np.cross(omega_sun, R_hat)
    T_hat /= np.linalg.norm(T_hat)
    N_hat = np.cross(R_hat, T_hat)

    # Rotation matrix from RTN to HGI
    M_rtn_to_hgi = np.column_stack((R_hat, T_hat, N_hat))

    return M_rtn_to_hgi @ velocity_rtn


def batch_rtn_to_hgi(V_rtn_interp, R_sc_interp):
    """
    Apply RTN → HGI transformation to arrays of vectors.

    Parameters
    ----------
    V_rtn_interp : ndarray, shape (N, 3)
        Velocity vectors in RTN
    R_sc_interp : ndarray, shape (N, 3)
        Spacecraft positions in HGI

    Returns
    -------
    V_hgi : ndarray, shape (N, 3)
        Vectors in HGI
    """
    V_hgi = np.zeros_like(V_rtn_interp)
    
    for i in range(len(V_rtn_interp)):
        V_hgi[i] = rtn_to_hgi(V_rtn_interp[i], R_sc_interp[i])
    
    return V_hgi


def spherical_to_cartesian(r, theta, phi):
    """
    Converts spherical HGI (r, theta, phi) to Cartesian HGI (x, y, z).

    Parameters
    ----------
    r : float
        Radial distance
    theta : float
        Colatitude (angle from +Z axis), in radians
    phi : float
        Longitude (angle from +X axis), in radians

    Returns
    -------
    ndarray, shape (3,)
        Cartesian coordinates [x, y, z]
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def batch_spherical_to_cartesian(r, theta, phi, degrees=False):
    """
    Converts batch of spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    r : array-like, shape (N,)
        Radial distances
    theta : array-like, shape (N,)
        Colatitudes (angle from +Z)
    phi : array-like, shape (N,)
        Longitudes (angle from +X)
    degrees : bool, default=False
        If True, interpret theta and phi as degrees

    Returns
    -------
    ndarray, shape (N, 3)
        Cartesian coordinates [x, y, z]
    """
    r = np.asarray(r)
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    if degrees:
        theta = np.radians(theta)
        phi = np.radians(phi)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.column_stack((x, y, z))


def batch_cartesian_to_spherical(cartesian, degrees=False):
    """
    Converts array of Cartesian vectors to spherical coordinates.

    Parameters
    ----------
    cartesian : ndarray, shape (N, 3)
        Input [x, y, z] coordinates
    degrees : bool, default=False
        If True, return angles in degrees

    Returns
    -------
    r : ndarray, shape (N,)
        Radial distances
    theta : ndarray, shape (N,)
        Colatitude (0 = +Z)
    phi : ndarray, shape (N,)
        Longitude (0 = +X, increases toward +Y), wrapped to [0, 2π)
    """
    x, y, z = cartesian[:, 0], cartesian[:, 1], cartesian[:, 2]
    
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    phi = np.mod(phi, 2 * np.pi)

    if degrees:
        theta = np.degrees(theta)
        phi = np.degrees(phi)

    return r, theta, phi


def custom_rotate_frame(vectors_hgi, reference_vector):
    """
    Rotate vectors from HGI to a frame aligned with the reference vector.

    The new frame is defined such that:
    - X' axis aligns with the reference vector
    - Y' axis is perpendicular to both the solar rotation axis and X'
    - Z' axis completes the right-handed system

    Parameters
    ----------
    vectors_hgi : ndarray, shape (N, 3)
        Vectors in the HGI frame
    reference_vector : ndarray, shape (3,)
        Reference vector (e.g., initial position) in HGI coordinates

    Returns
    -------
    vectors_rotated : ndarray, shape (N, 3)
        Vectors rotated into the new trajectory-aligned frame
    """
    X_unit = reference_vector / np.linalg.norm(reference_vector)
    
    # Solar rotation axis in HGI (Z-axis)
    Omega = np.array([0, 0, 1])
    
    Y_unit = np.cross(Omega, X_unit)
    Y_unit /= np.linalg.norm(Y_unit)
    
    Z_unit = np.cross(X_unit, Y_unit)
    
    # Rotation matrix: columns are new basis vectors in HGI
    R_matrix = np.stack([X_unit, Y_unit, Z_unit], axis=1)
    
    return (R_matrix.T @ vectors_hgi.T).T


# ============================================================================
# FILTERING AND MASKING
# ============================================================================

def mask_vectors_by_components(vectors, component_bounds):
    """
    Filters 3D vectors based on component bounds and removes NaNs.

    Parameters
    ----------
    vectors : ndarray, shape (N, 3)
        Input vectors
    component_bounds : list of tuple
        [(min_x, max_x), (min_y, max_y), (min_z, max_z)]

    Returns
    -------
    mask : ndarray, shape (N,)
        Boolean mask indicating valid rows
    """
    vectors = np.asarray(vectors)
    assert vectors.shape[1] == 3, "Expecting Nx3 array for vectors"
    
    mask = np.ones(vectors.shape[0], dtype=bool)

    for i, (lo, hi) in enumerate(component_bounds):
        mask &= np.isfinite(vectors[:, i])
        mask &= (vectors[:, i] >= lo) & (vectors[:, i] <= hi)

    return mask


# ============================================================================
# VELOCITY FITTING AND ANALYSIS
# ============================================================================

def fit_velocity_edges(t_array, V_array, t_start, t_end, plot=True, title=None):
    """
    Fit velocities using linear regression and extract edge velocities.
    
    Parameters
    ----------
    t_array : list of datetime
        Time array for velocity measurements
    V_array : ndarray, shape (N, 3)
        Velocity vectors (km/s)
    t_start : datetime
        Leading edge time
    t_end : datetime
        Trailing edge time
    plot : bool, default=True
        Whether to create verification plot
    title : str, optional
        Plot title
    
    Returns
    -------
    v_leading : ndarray, shape (3,)
        Velocity at leading edge (km/s)
    v_trailing : ndarray, shape (3,)
        Velocity at trailing edge (km/s)
    v_center : ndarray, shape (3,)
        Center velocity: (v_leading + v_trailing) / 2
    v_expansion : ndarray, shape (3,)
        Expansion velocity: (v_leading - v_trailing) / 2
    """
    idx_start = np.argmin([abs(t - t_start) for t in t_array])
    idx_end = np.argmin([abs(t - t_end) for t in t_array])
    
    t_interval = t_array[idx_start:idx_end+1]
    V_interval = V_array[idx_start:idx_end+1]
    
    t_seconds = np.array([(t - t_start).total_seconds() for t in t_interval])
    
    v_leading = np.zeros(3)
    v_trailing = np.zeros(3)
    fit_params = []
    
    t_predict_start = t_seconds[0]
    t_predict_end = t_seconds[-1]
    
    # Fit each component with linear regression
    for i in range(3):
        coeffs = np.polyfit(t_seconds, V_interval[:, i], deg=1)
        slope, intercept = coeffs[0], coeffs[1]
        fit_params.append((slope, intercept))
        
        v_leading[i] = slope * t_predict_start + intercept
        v_trailing[i] = slope * t_predict_end + intercept
    
    v_center = (v_leading + v_trailing) / 2
    v_expansion = (v_leading - v_trailing) / 2
    
    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        component_names = ['X', 'Y', 'Z']
        
        for i in range(3):
            slope, intercept = fit_params[i]
            
            t_line = np.linspace(t_seconds[0], t_seconds[-1], 100)
            v_line = slope * t_line + intercept
            
            axes[i].plot(t_seconds, V_interval[:, i], 'o', alpha=0.3, 
                        label='Measured data', markersize=4)
            axes[i].plot(t_line, v_line, 'r-', linewidth=2, label='Linear fit')
            
            axes[i].axvline(t_predict_start, color='g', linestyle='--', 
                           alpha=0.5, label='Leading edge')
            axes[i].axvline(t_predict_end, color='b', linestyle='--', 
                           alpha=0.5, label='Trailing edge')
            axes[i].plot(t_predict_start, v_leading[i], 'go', markersize=10, 
                        label=f'V_LE: {v_leading[i]:.1f} km/s')
            axes[i].plot(t_predict_end, v_trailing[i], 'bs', markersize=10, 
                        label=f'V_TE: {v_trailing[i]:.1f} km/s')
            
            axes[i].set_ylabel(f'V_{component_names[i]} (km/s)', fontsize=11)
            axes[i].grid(True, alpha=0.3)
            
            info_text = f'Slope: {slope:.6f} km/s²\nIntercept: {intercept:.2f} km/s'
            axes[i].text(0.02, 0.98, info_text, transform=axes[i].transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        axes[2].set_xlabel('Time (seconds since leading edge)', fontsize=11)
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    if title:
        print(f"{title}")
    #     print(f"{'='*60}")
    # print(f"Leading Edge Velocity:  [{v_leading[0]:7.2f}, {v_leading[1]:7.2f}, {v_leading[2]:7.2f}] km/s")
    # print(f"Trailing Edge Velocity: [{v_trailing[0]:7.2f}, {v_trailing[1]:7.2f}, {v_trailing[2]:7.2f}] km/s")
    # print(f"Center Velocity:        [{v_center[0]:7.2f}, {v_center[1]:7.2f}, {v_center[2]:7.2f}] km/s")
    # print(f"Expansion Velocity:     [{v_expansion[0]:7.2f}, {v_expansion[1]:7.2f}, {v_expansion[2]:7.2f}] km/s")
    # print(f"{'='*60}\n")
    
    return v_leading, v_trailing, v_center, v_expansion


# ============================================================================
# CME POSITION CALCULATION
# ============================================================================

def calculate_cme_positions(t_array, r_sc_array, t0, t1, t2, 
                           v_hgi, 
                           v_cm_mo, v_exp_mo):
    """
    Calculate parcel positions for CME sheath and MO regions at reference time t0.
    
    Implements CME-Viz tool logic for mapping time series measurements
    to spatial positions assuming self-similar expansion.
    
    Parameters
    ----------
    t_array : list of datetime
        Measurement times
    r_sc_array : ndarray, shape (N, 3)
        Spacecraft position vectors (km) at each time
    t0 : datetime
        Leading edge of sheath strikes spacecraft
    t1 : datetime
        Trailing edge of sheath / leading edge of MO
    t2 : datetime
        Trailing edge of MO
    v_cm_mo : ndarray, shape (3,)
        Center velocity of MO region (km/s)
    v_exp_mo : ndarray, shape (3,)
        Expansion velocity of MO region (km/s)
    
    Returns
    -------
    r_positions : ndarray, shape (N, 3)
        Parcel position vectors at time t0 (km)
    sheath_mask : ndarray, shape (N,)
        Boolean array: True for sheath region, False for MO region
    """
    t_array = list(t_array)
    r_sc_array = np.asarray(r_sc_array)
    v_cm_mo = np.asarray(v_cm_mo)
    v_exp_mo = np.asarray(v_exp_mo)
    
    # Interpolate spacecraft positions at key times
    t_seconds = np.array([(t - t_array[0]).total_seconds() for t in t_array])
    t0_sec = (t0 - t_array[0]).total_seconds()
    t1_sec = (t1 - t_array[0]).total_seconds()
    t2_sec = (t2 - t_array[0]).total_seconds()
    
    r_sc_t0 = np.array([np.interp(t0_sec, t_seconds, r_sc_array[:, i]) for i in range(3)])
    r_sc_t1 = np.array([np.interp(t1_sec, t_seconds, r_sc_array[:, i]) for i in range(3)])
    r_sc_t2 = np.array([np.interp(t2_sec, t_seconds, r_sc_array[:, i]) for i in range(3)])
    
    
    # # MO region parameters
    # l0_mo = ((v_cm_mo - v_exp_mo) * (t2_sec - t1_sec) - 
    #          2 * v_exp_mo * (t1_sec - t0_sec) - (r_sc_t2 - r_sc_t1))
    
    # # Calculate r_cm_mo ensuring continuity at t1
    # dt1 = t1_sec - t0_sec
    
    # r_cm_mo = ((r_sheath_at_t1 * (1 + 2 * v_exp_mo * dt1 / l0_mo) - 
    #             r_sc_t1 + v_cm_mo * dt1) * l0_mo / (2 * v_exp_mo * dt1))
    
    # Create region masks
    idx_t1 = np.argmin([abs(t - t1) for t in t_array])
    idx_t0 = np.argmin([abs(t - t0) for t in t_array])
    
    sheath_mask = np.zeros(len(t_array), dtype=bool)
    mo_mask = np.zeros(len(t_array), dtype=bool)
    
    sheath_mask[0:idx_t1+1] = True
    mo_mask[idx_t1+1:] = True
    
    # Calculate positions
    r_positions = np.zeros_like(r_sc_array, dtype=float)
    
    # Sheath region: r = [r_sc - v_cm*dt + 2*v_exp*r_cm*dt/l0] / [1 + 2*v_exp*dt/l0]
    sheath_indices = np.where(sheath_mask)[0]
    if len(sheath_indices) > 0:
        # Get the sheath times and positions
        t_sheath = [t_array[i] for i in sheath_indices]
        r_sc_sheath = r_sc_array[sheath_indices]
        v_sheath_timeseries = v_hgi[sheath_indices]  # shape (N_sheath, 3)
        
        # Compute time differences
        dt = np.diff(t_sheath)
        dt = np.array([d.total_seconds() for d in dt])
        dt = np.append(dt, dt[-1])  # keep length = N_sheath
        
        # Compute cumulative displacement
        Vdt = v_sheath_timeseries * dt[:, np.newaxis]  # shape (N_sheath, 3)
        displacement = np.cumsum(Vdt, axis=0)
        
        # Remap positions
        r_positions[sheath_indices] = r_sc_sheath - displacement
    
    # Calculate r_cm_mo ensuring continuity at t1
    dt1 = t1_sec - t0_sec
    r_sheath_at_t1 = r_positions[idx_t1]
    l0_sheath = r_positions[idx_t0] - r_positions[idx_t1]
    l0_mo = ((v_cm_mo - v_exp_mo)*(t2_sec - t0_sec) - l0_sheath - (r_sc_t2 - r_sc_t0))
    two_v_exp_times_r_cm_mo_by_l0_mo = (r_sheath_at_t1 * (1 + 2*v_exp_mo*dt1/l0_mo) - r_sc_t1 + v_cm_mo*dt1)/dt1
    #r_cm_mo = r_sheath_at_t1 - l0_mo / 2

    
    # MO region
    for idx in np.where(mo_mask)[0]:
        t = t_array[idx]
        r_sc = r_sc_array[idx]
        dt = (t - t0).total_seconds()
        numerator = r_sc - v_cm_mo * dt + two_v_exp_times_r_cm_mo_by_l0_mo * dt
        denominator = 1 + 2 * v_exp_mo * dt / l0_mo
        r_positions[idx] = numerator / denominator
    
    # print("\n" + "="*70)
    # print("CME REMAPPING DIAGNOSTICS")
    # print("="*70)
    # # print(f"Sheath dimensions (l0):  [{l0_sheath[0]:.2e}, {l0_sheath[1]:.2e}, {l0_sheath[2]:.2e}] km")
    # # print(f"MO dimensions (l0):      [{l0_mo[0]:.2e}, {l0_mo[1]:.2e}, {l0_mo[2]:.2e}] km")
    # # print(f"Sheath center (r_cm):    [{r_cm_sheath[0]:.2e}, {r_cm_sheath[1]:.2e}, {r_cm_sheath[2]:.2e}] km")
    # # print(f"MO center (r_cm):        [{r_cm_mo[0]:.2e}, {r_cm_mo[1]:.2e}, {r_cm_mo[2]:.2e}] km")
    # print(f"\nSheath points: {np.sum(sheath_mask)}")
    # print(f"MO points:     {np.sum(mo_mask)}")
    # print("="*70 + "\n")

    # print("meow"*100)
    
    return r_positions, sheath_mask


def calculate_cme_positions_old(t_array, r_sc_array, t0, t1, t2, 
                           v_cm_sheath, v_exp_sheath, 
                           v_cm_mo, v_exp_mo):
    """
    Calculate parcel positions for CME sheath and MO regions at reference time t0.
    
    Implements CME-Viz tool logic for mapping time series measurements
    to spatial positions assuming self-similar expansion.
    
    Parameters
    ----------
    t_array : list of datetime
        Measurement times
    r_sc_array : ndarray, shape (N, 3)
        Spacecraft position vectors (km) at each time
    t0 : datetime
        Leading edge of sheath strikes spacecraft
    t1 : datetime
        Trailing edge of sheath / leading edge of MO
    t2 : datetime
        Trailing edge of MO
    v_cm_sheath : ndarray, shape (3,)
        Center velocity of sheath region (km/s)
    v_exp_sheath : ndarray, shape (3,)
        Expansion velocity of sheath region (km/s)
    v_cm_mo : ndarray, shape (3,)
        Center velocity of MO region (km/s)
    v_exp_mo : ndarray, shape (3,)
        Expansion velocity of MO region (km/s)
    
    Returns
    -------
    r_positions : ndarray, shape (N, 3)
        Parcel position vectors at time t0 (km)
    sheath_mask : ndarray, shape (N,)
        Boolean array: True for sheath region, False for MO region
    """
    t_array = list(t_array)
    r_sc_array = np.asarray(r_sc_array)
    v_cm_sheath = np.asarray(v_cm_sheath)
    v_exp_sheath = np.asarray(v_exp_sheath)
    v_cm_mo = np.asarray(v_cm_mo)
    v_exp_mo = np.asarray(v_exp_mo)
    
    # Interpolate spacecraft positions at key times
    t_seconds = np.array([(t - t_array[0]).total_seconds() for t in t_array])
    t0_sec = (t0 - t_array[0]).total_seconds()
    t1_sec = (t1 - t_array[0]).total_seconds()
    t2_sec = (t2 - t_array[0]).total_seconds()
    
    r_sc_t0 = np.array([np.interp(t0_sec, t_seconds, r_sc_array[:, i]) for i in range(3)])
    r_sc_t1 = np.array([np.interp(t1_sec, t_seconds, r_sc_array[:, i]) for i in range(3)])
    r_sc_t2 = np.array([np.interp(t2_sec, t_seconds, r_sc_array[:, i]) for i in range(3)])
    
    # Sheath region parameters
    l0_sheath = (v_cm_sheath - v_exp_sheath) * (t1_sec - t0_sec) - (r_sc_t1 - r_sc_t0)
    r_cm_sheath = r_sc_t0 - l0_sheath / 2
    
    # MO region parameters
    #l0_mo = ((v_cm_mo - v_exp_mo) * (t2_sec - t0_sec) - 
    #           l0_sheath - (r_sc_t2 - r_sc_t0))
    
    # Calculate r_cm_mo ensuring continuity at t1
    dt1 = t1_sec - t0_sec
    r_sheath_at_t1 = ((r_sc_t1 - v_cm_sheath * dt1 + 
                       2 * v_exp_sheath * r_cm_sheath * dt1 / l0_sheath) / 
                      (1 + 2 * v_exp_sheath * dt1 / l0_sheath))
    
    #r_cm_mo = ((r_sheath_at_t1 * (1 + 2 * v_exp_mo * dt1 / l0_mo) - 
    #            r_sc_t1 + v_cm_mo * dt1) * l0_mo / (2 * v_exp_mo * dt1))
    
    # Create region masks
    idx_t1 = np.argmin([abs(t - t1) for t in t_array])
    idx_t0 = np.argmin([abs(t - t0) for t in t_array])
    
    sheath_mask = np.zeros(len(t_array), dtype=bool)
    mo_mask = np.zeros(len(t_array), dtype=bool)
    
    sheath_mask[0:idx_t1+1] = True
    mo_mask[idx_t1+1:] = True
    
    # Calculate positions
    r_positions = np.zeros_like(r_sc_array, dtype=float)
    
    # Sheath region: r = [r_sc - v_cm*dt + 2*v_exp*r_cm*dt/l0] / [1 + 2*v_exp*dt/l0]
    for idx in np.where(sheath_mask)[0]:
        t = t_array[idx]
        r_sc = r_sc_array[idx]
        dt = (t - t0).total_seconds()
        
        numerator = r_sc - v_cm_sheath * dt + 2 * v_exp_sheath * r_cm_sheath * dt / l0_sheath
        denominator = 1 + 2 * v_exp_sheath * dt / l0_sheath
        
        r_positions[idx] = numerator / denominator
    
    #l0_sheath = r_positions[idx_t0] - r_positions[idx_t1]
    l0_mo = ((v_cm_mo - v_exp_mo) * (t2_sec - t0_sec) - 
              l0_sheath - (r_sc_t2 - r_sc_t0))
    r_cm_mo = r_sheath_at_t1 - l0_mo / 2
    
    # MO region
    for idx in np.where(mo_mask)[0]:
        t = t_array[idx]
        r_sc = r_sc_array[idx]
        dt = (t - t0).total_seconds()
        
        numerator = r_sc - v_cm_mo * dt + 2 * v_exp_mo * r_cm_mo * dt / l0_mo
        denominator = 1 + 2 * v_exp_mo * dt / l0_mo
        
        r_positions[idx] = numerator / denominator
    
    # print("\n" + "="*70)
    # print("CME REMAPPING DIAGNOSTICS")
    # print("="*70)
    # print(f"Sheath dimensions (l0):  [{l0_sheath[0]:.2e}, {l0_sheath[1]:.2e}, {l0_sheath[2]:.2e}] km")
    # print(f"MO dimensions (l0):      [{l0_mo[0]:.2e}, {l0_mo[1]:.2e}, {l0_mo[2]:.2e}] km")
    # print(f"Sheath center (r_cm):    [{r_cm_sheath[0]:.2e}, {r_cm_sheath[1]:.2e}, {r_cm_sheath[2]:.2e}] km")
    # print(f"MO center (r_cm):        [{r_cm_mo[0]:.2e}, {r_cm_mo[1]:.2e}, {r_cm_mo[2]:.2e}] km")
    # print(f"\nSheath points: {np.sum(sheath_mask)}")
    # print(f"MO points:     {np.sum(mo_mask)}")
    # print("="*70 + "\n")
    
    # print("Meowmeowmeowmeow"*100)
    return r_positions, sheath_mask



# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def draw_time_plane(ax, x_position, color='k', alpha=0.2):
    """
    Draw a rectangular plane at fixed X position spanning full Y-Z range.

    Parameters
    ----------
    ax : matplotlib Axes3D
        3D axes object
    x_position : float
        X coordinate for the plane
    color : str, default='k'
        Plane color
    alpha : float, default=0.2
        Transparency level
    """
    y = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 2)
    z = np.linspace(ax.get_zlim()[0], ax.get_zlim()[1], 2)
    Y, Z = np.meshgrid(y, z)
    X_plane = np.full_like(Y, x_position)
    ax.plot_surface(X_plane, Y, Z, color=color, alpha=alpha, zorder=0)


def save_gif(ani, filename, fps=20, dpi=100):
    """
    Save matplotlib animation as an optimized, looping GIF.
    
    Parameters
    ----------
    ani : matplotlib.animation.FuncAnimation
        Animation object to save
    filename : str
        Output filename (should end with .gif)
    fps : int, default=20
        Frames per second
    dpi : int, default=100
        Resolution (dots per inch)
    """
    temp_file = filename.replace(".gif", "_raw.gif")
    writer = PillowWriter(fps=fps)
    ani.save(temp_file, writer=writer, dpi=dpi)
    
    # Optimize with adaptive palette and dithering
    img = Image.open(temp_file)
    frames = []
    try:
        while True:
            frame = img.convert("P", palette=Image.ADAPTIVE, dither=Image.FLOYDSTEINBERG)
            frames.append(frame)
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    
    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        optimize=True,
        disposal=2
    )
    
    print(f"✅ Saved optimized GIF: {filename}")