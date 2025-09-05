#!/usr/bin/env python
# coding: utf-8

'''
Visualisation template for plotting model data on its icosahedral grid on a map

Inputs:
- ICON/eCLM grid file
- External parameter file / Surface file
- Rotated pole longitude and latitude

Outputs:
- 2D plot of eCLM/ICON variables
'''

import time
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

__authors__ = "Stefan POLL"
__references__ = [
    "https://github.com/HPSCTerrSys/TSMP2_auxiliary-tools",
    "https://docs.dkrz.de/doc/visualization/sw/python/source_code/python-matplotlib-example-unstructured-icon-triangles-plot-python-3.html",
    "https://docs.dkrz.de/doc/visualization/sw/python/source_code/python-matplotlib-triangular-grid-with-tripcolor-ICON.html",
    ]

def load_datasets(dsPnFn: str, gridPnFn: str):
    ds = xr.open_dataset(dsPnFn)
    dsGrid = xr.open_dataset(gridPnFn)
    return ds, dsGrid


def extract_coordinates(dsGrid, model):
    if (model=='eclm'):
        clon = np.squeeze(dsGrid.xc.values) # ncells
        clat = np.squeeze(dsGrid.yc.values) # ncells
        vlon = np.squeeze(dsGrid.xv.values) # ncells, 3
        vlat = np.squeeze(dsGrid.yv.values) # ncells, 3
    else:
        clon = np.rad2deg(dsGrid.clon.values) # ncells
        clat = np.rad2deg(dsGrid.clat.values) # ncells
        vlon = np.rad2deg(dsGrid.vlon.values) # nvortex
        vlat = np.rad2deg(dsGrid.vlat.values) # nvortex
    return clon, clat, vlon, vlat


def create_mask(clon, clat):
    return (
        (clat > 0)
        & (clat < 90)
        & (clon > -180)
        & (clon < 180)
    )


def make_triangulation(vlon, vlat, mask=None):
    """
    Convert (ncells, 3) lon/lat arrays into a matplotlib Triangulation.
    """
    if mask is not None:
        vlon = vlon[mask]
        vlat = vlat[mask]

#    points = np.column_stack([vlon.ravel(), vlat.ravel()])
#
#    coords, inverse_idx = np.unique(points, axis=0, return_inverse=True)
#    x = coords[:, 0]
#    y = coords[:, 1]
#
#    triangles = inverse_idx.reshape(vlon.shape[0], vlon.shape[1])  # (ncells, 3)
#    triang = tri.Triangulation(x, y, triangles)

#    triangles = np.arange(vlon.size).reshape(-1, 3)
#    points = np.column_stack([vlon.ravel(), vlat.ravel()])
#    triang = tri.Triangulation(points[:,0], points[:,1], triangles)

#    points = np.column_stack([vlon.ravel(), vlat.ravel()])
#    coords, inverse_idx = np.unique(points, axis=0, return_inverse=True)
#    triangles = inverse_idx.reshape(vlon.shape[0], 3)
#    triang = tri.Triangulation(coords[:, 0], coords[:, 1], triangles)

    ncells = vlon.shape[0]
    triang = np.zeros((ncells,vlon.shape[1] , 2), np.float32)
    for i in range(0, ncells, 1):
        triang[i,:,0] = np.array(vlon[i,:])
        triang[i,:,1] = np.array(vlat[i,:])

    return triang


def get_triangulation(dsGrid, mask, vlon, vlat, model):
    if (model=='eclm'):
        # check mask for eCLM
        triang = make_triangulation(vlon, vlat, mask)
        used_vertices = np.arange(vlon.shape[0]) 
    else:
        voc = dsGrid.vertex_of_cell.T[mask].values - 1
        triang = tri.Triangulation(vlon, vlat, voc)
        used_vertices = np.unique(voc)
    return triang, used_vertices


def rotate_coordinates(vlon, vlat, lon_northpole: float, lat_northpole: float):
    lon_northpole_rot = np.radians(lon_northpole)
    lat_northpole_rot = np.radians(lat_northpole)

    vlon_rad = np.deg2rad(vlon)
    vlat_rad = np.deg2rad(vlat)

    vlon_rot = np.degrees(np.arctan2(
        -np.cos(vlat_rad) * np.sin(vlon_rad - lon_northpole_rot),
        -np.cos(vlat_rad) * np.sin(lat_northpole_rot) * np.cos(vlon_rad - lon_northpole_rot)
        + np.sin(vlat_rad) * np.cos(lat_northpole_rot)
    ))

    vlon_rot[vlon_rot < -180] += 360
    vlon_rot[vlon_rot > 180] -= 360

    vlat_rot = np.degrees(np.arcsin(
        np.sin(vlat_rad) * np.sin(lat_northpole_rot)
        + np.cos(vlat_rad) * np.cos(lat_northpole_rot) * np.cos(vlon_rad - lon_northpole_rot)
    ))
    return vlon_rot, vlat_rot


def get_rotated_bounds(vlon_rot, vlat_rot, used_vertices):
    vlat_rot_min = vlat_rot[used_vertices].min()
    vlat_rot_max = vlat_rot[used_vertices].max()
    vlon_rot_min = vlon_rot[used_vertices].min()
    vlon_rot_max = vlon_rot[used_vertices].max()
    return vlon_rot_min, vlon_rot_max, vlat_rot_min, vlat_rot_max


def select_plotting_var(ds, variable, mask):

    if ( variable == 'terrain' ):
        var = np.ma.masked_where(
               ds['FR_LAND'][:].isel(cell=mask).values <= 0.5,
               ds['topography_c'][:].isel(cell=mask).values
              )
        cmap_used = build_colormap_terrain()

    if ( variable == 'pft' ):
        pct_dompft=ds['PCT_NAT_PFT'][:].isel(gridcell=mask).idxmax(dim="natpft",skipna=True).values
        var = np.ma.masked_where(
               ds['LANDFRAC_PFT'][:].isel(gridcell=mask).values <= 0.5,
               pct_dompft
              )
        cmap_used = build_colormap_pft()

    return var, cmap_used

def build_colormap_pft():

    # create colormap PFT
    pftcol = np.array( [[138/256, 102/256, 66/256, 1], # 1
                       [110/256, 139/256,  61/256, 1], # 2
                       [188/256, 238/256, 104/256, 1], # 3
                       [  0/256, 205/256,   0/256, 1], # 4
                       [169/256, 169/256, 169/256, 1], # 5
                       [ 91/256,  79/256,  61/256, 1], # 6
                       [169/256, 169/256, 169/256, 1], # 7
                       [ 20/256, 100/256,  40/256, 1], # 8
                       [169/256, 169/256, 169/256, 1], # 9
                       [169/256, 169/256, 169/256, 1], #10
                       [  0/256, 255/256,   0/256, 1], #11
                       [169/256, 169/256, 169/256, 1], #12
                       [169/256, 169/256, 169/256, 1], #13
                       [ 20/256, 220/256,  20/256, 1], #14
                       [169/256, 169/256, 169/256, 1], #15
                       [238/256, 216/256, 174/256, 1], #16
                       [205/256,   0/256,   0/256, 1]])#17

    return ListedColormap(pftcol, name="pftcmp") 

def build_colormap_terrain(base_cmap="terrain", n_colors=50):
    cmap = plt.get_cmap(base_cmap)
    colors = cmap(np.linspace(0, 1, 500))

    # Detect and filter blue-ish colors
    blue_mask = (colors[:, 2] > colors[:, 0]) & (colors[:, 2] > colors[:, 1])
    non_blue_colors = colors[~blue_mask]

    # Resample to n_colors
    indices = np.linspace(0, len(non_blue_colors) - 1, n_colors).astype(int)
    colors_resampled = non_blue_colors[indices]

    return ListedColormap(colors_resampled, name=f"{base_cmap}_no_blue")


def plot_map(var, triang, mask, vlon_rot_min, vlon_rot_max, vlat_rot_min, vlat_rot_max, cmap, plotFn, model):
    crs_data = ccrs.PlateCarree()
    crs_map = ccrs.RotatedPole(pole_longitude=-162, pole_latitude=39.25)

    fig1 = plt.figure(figsize=(5.0, 5.0))
    ax1 = plt.subplot(111, projection=crs_map)
    ax1.set_aspect('equal')

    ax1.coastlines(resolution='50m', linewidth=0.5)
    ax1.add_feature(cfeature.OCEAN, color='azure')
#    ax1.set_title('ICON external parameters (EUR-12),\nplotting demo with icosahedral grid', fontsize=9)
    
    if (model=='eclm'):
        levelsVals = (np.arange(17)+1)
    else:
        levelsVals = (np.arange(51) * 50)

#    print("Triangles:", triang.triangles.shape[0])
#    print("Vertices:", triang.x.shape[0])
    print("Var:", var.shape)
   
    if (model=='eclm'):
      pdo = ax1.tricontourf(triang, var, transform=crs_data)
    else:
      pdo = ax1.tripcolor(
        triang,
        facecolors=var,
        transform=crs_data,
        shading='flat',
#        shading='gouraud',
        edgecolors='none',
        rasterized=False,
        cmap=cmap,
        vmin=levelsVals[0],
        vmax=levelsVals[-1]
    )

    plt.xlim(vlon_rot_min, vlon_rot_max)
    plt.ylim(vlat_rot_min, vlat_rot_max)

    cb = plt.colorbar(pdo, ax=ax1, extend='both', pad=0.03, shrink=0.8, orientation='horizontal', ticks=levelsVals[::10])
    cb.ax.tick_params(labelsize=8)
    cb.set_label('Surface altitude [m]', fontsize=9)

    fig1.savefig(plotFn, bbox_inches='tight', pad_inches=0.1, dpi=2000)
#    fig1.savefig('./map_output_no_blue.png', bbox_inches='tight', pad_inches=0.1, dpi=2000)

    plt.show()


def main():
    t1 = time.time()

    lon_northpole = -162.0
    lat_northpole = 39.25

#    variable = 'terrain'
    variable = 'pft'

    if (variable == 'pft'):
       model = 'eclm'
       dsPnFn = '/p/scratch/cslts/poll1/sim/euro-cordex/wfe_eur-11_icon-eclm-pfl_prod/dta/geo/eclm/static/surfdata_ICON-11_hist_16pfts_Irrig_CMIP6_simyr2000_c230302_gcvurb-pfsoil_halo.nc'
       gridPnFn = '/p/scratch/cslts/poll1/sim/euro-cordex/wfe_eur-11_icon-eclm-pfl_prod/dta/geo/eclm/static/domain.lnd.ICON-11_ICON-11.230302_landlake_halo.nc'
    else:
       model = 'icon'
       dsPnFn = '/p/scratch/cslts/poll1/sim/euro-cordex/wfe_eur-11_icon-eclm-pfl_prod/dta/geo/icon/static/external_parameter_icon_europe011_DOM01_tiles.nc'
       gridPnFn = '/p/scratch/cslts/poll1/sim/euro-cordex/wfe_eur-11_icon-eclm-pfl_prod/dta/geo/icon/static/europe011_DOM01.nc'
    plotFn = f'./map_{model}_{variable}_EUR-12.pdf'

    ds, dsGrid = load_datasets(dsPnFn, gridPnFn)
    clon, clat, vlon, vlat = extract_coordinates(dsGrid, model)
    mask = create_mask(clon, clat)

    triang, used_vertices = get_triangulation(dsGrid, mask, vlon, vlat, model)
    vlon_rot, vlat_rot = rotate_coordinates(vlon, vlat, lon_northpole, lat_northpole)
    vlon_rot_min, vlon_rot_max, vlat_rot_min, vlat_rot_max = get_rotated_bounds(vlon_rot, vlat_rot, used_vertices)

    var, cmap_used = select_plotting_var(ds, variable, mask)
    plot_map(var, triang, mask, vlon_rot_min, vlon_rot_max, vlat_rot_min, vlat_rot_max, cmap_used, plotFn, model)

    print('exec wallclock time =  %0.3f s' % (time.time() - t1))


if __name__ == '__main__':
    main()

