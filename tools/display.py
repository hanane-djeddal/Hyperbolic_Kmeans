import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
import math
import random
from decimal import *

def affiche_resultat(DataFnorm,les_centres,l_affectation):
    plt.scatter(les_centres['Coord_X'],les_centres['Coord_Y'],color='r',marker='x')
    c=[]
    for l in l_affectation.values():
        X=[]
        for i in l:
            X.append(DataFnorm.loc[i])
        plt.scatter(pd.DataFrame(X)['Coord_X'],pd.DataFrame(X)['Coord_Y'])
        
        
from scipy.spatial import Voronoi, voronoi_plot_2d,ConvexHull
import matplotlib as mpl
from matplotlib import cm
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)
def affichage_vor(DataFnorm,l_affectation,minima,maxima,cap=""):
    speed=[]
    speed_trafic=[]
    cpt=0
    points=[]
    coordX=[]
    coordY=[]
    for l in l_affectation.values():
        X=[]
        for i in l:
            x=DataFnorm.loc[i]['Coord_X']
            y=DataFnorm.loc[i]['Coord_Y']
            speed_trafic.append(DataFnorm.loc[i]['Coord_Z'])
            coordX.append(x)
            coordY.append(y)
            points.append([x,y])
            speed.append(cpt)
        cpt+=1
    
    # make up data points
    points=np.array(points)

    # find min/max values for normalization
    #minima = min(min(y_bd),min(y_bu))
    #maxima = max(np.array(speed_trafic))
   
    # normalize chosen colormap
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.twilight_shifted)



    # compute Voronoi tesselation
    vor = Voronoi(points)
    fig=voronoi_plot_2d(vor, show_points=True, show_vertices=False, s=1)

    # plot
    regions, vertices = voronoi_finite_polygons_2d(vor)



    # colorize
    r=0
    for region in regions:
        polygon = vertices[region]
        plt.fill(*zip(*polygon), color=mapper.to_rgba(speed[r])) 
        r+=1

    """for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        #if not -1 in region:
        polygon = [vor.vertices[i] for i in region]
        plt.fill(*zip(*polygon), color=mapper.to_rgba(speed[r])) """

    plt.plot(coordX, coordY, 'ko',markersize=1)
    plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
    plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)
    #fig.tight_layout(rect=[0, 0, 0.85, 0.95])
    #ax_cb = fig.add_axes([0.85, 0.10, 0.05, 0.8])
    #cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=cm.Blues,norm=norm, orientation='vertical')
    fig.suptitle(cap, fontsize=10)
    plt.show()
    plt.show()
def affichage_vor_DCCA(clusters,coord, cap,minima,maxima):
    speed=clusters
    cpt=0
    points=np.array(coord)
    coordX=list(np.array(coord)[:,0])
    coordY=list(np.array(coord)[:,1])
    
    # make up data points

    # find min/max values for normalization
    #minima = min(min(y_bd),min(y_bu))
    #maxima = max(np.array(speed_trafic))
   
    # normalize chosen colormap
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.twilight_shifted)



    # compute Voronoi tesselation
    vor = Voronoi(points)
    fig=voronoi_plot_2d(vor, show_points=True, show_vertices=False, s=1)

    # plot
    regions, vertices = voronoi_finite_polygons_2d(vor)



    # colorize
    r=0
    for region in regions:
        polygon = vertices[region]
        plt.fill(*zip(*polygon), color=mapper.to_rgba(speed[r])) 
        r+=1

    """for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        #if not -1 in region:
        polygon = [vor.vertices[i] for i in region]
        plt.fill(*zip(*polygon), color=mapper.to_rgba(speed[r])) """

    plt.plot(coordX, coordY, 'ko',markersize=1)
    plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
    plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)
    #fig.tight_layout(rect=[0, 0, 0.85, 0.95])
    #ax_cb = fig.add_axes([0.85, 0.10, 0.05, 0.8])
    #cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=cm.Blues,norm=norm, orientation='vertical')
    fig.suptitle(cap, fontsize=10)
    plt.show()
    plt.show()