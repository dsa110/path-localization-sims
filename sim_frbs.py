import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import astropy.cosmology as cosm
import scipy.integrate as integrate

# path imports
from astropy.coordinates import SkyCoord
from astropath import path
from astropath import localization
from astropath import chance
from astropath import bayesian


def angle_conversion(inc, tilt):
    '''helper function for DM_calc to convert inclination and tilt of galaxy into spherical angles for line-of-sight'''
    # rotated the point (0, 1, 0) around the x the y axes and then converted the x, y, z position into polar coords
    theta = np.arccos(np.sin(inc)*np.cos(tilt)/ 
                np.sqrt((np.sin(inc)*np.sin(tilt))**2+
                np.cos(inc)**2+
                (np.sin(inc)*np.cos(tilt))**2))+np.pi
    phi = np.arctan(np.cos(inc)/(np.sin(inc)*np.sin(tilt)))
    return theta, phi

def normalize(total, r0, mx, em=False):
    '''helper function for DM_calc function that returns a normalization constant for the electron density based on the emmision measure'''
    z0 = r0/7.3
    def ellipse_int2(z):
        r = np.sqrt(1-z**2/(z0*mx)**2)
        return r0*np.exp(-np.abs(2*z/z0))*(1-np.exp(-2*r/r0))
    result = integrate.quad(ellipse_int2, -z0*mx, z0*mx)[0]
    return total/result

def sfr_EM(r0_mpc, sfr):
    '''helper function for DM_calc which gives the H_alpha emmision measure based on the SFR tendulkar+2017'''
    a = r0_mpc*3.086e24
    Lha_int = 1.26e41 * sfr # erg s-1
    Sha = Lha_int/((a*3600*360/.673)**2/7.3)
    EM_ha = 2.75*Sha*3/17*10**(18)
    return EM_ha

def DM_calc(frb_loc, scale_height, mx, inc, tilt, sfr, plot=False):
    ''' takes in the frb location and galaxy information and returns (DM (pc cm-3), path_length, maximum density)
    '''
    # inclination [0, 1.57], tilt [0, 6.28]
    if inc > np.pi/2 or inc < 0:
        raise ValueError('incination must be in range [0, pi/2]')
    if tilt > 2*np.pi or tilt < 0:
        raise ValueError('tilt must be in range [0, 2pi]')
    mx *= 2 # extending the maximum for the ellipse so that we don't have DM = 0 
    x, y, z = frb_loc # initilizing location
    x0, y0, z0 = x, y, z
    theta, phi = angle_conversion(inc, tilt) 
    xylim = scale_height*mx # limits 
    zlim = xylim / 7.3
    
    EM = sfr_EM(scale_height, sfr)
    norm = np.sqrt(normalize(EM, scale_height*1e6, 40))
    def density(r, z, x, y): # electron density function
        return norm*np.exp(-np.absolute(r/scale_height))*np.exp(-np.absolute(z/scale_height*7.3))
    def ellipse(loc, limit): # equation of ellipse to check limits
        return np.sqrt((loc[0]**2+loc[1]**2)/(limit[0]**2)+loc[2]**2/(limit[1]**2))
    
    integral = [0] # integral initialization
    s = .0001 # stepsize in mpc
    # run until the ellipse of equal density is hit
    while ellipse((x, y, z), (xylim, zlim))<1:
        r = (x**2+y**2)**(1/2)
        integral.append(density(r, z, x, y)) # left hand riemman sum (underestimate)
        # stepping in xyz
        x1 = x + s*np.sin(theta)*np.cos(phi)
        y1 = y + s*np.sin(theta)*np.sin(phi)
        z1 = z + s*np.cos(theta)
        x, y, z = (x1, y1, z1) #resetting the location
    total_dist = math.dist([x0, y0, z0], [x, y, z])*10**6
    return sum(np.array(integral)*s*10**6), total_dist, max(integral)

def DM_scat(dm, redshift):
    '''helper function for sim_frbs which gives the scattering wrt DM by using the MW DM-tau relation and correcting for geometry and redshift'''
    tau = 3.6e-6*dm**2.2 * (1.+.00194*dm**2)*(1000/327)**(-4.0)
    return tau*3/(1+redshift)**3

def igm_dm(z, f_igm, sample=False):
    '''helper function for sim_frbs which calculates the dm contribution from the igm based on the redshift and baryonic fraction'''
    cosmo = cosm.Planck15
    Om0 = .315
    Ode0 = .685
    h = .673
    myz = 10.**((np.arange(1000)+0.5)*(np.log10(z)+3.)/1000.-3.)
    mydz = 10.**((np.arange(1000)+1.)*(np.log10(z)+3.)/1000.-3.)-10.**((np.arange(1000)+0.)*(np.log10(z)+3.)/1000.-3.)
    Iz = np.sum(mydz*(1.+myz)**2./np.sqrt(Om0*(1.+myz)**3.+Ode0))
    dm = 935.*f_igm*(h/0.7)**(-1.)*Iz
    # variation around different DMs based on sightline differences
    if sample:
        if z <1.5:
            return np.random.normal(dm, .5*dm) #should be variation during this McQuinn+2014 
        if z>=1.5:
            return np.random.normal(dm, .2*dm)
    else:
        return dm
    
def pick_off(frb, mx, pdf=None):
    '''adjusted offset picker, picks based on inclination and tilt for a 3D location within the galaxy, to allow for DM calculations. Then the location is converted into RA and DEC based on D_a. To use PATH offsets set pdf equal to the desired pdf'''
    ra = frb[0]
    dec = frb[1]
    r0 = frb[10]/.673/3 # converting to scale height 
    z0 = r0 / 7.3 
    da = frb[4]
    inc = np.arccos(frb[6])
    tilt = 0.001
    #tilt = 2*np.pi*np.random.random()
    
    # picking r and z values based on exponential dist
    r = pd.DataFrame(np.linspace(0., r0*mx, 10000))
    r_weight = [i[0] for i in np.exp(-r.values/r0)]
    z = pd.DataFrame(np.linspace(0., z0*mx, 10000))
    z_weight = [i[0] for i in np.exp(-z.values/z0)]
    zs = z.sample(n=1, weights=z_weight).values[0][0]*np.random.choice([-1, 1])
    rs = r.sample(n=1, weights=r_weight).values[0][0]
    theta = 2*np.pi*np.random.random(1)[0] 
    # this does pick within a puck but in with a mx=5 ~.003 were in a ellipse
    
    # getting x, y, z coords
    xshift = rs*np.cos(theta)
    yshift = rs*np.sin(theta)
    # rotating along the xaxis (inc)
    yrot = yshift*np.cos(inc)-zs*np.sin(inc)
    zrot = yshift*np.sin(inc)+zs*np.cos(inc)
    # rotating aling the yaxis (tile)
    xtilt = xshift*np.cos(tilt)+zrot*np.sin(tilt)
    ztilt = zrot*np.cos(tilt)-xshift*np.sin(tilt)
    # using angular diameter distance to get ra, dec
    ra += xtilt/da*180/np.pi
    dec += ztilt/da*180/np.pi

    # DM and tau calculations
    dm_host, path_length, max_dens = DM_calc((xshift, yshift, zs), r0, mx, inc, tilt, frb[6])
    dm_mw = 50 + 30 # 50 from halo, 30 from ism see Macquart+2020
    dm_igm = igm_dm(frb[5], .7, sample=True) # assuming f_baryons = .7
    dm = dm_mw + dm_igm + dm_host/(1+frb[5])
    scat_host = DM_scat(dm_host, frb[5])
    scat = scat_host + 7e-6 # MW scattering estimated from Chawla
    
    if pdf != None: # older version
        radii = pd.DataFrame(np.linspace(0., mx, 10000))
        ## only applies same offset priors as aggarwal
        weights = [i[0] for i in bayesian.pw_Oi(radii.values, frb[3], dict(max=mx, PDF=pdf))]
        r = radii.sample(n=1, weights=weights).values[0][0]
        theta = 2*np.pi*np.random.random(1)[0]
        ra += r/3600*np.cos(theta)
        dec += r/3600*np.sin(theta)
        return (ra, dec, r)
    else:
        return (ra, dec, (dm_host, dm_igm, dm), path_length, max_dens, (scat_host, scat))
    
def sim_frbs(catalog, n, sigma_frb, cand_info, offset_info):
    '''
    Simulates a number of frbs in a given catalog for true priors
    
    Parameters:
    catalog (df) catalog containing galaxies to sim, needs prior variables
    n (int) number of frbs to sim
    sigma_frb (float) sets error ellipse diameter, optional keyword 'alternate' gives random value [.1, 1]
    cand_info (1 element arr) contains prior keyword (str) possible values shown as functions below
    offset_info (tuple) (maximum size (int), keyword for Aggarwal off_priors (str))
    
    Returns:
    dataframe with parameters of simulated frbs
    
    Example:
    sim_frbs(cat, 10000, 'uniform', ('inverse'), (2., 'uniform')
    '''
    # initializing frb catalog
    frbs = []
    # getting candidates
    cand_prior = cand_info
    # setting the weights given the prior
    if cand_prior == 'inverse':
        weights = 1/chance.driver_sigma(catalog.Rc.values)
    if cand_prior == 'inverse1':
        weights = 1/chance.driver_sigma(catalog.Rc.values)/catalog.diskSize.values
    if cand_prior == 'inverse2':
        weights = 1/chance.driver_sigma(catalog.Rc.values)/catalog.diskSize.values**2
    if cand_prior == 'uniform':
        weights = np.ones(len(catalog))
    if cand_prior == 'sfr':
        weights = catalog.sfr.values
    if cand_prior == 'mass':
        weights = catalog.stellarMass.values
    # applying weights (normalization within)
    frb_cands = catalog.sample(n=n, weights=weights)
    # getting offsets for each candidate
    mx, pdf = offset_info
    for i, r in frb_cands.iterrows():
        # gives the random frb ellipse diameter or the global one
        if sigma_frb=='alternate':
            sigma = .1+.9*np.random.random()
        else:
            sigma = sigma_frb
        if pdf == None:
            ra, dec, dm, path_length, max_dens, scat = pick_off(r, mx)
            frbs.append([ra, dec, sigma, dm[0], dm[1], dm[2], path_length, max_dens, scat[0], scat[1], int(i)])
        else:
            ra, dec, r = pick_off(r, mx, pdf)
            frbs.append([ra, dec, sigma, 'na', 'na', 'na', r, int(i)])
        # loading data into frb catalog
    return pd.DataFrame(frbs, columns=['ra', 'dec', 'radius', 'dm_host', 'dm_igm', 'dm', 'path_length', 'n_max', 'tau_host', 'tau', 'gal_Index'])

