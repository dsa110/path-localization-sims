import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from IPython.display import display
# path imports
from astropy.coordinates import SkyCoord

from astropath import path
from astropath import localization
from astropath import chance
from astropath import bayesian



def pick_off(frb, mx, pdf):
    '''Helper function for sim_frbs, picks offset from a central loc. Uses priors from PATH bayesian.py'''
    ra = frb[0]
    dec = frb[1]
    radii = pd.DataFrame(np.linspace(0., mx, 1000))
    ## only applies same offset priors as aggarwal
    weights = [i[0] for i in bayesian.pw_Oi(radii.values, frb[3], dict(max=mx, PDF=pdf))]
    r = radii.sample(n=1, weights=weights).values[0][0]
    theta = 2*np.pi*np.random.random(1)[0]
    ra += r/3600*np.cos(theta)
    dec += r/3600*np.sin(theta)
    return (ra, dec, r)

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
        ra, dec, off = pick_off(r, mx, pdf)
        # gives the random frb ellipse diameter or the global one
        if sigma_frb=='alternate':
            sigma = .1+.9*np.random.random()
        else:
            sigma = sigma_frb
        # loading data into frb catalog
        frbs.append([ra, dec, sigma, off, int(i)])
    return pd.DataFrame(frbs, columns=['ra', 'dec', 'radius', 'offset','gal_Index'])



def get_candidates(gal_cat, frb_loc, r):
    '''Helper function for single_path, grabs all galaxies withing a certain radius from a central loc'''
    radius = r/3600
    dec = gal_cat[np.logical_and(gal_cat['dec'] < frb_loc[1]+radius, gal_cat['dec'] > frb_loc[1]-radius)]
    candidates = dec[np.logical_and(dec['ra'] < frb_loc[0]+radius, dec['ra'] > frb_loc[0]-radius)]
    return candidates

def single_path(gal_cat, frb, cand_info, offset_info, search_rad=15, plot=False):
    '''
    Runs PATH for a single frb given assumed priors, p_u, and search radius.
    Plotting returns all the candidates colored by their association probability. 
    The orange ring is the frb ellipse, green ring means there was a correct association
    if there is an incorrect association the guessed galaxy will be red and the true will be cyan
    
    Parameters:
    gal_cat (df) galaxy catalog to search through for candidates
    frb (arr) containing the values of a single row from the output of sim_frbs
    cand_info (tuple) (unknown probability (float), keyword for Aggarwal cand_priors)
    offset_info (tuple) (maximum size (int), keyword for Aggarwal off_priors (str))
    search_rad (int) radius to search around frb in arcsec
    plot (boolean) True plots
    Returns:
    dataframe of candidates, and thier association probabilities
    Example:
    candidates = single_path(frbs.iloc[22], (0., 'inverse'), (6, 'exp'), search_rad=7, plot=True)
    '''
    Path = path.PATH()
    # init frb
    frb_coord = SkyCoord(frb[0], frb[1], unit='deg')
    eellipse = dict(a=frb[2], b=frb[2], theta=0.)
    Path.init_localization('eellipse', center_coord=frb_coord, eellipse=eellipse)
    
    # init candidates
    candidates = get_candidates(gal_cat, (frb_coord.ra.value, frb_coord.dec.value), search_rad)
    Path.init_candidates(candidates.ra.values,
                         candidates.dec.values,
                         candidates.diskSize.values,
                         mag=candidates.Rc.values)
    
    # init priors
    P_u, cand_pdf = cand_info
    mx, off_pdf = offset_info
    Path.init_cand_prior(cand_pdf, P_U=P_u)
    Path.init_theta_prior(off_pdf, mx)
    # calculating
    Path.calc_priors()
    P_Ox, P_Ux = Path.calc_posteriors('fixed', box_hwidth=30., max_radius=30)
    # adding true galaxy index to results df
    Path.candidates['gal_Index'] = candidates.index
    
    # adding probabilities to candidates for easier plotting
    candidates['pOx'] = Path.candidates['P_Ox'].values
    if plot:
        figure, axes = plt.subplots(figsize=(10, 10))
        display(candidates[candidates['pOx']>.05])
        
        # plotting galaxies based on probability
        for i in candidates.sort_values('diskSize', ascending=False).values:
            axes.add_patch(plt.Circle((i[0], i[1]), i[3]/3600, facecolor=plt.cm.Blues(i[7]), alpha=1, edgecolor='k'))
        
        # circle outlines for frbs, true_gal, guessed_gal
        axes.add_patch(plt.Circle((frb[0], frb[1]), frb[2]/3600, fill=False, color='tab:orange', linewidth=2))
        tru = gal_cat[gal_cat.index == frb[4]].values[0] # getting tru_gal variables
        axes.add_patch(plt.Circle((tru[0], tru[1]), tru[3]/3600, fill=False, edgecolor='tab:cyan', linewidth=2))
        best_index = Path.candidates[Path.candidates.P_Ox == Path.candidates.P_Ox.max()]['gal_Index'].values[0]
        best = gal_cat[gal_cat.index == best_index].values[0] # getting best_gal variables
        if frb[4]==best_index: 
            axes.add_patch(plt.Circle((best[0], best[1]), best[3]/3600, fill=False, edgecolor='tab:green', linewidth=2))
        else:
            axes.add_patch(plt.Circle((best[0], best[1]), best[3]/3600, fill=False, edgecolor='tab:red', linewidth=2))
        # making color map
        colors = candidates.pOx.values
        colors[-1] = 1.
        plt.scatter(candidates.ra.values, candidates.dec.values, c=colors, alpha=1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar()
    return Path.candidates



def multiple_path(gal_cat, frbs, cand_info, offset_info, search_rad=15, save=None, plot=False):
    '''
    Runs path for an entire catalog of frbs, saves in csv
    
    Parameters:
    gal_cat (df) galaxy catalog to search through for candidates
    frbs (arr) output of sim_frbs
    cand_info (tuple) (unknown probability (float), keyword for Aggarwal cand_priors)
    offset_info (tuple) (maximum size (int), keyword for Aggarwal off_priors (str))
    search_rad (int) radius to search around frb in arcsec
    save (str) filename which will be appended with the length of the input frb cat
    plot (boolean) True plots
    Returns:
    dataframe of important statistics for analysis
    Example:
    multiple_path(frbs, (0.05, 'inverse'), (6, 'exp'), search_rad=7, save='inverse', gal_cat=galaxies)
    '''
    stats = []
    count = 0
    for i, r in frbs.iterrows():
        results = single_path(gal_cat, r, cand_info, offset_info, search_rad=search_rad, plot=False)
        pox = results.P_Ox.values
        true_gal = r[4]
        best_gal = results[results.P_Ox == results.P_Ox.max()]['gal_Index'].values[0]
        stats.append([pox[pox > .01], max(pox), best_gal==true_gal, true_gal, len(results)])
        count += 1
        if count%500==0:
            print('{} '.format(count), end='')
    stat = pd.DataFrame(stats, columns=['all_pOx', 'max_pOx', 'correct', 'gal_Index', 'num_cand'])
    if save != None:
        stat.to_csv('./sims/{0}_{1}.csv'.format(save, len(stat)), header=True, index=False)
    return stat

def import_stats(name):
    '''imports the output of multiple_path. This is not the best way to do this and liklely has errors
    Should work at least until 10000 frbs though'''
    s = pd.read_csv('sims/{}.csv'.format(name))
    new = [ast.literal_eval(i.replace('\n', '').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace(' ', ', ')) for i in s.all_pOx.values]
    #s.drop('all_pOx')
    s['all_pOx'] = new
    return s


def analyze(stats):
    '''
    Runs analysis to recreate figures 5 and 6 from Aggarwal+2021 also gives percentage secure and percentage correct
    
    Parameters:
    stats (df) output of multiple_frbs or import_stats
    Returns:
    P(O|x) histogram data, max[P(O|x)] histogram data, fig6 data, 
    Example:
    multiple_path(frbs, (0.05, 'inverse'), (6, 'exp'), search_rad=7, save='inverse', gal_cat=galaxies), (percentage_secure, TP, P01, per_corr)
    '''
    plt.figure()
    a = plt.hist(np.concatenate(stats.all_pOx.values), 50, density=True)
    plt.ylabel('PDF')
    plt.xlabel('$P(O_i|x)$')
    plt.figure()
    b = plt.hist(stats.max_pOx, 50, density=True)
    plt.ylabel('PDF')
    plt.xlabel('max[$P(O_i|x)$]')

    correct = [(stats.max_pOx.values[i], stats.correct.values[i])for i in range(len(stats))]
    correct.sort()
    n = int(len(correct)/10)
    chunks = [correct[i:i + n] for i in range(0, len(correct), n)]
    
    max_poxs = []
    percentage = []
    mins = []
    maxs = []
    for i in chunks:
        maxes = [j[0] for j in i]
        mn = np.mean(maxes)
        max_poxs.append(mn)
        mins.append(mn-min(maxes))
        maxs.append(max(maxes)-mn)
        tfs = [j[1] for j in i]
        percentage.append(sum(tfs)/len(tfs))
    minmax = [mins, maxs]
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', zorder=1)
    plt.scatter(max_poxs, percentage, zorder=10, s=10)
    plt.errorbar(max_poxs, percentage, xerr=minmax, capsize=3, linestyle='', zorder=10)
    plt.xlabel('max[$P(O_i|x)$]')
    plt.ylabel('Fraction correct');

    per_corr = sum([i[1] for i in correct])/len(correct)

    secure = np.array(stats.max_pOx.values)
    percentage_secure = len(secure[secure>.95])/len(stats)
    print('f(T+secure): {0:.2f}'.format(percentage_secure))
    tp = [i[1] for i in correct if i[0] > .95]
    try:
        TP = sum(tp)/len(tp)
        print('TP: {0:.2f}'.format(TP))
    except ZeroDivisionError:
        print('TP: N/A')
    zero = [len(i) for i in stats.all_pOx.values] / stats.num_cand.values
    P01 = 1-np.mean(zero)
    print('p<.01: {0:.2f}'.format(P01))
    print('percentage correct: {0:.2f}'.format(per_corr))
    return a, b, (max_poxs, percentage, minmax), (percentage_secure, TP, P01, per_corr)