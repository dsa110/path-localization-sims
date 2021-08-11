import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from IPython.display import display
from matplotlib.patches import Ellipse
import math

# path imports
from astropy.coordinates import SkyCoord
from astropath import path
from astropath import localization
from astropath import chance
from astropath import bayesian

def get_candidates(frb_loc, r, true_gal=-1, gal_cat=None):
    '''Helper function for single_path, grabs all galaxies withing a certain radius from a central loc'''
    radius = r/3600
    dec = gal_cat[np.logical_and(gal_cat['dec'] < frb_loc[1]+radius, gal_cat['dec'] > frb_loc[1]-radius)]
    candidates = dec[np.logical_and(dec['ra'] < frb_loc[0]+radius, dec['ra'] > frb_loc[0]-radius)]
    return candidates

def single_path(frb, cand_info, offset_info, search_rad=15, plot=False, gal_cat=None):
    '''
    Runs PATH for a single frb given assumed priors, p_u, and search radius.
    Plotting returns all the candidates colored by their association probability. 
    The orange ring is the frb ellipse, green ring means there was a correct association
    if there is an incorrect association the guessed galaxy will be red and the true will be cyan
    
    Parameters:
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
    candidates = get_candidates((frb_coord.ra.value, frb_coord.dec.value), search_rad, gal_cat=gal_cat)
    Path.init_candidates(candidates.ra.values,
                         candidates.dec.values,
                         candidates.diskSize.values,
                         mag=candidates.Rc.values, 
                         sfr=candidates.sfr.values)

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
            axes.add_patch(plt.Circle((i[0], i[1]), i[3]/3600, facecolor=plt.cm.Blues(i[-1]), alpha=1, edgecolor='k'))
        
        # circle outlines for frbs, true_gal, guessed_gal
        axes.add_patch(plt.Circle((frb[0], frb[1]), frb[2]/3600, fill=False, color='tab:orange', linewidth=2))
        tru = gal_cat[gal_cat.index == frb[-1]].values[0] # getting tru_gal variables
        axes.add_patch(plt.Circle((tru[0], tru[1]), tru[3]/3600, fill=False, edgecolor='tab:cyan', linewidth=2))
        best_index = Path.candidates[Path.candidates.P_Ox == Path.candidates.P_Ox.max()]['gal_Index'].values[0]
        best = gal_cat[gal_cat.index == best_index].values[0] # getting best_gal variables
        if frb[-1]==best_index: 
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

def multiple_path(frbs, cand_info, offset_info, search_rad=15, save=None, plot=False, gal_cat=None):
    '''
    Runs path for an entire catalog of frbs, saves in csv
    
    Parameters:
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
        results = single_path(r, cand_info, offset_info, search_rad=search_rad, plot=False, gal_cat=gal_cat)
        pox = results.P_Ox.values
        true_gal = r[-1]
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