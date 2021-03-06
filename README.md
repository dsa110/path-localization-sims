# path-localization-sims
FRB simulations using MPA galaxy sims and PATH localization tool (SURF project)

## General Usage Info 
`sim_frbs.py` and `path_help.py` are used to simulate frbs in a given catalog and and analyze localization of those frbs using PATH. The catalog currently in use is from the Millinium project the query is included in `path_sims_example.ipynb` which shows the module usage. 

(OUTDATED: needs updating)


`MPApath_help` is a python module with functions designed to simulate FRBs in the sky based on a number of priors and then attempt to identify the true host galaxy using the PATH code (Aggarwal+2021). An example of the usage of the functions is found in the jupyter notebook `MPApath_sims.ipynb`. 

This code requires the PATH package to be installed which can be found [here](https://github.com/FRBs/astropath), and the [paper](https://arxiv.org/abs/2102.10627). The only other neccessary tool to make this run is a galaxy calalog of some kind. The functions take in a pandas DataFrame with the baseline requirements of ra (deg), dec (deg), r-band mag (AB), and apparent disk size (''). Currently there is input for galaxy sfr and total stellar mass as well as more to come. 

First the `sim_frbs()` function must be used to simulate a number of frbs. It takes in a galaxy catalog, number of frbs, error ellipse, and prior info, and then distrutes that number of frbs within that catalog based on the candidate and offset priors. The function returns a DataFrame with frbs locations, ellipse size, and index of the host galaxy. 

The next step is to apply PATH using either the `single_path` or `multiple_path`. The only difference is the latter takes in the entire DataFrame from the output of `sim_frbs` where as the former takes in an array with the values of a single row. Other inputs are the galaxy catalog in which to search for hosts, asumed candidate and offset priors, and a defined search radius in arcsec. `single_path` also contains an optional argument `plot` which if set to `True` will plot all of the possible galaxies as circles based on their angular size. Their color indicates the probability that the FRB is associated with them. The FRB ellipse is shown as an orange ring. If a galaxy has a green outline it was the most likely host and correctly matched the true host. If a galaxy has a red outline it was the most likely host but did not match the true host, which will then have a cyan outline. `multiple_path` also has an argument `save` which can be set to a string with the name of a file which will be places in the `sims` folder and appended with the number of FRBs in the simulation. 

`multiple_path` will also output an DataFrame called stats which can be used to reproduce plots from Aggarwal+2021. The function `analyze` takes in the stats DataFrame and outputs three plots. The first is a normalized histogram for the P(O|x) for each of the candidate galaxies as long as the probability is greater than 1%. The second is a histogram for the max[P(O|x)] of each FRB. The last plot is takes bins 1/10 the size of the FRB array and calculates the fraction correct of each bin and places it on a plot with the x-axis and the max[P(O|x)] with bars showing the min and max in each bin. A 1-1 line is also plotted. The percentage of secure associations, the probabilit that those associaitons are correc, the number of candidates under a probability of 1%, and the total percentage correct are also printed. 

As stated before an example of these functions an be seen in `MPApath_help.iypnb` which also contains a number of other cells for analysis of the galaxy catalog and the candidate galaxies. 

I have also included a folder `sims` which contains the stats files for a number of different simulations that I have already run. These can be imported with the `import_stats` function. 
