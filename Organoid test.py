# %% import packages
import scipy.io
import matplotlib.pyplot as plt
import numpy
import seaborn as sns
import bct

# %% load data
mat_data = scipy.io.loadmat("/imaging/astle/kr01/organoid/OrgNets/C_d153_s6_dt10.mat")
# print keys (headings in mat file)
print(mat_data.keys())
# set default font size for plots
plt.rcParams.update({'font.size': 10})

# %% plot adjM key use redblue colormap centered on 0
# nrows = 2
# ncols = 4
# fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8.3, 4),dpi=300)
# fig.tight_layout()
adjM = mat_data['adjM']
# remove nan values but keep shape
adjM = numpy.nan_to_num(adjM)
# set up figure
plt.figure(figsize=(10, 4))
# plt.subplot(nrows, ncols, 5)
plt.subplot(1, 2, 2)
# plot distribution
plt.hist(adjM[adjM != 0].flatten(), bins=100)
# plot heatmap
# plt.subplot(nrows, ncols, 1)
plt.subplot(1, 2, 1)
sns.heatmap(
    adjM, cmap='RdBu_r', center=0, cbar=True,
)
# %% plot negative-only adjM key use redblue colormap centered on 0
# remove positive values
adjM_neg = adjM.copy()
adjM_neg[adjM_neg > 0] = 0
# set up figure
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 2)
# plot distribution excluding 0s
plt.hist(adjM_neg[adjM_neg != 0].flatten(), bins=100)
# plot heatmap
plt.subplot(1, 2, 1)
sns.heatmap(
    adjM_neg, cmap='RdBu_r', center=0, cbar=True,
)
# %% thresholding 1 proportional threshold only
# apply proportional threshold
# decide on density desired as a proportion
density_desired = 0.1
# calculate percentile threshold
percentile_threshold = (1-density_desired)*100
# get weight value of percentile threshold
threshold = numpy.percentile(adjM, percentile_threshold)
# print "percentile is: 1-density_desired %; threshold is: threshold"
print(f"percentile is: {percentile_threshold} %; threshold is: {threshold}")
# apply threshold
adjM_threshold = adjM.copy()
adjM_threshold[adjM_threshold < threshold] = 0
# set up figure
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 2)
# plot distribution (excluding 0s)
plt.hist(adjM_threshold[adjM_threshold != 0].flatten(), bins=100)
# plot heatmap
plt.subplot(1, 2, 1)
sns.heatmap(
    adjM_threshold, cmap='RdBu_r', center=0, cbar=True,
)
# %% thresholding 2 additional absolute threshold before proportional threshold
# apply absolute threshold of 0.1
threshold_abs = 0.1 # sttc value
adjM_threshold_abs = adjM.copy()
adjM_threshold_abs[adjM_threshold_abs < threshold_abs] = 0
# apply proportional threshold
# decide on density desired as a proportion
density_desired = 0.1
# calculate percentile threshold
percentile_threshold = (1-density_desired)*100
# get weight value of percentile threshold
threshold = numpy.percentile(adjM_threshold_abs, percentile_threshold)
# print "percentile is: 1-density_desired %; threshold is: threshold"
print(f"percentile is: {percentile_threshold} %; threshold is: {threshold}")
# apply threshold
adjM_threshold2 = adjM_threshold_abs.copy()
adjM_threshold2[adjM_threshold2 < threshold] = 0
# set up figure
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 2)
# plot distribution (excluding 0s)
plt.hist(adjM_threshold2[adjM_threshold2 != 0].flatten(), bins=100)
# plot heatmap
plt.subplot(1, 2, 1)
sns.heatmap(
    adjM_threshold2, cmap='RdBu_r', center=0, cbar=True,
)

# %% plot graph metrics
# choose network
adjM_chosen = adjM_threshold2
# plot degree distribution
# calculate degree
degree = numpy.sum(adjM_chosen != 0, axis=0)
# plot degree distribution
nrows = 1
ncols = 4
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8.3, 2),dpi=300)
fig.tight_layout()
# plt.figure(figsize=(8.3, 2),dpi=300)
plt.subplot(nrows, ncols, 2)
# set bins to maximum degree
nbins=max(degree)
sns.histplot(degree, bins=nbins, kde=True, color='black',edgecolor=(0,0,0,0))
plt.xlabel("Degree")
plt.ylabel(" ")
plt.title(" ")
# plot edge weight distribution
plt.subplot(nrows, ncols, 1)
sns.histplot(adjM_chosen[adjM_chosen != 0].flatten(), bins=nbins, kde=True, color='black',edgecolor=(0,0,0,0))
plt.xlabel("Edge weight")
plt.ylabel("Frequency")
plt.title(" ")
# plot clustering coefficient
# calculate clustering coefficient using bctpy
clustering = bct.clustering_coef_bu(adjM_chosen)
# plot clustering coefficient distribution
plt.subplot(nrows, ncols, 3)
sns.histplot(clustering,bins=nbins, kde=True, color='black',edgecolor=(0,0,0,0))
plt.xlabel("Clustering Coefficient")
plt.ylabel(" ")
plt.title(" ")
# plot betweenness centrality
# calculate betweenness centrality using bctpy
betweenness = bct.betweenness_bin(adjM_chosen)
# plot betweenness centrality distribution
plt.subplot(nrows, ncols, 4)
sns.histplot(betweenness, bins=nbins, kde=True, color='black',edgecolor=(0,0,0,0))
plt.xlabel("Betweenness Centrality")
plt.ylabel(" ")
plt.title(" ")
# %% modularity with module no. sorted by module size
# note community_louvain is probabilistic so results may vary
# calculate modularity using bctpy
# get module affiliation
# ci = bct.modularity_und(adjM_chosen)
# plot modularity using community_louvain
modularity = bct.community_louvain(adjM_chosen)
# get module sizes
module_sizes = numpy.bincount(modularity[0])
# sort modules by size
sorted_modules = numpy.argsort(module_sizes)[::-1]
# sort module affiliation
sorted_modularity = numpy.zeros(modularity[0].shape)
for i in range(len(sorted_modules)):
    sorted_modularity[modularity[0] == sorted_modules[i]] = i
plt.figure()
# set bin width to 1
binwidth = 1
plt.hist(sorted_modularity, bins=numpy.arange(min(sorted_modularity), max(sorted_modularity) + binwidth, binwidth),edgecolor=(0,0,0,1))
plt.xlabel("Module #")
plt.ylabel("No. nodes")
plt.title("Module sizes")
# align ticks with bin centers
plt.xticks(ticks=numpy.arange(min(sorted_modularity) +.5, max(sorted_modularity) + .5, 1),
           labels=numpy.arange(min(sorted_modularity), max(sorted_modularity)).astype(int)+1)

# %% other plot ideas
# plot module degree (nedges rather than nnodes on y axis) and plot in order of nedges in module
# plot un/thresholded network graph or modules or other graph metrics
# plot how graph metrics change as a proportion of absolute threshold and/or proportional threshold
# e.g. could plot density curve for each threshold value on the same plot with darker shade of blue for increasing proportion threshold
# (plot kde of each metric on same plot as a function of threshold percentiles 1 % to 100 % in 1 % increments)
