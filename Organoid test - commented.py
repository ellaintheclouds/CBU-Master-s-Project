# C:\Users\er05\AppData\Local\anaconda3\Scripts\activate 
# PATH=C:\Users\er05\AppData\Local\anaconda3\Scripts;%PATH%

# Import packages ----------
import os # for interacting with the operating system
import scipy.io # for loading .mat files (MATLAB data)
import matplotlib.pyplot as plt # for plotting - the "as plt part allows us to refer to the package as plt"
import numpy # for numerical operations (e.g. matrix manipulation)
import seaborn as sns # for heatmaps and enhanced data visualisation
import bct # for graph-theoretic analysis (from the brain connectivity toolbox)

# Load data ----------
# Check current working directory
print("Current working directory:", os.getcwd())

# Change working directory to the location of the .mat file
os.chdir("C:/Users/El Richardson/OneDrive/Documents/Personal Documenets/Education/Postgraduate/BTN/Project/Code/Test scripts/Test data")

# Load data
mat_data = scipy.io.loadmat("M_d28_s1_dt10") # .mat file containing adjacency matrix is loaded

# print keys (headings in mat file)
print(mat_data.keys()) # helps to see what it contains

# Processing ----------
# # set default font size for plots
plt.rcParams.update({'font.size': 6})# Plot adjM key use redblue colormap centered on 0

# create a grid of subplots and set up a figure layout
#nrows = 2
#ncols = 4
#fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8.3, 4),dpi=300)
#fig.tight_layout()

adjM = mat_data['adjM'] # a 2D matrix where each cell represents a connection between two nodes (brain regions)

# remove NaN values (replacing with zeros, to avoid issues in calculations) but keep shape
adjM = numpy.nan_to_num(adjM)

# Finding and replacing erroneous values in adjM ----------
# Filter values in adjM to a specific range around 0.5
#lower_bound = 0.5
#upper_bound = 0.5001
#filtered_values = adjM[(adjM > lower_bound) & (adjM < upper_bound)]
#
# Set up figure
#plt.figure(figsize=(10, 4))
#
#
# Plot histogram of filtered values
#plt.hist(filtered_values.flatten(), bins=100)
#plt.title('Histogram of adjM values between 0.5 and 0.5001')
#plt.xlabel('Value')
#plt.ylabel('Frequency')
#
# Show the plot
#plt.show()
#
#from collections import Counter
#
# Flatten the filtered_values array and convert it to a list
#filtered_values_list = filtered_values.flatten().tolist()
#
# use Counter to count the frequency of each value
#counter = Counter(filtered_values_list)
#
# find the most common values
#most_common_value, count = counter.most_common(1)[0]
#print(f"The most frequently occurring value is {most_common_value} with {count} occurrences.")
#second_most_common_value, second_count = counter.most_common(2)[1]
#print(f"The second most frequently occurring value is {second_most_common_value} with {second_count} occurrences.")
#
# exclude overly repeated values
#adjM[(adjM == most_common_value) | (adjM == second_most_common_value)] = 0

# Plot adjacency matrix ----------
# set up figure
plt.figure(figsize=(10, 4))
# this creates a new figure where the plots will be drawn
# the figsize argument specifies the size of the figure in inches (width, height)
# at this stage, the figure is empty

# first subplot: plot distribution
plt.subplot(1, 2, 2)
# creates a grid of subplots with 1 row and 2 columns, and selects the second subplot
# after calling plt.subplot(), anything you plot will go into this specific subplot
plt.hist(adjM[adjM != 0].flatten(), bins=100)  # shows the distribution of connection weights (non-zero values)
# plots a histogram of the non-zero values of adjM
# zeros are filteres out to exclude unused connections or self loops in the adjacency matrix
# .flatten() converts the 2D matrix into a 1D array so that the values can be used in a histogram
# at this stage, the second subplot in the figure contains a histogram showing the distribution of the non-zero values of the adjacency matrix
plt.title("Distribution of Connection Weights")

# second subplot: plot heatmap
plt.subplot(1, 2, 1)
# creates another subplot in the same 1-row, 2-column grid, but this time selects the first subplot
sns.heatmap(adjM, cmap='RdBu_r', center=0, cbar=True) # plots the adjacency matrix as a heatmap (red = positive, blue = negative, white = 0)
# plots a heatmap of the adjacency matrix
# cmap='RdBu_r' specifies the colormap to use (red positive, blue negative, white zero)
# centre=0 ensures that 0 is the midpoint of the colour scale
# cbar=True adds a colour bar to the side of the heatmap
# at this stage, the first subplot in the figure contains a heatmap of the adjacency matrix, showing the connection weights as colours
plt.title("Plot 1: Adjacency Matrix Heatmap")

plt.show()

# Plot negative-only adjM key use redblue colormap centered on 0 ----------
# remove positive values - only negative connections are retained by setting positive values to 0
adjM_neg = adjM.copy()
adjM_neg[adjM_neg > 0] = 0

# set up figure
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 2)

# first subplot: plot distribution
plt.hist(adjM_neg[adjM_neg != 0].flatten(), bins=100)
plt.title("Distribution of Negative Connection Weights")

# second subplot: plot heatmap
plt.subplot(1, 2, 1)
sns.heatmap(adjM_neg, cmap='RdBu_r', center=0, cbar=True,)
plt.title("Plot 2: Negative-Only Adjacency Matrix Heatmap")

plt.show()

# Thresholding 1 proportional threshold only ----------
# apply proportional threshold
# decide on density desired as a proportion
density_desired = 0.1

# calculate percentile threshold
# thresholding removes weaker connections to focus on the stronger ones
percentile_threshold = (1-density_desired)*100

# get weight value of percentile threshold
# a threshold value is computed based on the percentile of weights in the adjacency matrix
threshold = numpy.percentile(adjM, percentile_threshold)

# print "percentile is: 1-density_desired %; threshold is: threshold"
print(f"percentile is: {percentile_threshold} %; threshold is: {threshold}")

# apply threshold
adjM_threshold = adjM.copy()
adjM_threshold[adjM_threshold < threshold] = 0 # values below the threshold are set to 0

# set up figure
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 2)

# first subplot: plot distribution
plt.hist(adjM_threshold[adjM_threshold != 0].flatten(), bins=100)
plt.title("Distribution of Connection Weights (Thresholded)")

# second subplot: plot heatmap
plt.subplot(1, 2, 1)
sns.heatmap(adjM_threshold, cmap='RdBu_r', center=0, cbar=True,)
plt.title("Plot 3: Thresholded Adjacency Matrix Heatmap")

plt.show()

# Thresholding 2 additional absolute threshold before proportional threshold----------
# apply absolute threshold of 0.1 before proportional thresholding to ensure connections below an absolute value are removed first
threshold_abs = 0.1 # spike time tilling coefficient (STTC) value - an absolute threshold
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
plt.title("Distribution of Connection Weights (Thresholded)")

# plot heatmap
plt.subplot(1, 2, 1)
sns.heatmap(adjM_threshold2, cmap='RdBu_r', center=0, cbar=True,)
plt.title("Plot 4: Absolute and Proportional Thresholded adjM Heatmap")

plt.show()

# Plot graph metrics ----------
# choose network
adjM_chosen = adjM_threshold2

# plot degree distribution
# calculate degree
degree = numpy.sum(adjM_chosen != 0, axis=0) # the number of connections for each node
# adjM_chosen != 0 returns a boolean matrix where True indicates a connection
# numpy.sum() sums the number of connections for each node
# axis=0 specifies that the sum should be calculated along the rows (i.e., for each node)

# In the context of a graph represented by an adjacenct matrix, the degree of a node is the number of connections (edges) it has
# By summing the boolean matrix along the columns, you count the number of connections for each node
# This, overall, allows you to calculate the degree of each node in the graph

# Set up figure 4: plot of degree distribution
nrows = 1
ncols = 4
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8.3, 2),dpi=300)
fig.tight_layout()

# Subplot 1: plot edge weight distribution to visualise how connected the nodes are
# plt.figure(figsize=(8.3, 2),dpi=300)
plt.subplot(nrows, ncols, 2)

# set bins to maximum degree
nbins=max(degree)
sns.histplot(degree, bins=nbins, kde=True, color='black',edgecolor=(0,0,0,0))
plt.xlabel("Degree")
plt.ylabel(" ")
plt.title(" ")

# Subplot 2: plot edge weight distribution to visualise how connected the nodes are
plt.subplot(nrows, ncols, 1)
sns.histplot(adjM_chosen[adjM_chosen != 0].flatten(), bins=nbins, kde=True, color='black',edgecolor=(0,0,0,0))
plt.xlabel("Edge weight")
plt.ylabel("Frequency")
plt.title(" ")

# Subplot 3: clustering coefficient
# calculate clustering coefficient using bctpy
clustering = bct.clustering_coef_bu(adjM_chosen) # measures the degree to which nodes in a graph cluster together

# plot clustering coefficient distribution
plt.subplot(nrows, ncols, 3)
sns.histplot(clustering,bins=nbins, kde=True, color='black',edgecolor=(0,0,0,0))
plt.xlabel("Clustering Coefficient")
plt.ylabel(" ")
plt.title(" ")

# Subplot 4: betweenness centrality
# calculate betweenness centrality using bctpy
betweenness = bct.betweenness_bin(adjM_chosen) # quantifies how often a node acts as a bridge in the shortest path between two other nodes

# plot betweenness centrality distribution
plt.subplot(nrows, ncols, 4)
sns.histplot(betweenness, bins=nbins, kde=True, color='black',edgecolor=(0,0,0,0))
plt.xlabel("Betweenness Centrality")
plt.ylabel(" ")
plt.title(" ")

plt.show()
##########################################################################################################################
# Modularity with module no. sorted by module size ----------
# calculate modularity and community structure
#ci = bct.modularity_und(adjM_chosen) this is not working
# this returns the modularity index and the community affiliation vector
# ci stores the community affiliation vector which indicates the community to which each node belongs

# plot modularity using community_louvain (note community_louvain is probabilistic so results may vary)
modularity = bct.community_louvain(adjM_chosen) # modularity detects communities (groups of nodes)
# community_louvain is a method to detect communities (groups of strongly interconnected nodes) within a network, using the Louvain algorithm
# this is probabilistic - results can vary between runs because the algorithm may find different local optima
# modularity[0] returns a community affiliation vector, which assigns each node to a community
# modularity[1] returns the modularity index, which quantifies the division into communities

# Calculating and sorting module sizes -
# get module sizes
module_sizes = numpy.bincount(modularity[0])
# counts the number of nodes in each community (module)

# sort modules by size
sorted_modules = numpy.argsort(module_sizes)[::-1]
# sorts the module indices by their sizes in descending order

# Re-sorting community affiliations -
# sort module affiliation
sorted_modularity = numpy.zeros(modularity[0].shape)
for i in range(len(sorted_modules)):
    sorted_modularity[modularity[0] == sorted_modules[i]] = i
# the community affiliation vector is updated so that the largest community is assigned module number 0, and so on
# sorted_modules[i] gives the index of community sorted by size
# modularity[0] == sorted_modules[i] finds all nodese belonging to the ith largest community
# sorted_modularity[...] = i assigns the new module number i to those nodes

# Plotting module sizes-
# plotting a histogram of module sizes (number of nodes in each module)
plt.figure()

# set bin width to 1
binwidth = 1
plt.hist(
    sorted_modularity, 
    bins=numpy.arange( 
        min(sorted_modularity), 
        max(sorted_modularity) + binwidth, 
        binwidth
        ),
        edgecolor=(0,0,0,1)
        )
# bins() argument ensures that each module gets its own histogram
# creates bin edges from the smallest to the largest module, with a width of 1

# add labels and title
plt.xlabel("Module #")
plt.ylabel("No. nodes")
plt.title("Module sizes")

# align ticks with bin centers
plt.xticks(ticks=numpy.arange(min(sorted_modularity) +.5, max(sorted_modularity) + .5, 1),
           labels=numpy.arange(min(sorted_modularity), max(sorted_modularity)).astype(int)+1)
# module numbers are incremented by 1 to ensure that they start from 1 instead of 0

plt.show()

# Other plot ideas ----------
# plot module degree (nedges rather than nnodes on y axis) and plot in order of nedges in module

# plot un/thresholded network graph or modules or other graph metrics

# plot how graph metrics change as a proportion of absolute threshold and/or proportional threshold
# e.g. could plot density curve for each threshold value on the same plot with darker shade of blue for increasing proportion threshold

# (plot kde of each metric on same plot as a function of threshold percentiles 1 % to 100 % in 1 % increments)