###############################################################################################################
Folder
###############################################################################################################
data: contains the Karate club toy dataset, three benmark dataset for semi-supervised learning: Cora, Citeseer
and Pubmed, and a large network dataset:GSN

images: a folder of genrated visualzation plot

video: two videos of showing how GCN model improves during the traning process (GCN model applied on the toy dataset) 

###############################################################################################################
Python Scirpts
###############################################################################################################
models.py: It contains the GCN, GCN with residual connection and GAT models 

spectrum_embedding.py: It contains the code to run spectrum embedding model under a supervised task

tsne.py: It contains the code to run tsne model under a supservised task
*Note for t-sne we use the off the shelf implementation from scikit-learn.*

train_utlis:utlis class for training neural model including metrics and evaltion script

plot_utlis:utlis class for plotting graph

plot.py: a script to plot results for showing impact of nerual model depth and contain 
a plot function for wall clock time analysis





###############################################################################################################
Notebooks
###############################################################################################################
visualization.ipynb : a notebook contains the code to genarate visualizations for qualitative analysis.


###############################################################################################################
Model.ipynb : a notebook contains the code to compare model perforances quantatively and contains code segment 
that demnonstartes how to run the the implemented models. The notebook is to analyze semi-supervised perforamnce of 
the implemented graph nerual model.
