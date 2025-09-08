# Load dataset
data("iris")

# Remove the species column (unsupervied = no labels) and scale the data
iris_data <- scale(iris[, 1:4])

# Preview data
head(iris_data)



#-----------------------------
# kmean() K-Means Clutering
#-----------------------------


# Set seed for reproducibility
set.seed(79)

# Apply K-Means Clustering with 3 clusters
km <- kmeans(iris_data, centers = 3, nstart = 25)

# View cluster assignments
km$cluster

# Add cluster to iris dataset
iris$km_cluster <- as.factor(km$cluster)

# Visualize with ggplot2
library(ggplot2)
ggplot(iris, aes(Sepal.Length, Sepal.Width, color = km_cluster)) +
  geom_point() +
  labs(title = "K-Means Clustering on Iris Data") + 
  theme_minimal()


# How to chose the number of clusters (Elbow method)
if(!require(factoextra)) install.packages("factoextra") 
library(factoextra)

# Plot tot.withinss for k = 1 to k = 10 by default
fviz_nbclust(iris_data, kmeans, method = "wss") +
  labs(subtitle = "Elbow Method for Optimal Number of Clusters")


# How to chose the number of clusters (Silhouette Width method)
# Choose the number of clusters that maximizes the average silhouette width
fviz_nbclust(iris_data, kmeans, method = "silhouette") +
  labs(subtitle = "Silouette Method for Optimal Number of Clusters")


# How to chose the number of clusters (Gap Statistic method)
# Look for biggest gap
# Gap statistic calculation (Take a bit longer to run)
set.seed(79)
fviz_nbclust(iris_data, kmeans, method = "gap_stat", nstart = 25, nboot = 50) +
  labs(subtitle = "Gap Statistic Method for Optimal Number of Clusters")


# Visualized clusters using factoextra::fviz_cluster()
# This plot the first two principal components by default and color point by cluster
fviz_cluster(km, data = iris_data) + 
  labs(title = "K-Means Clustering Visulaization")



#------------------------------------
# hclust() Hierarchical Clustering
#------------------------------------


# Compute distance metrix
dist_matrix <- dist(iris_data)

# Apply hierarchical clustering
hc <- hclust(dist_matrix)

# Plot dendrogram
plot(hc, main = "Hierarchical Clustering Dendrogram on Iris Data")
abline(h = 5, col = "salmon", lty = 2)

# Cut the tree into 3 clusters or specify h. cutree(hc, h = 5)
hc_clusters <- cutree(hc, k = 3)

# Add to Iris Dataset
iris$hc_cluster <- as.factor(hc_clusters)

# Plot clusters
ggplot(iris, aes(Sepal.Length, Sepal.Width, color = hc_cluster)) +
  geom_point() +
  labs(title = "Heirarchical Clustring on Iris Data") +
  theme_minimal()


# Visual dendrogram with clusters highlighted (cut into 3 clusters)
fviz_dend(hc, k = 3,
          cex = 0.5, # label size
          k_colors = c("red", "green", "blue"), # cluster colors
          color_labels_by_k = TRUE,
          rect = TRUE # add rectangles around clusters
          )



#-----------------------------------------------
# prcomp() Principal Component Analysis (PCA)
#-----------------------------------------------


# Run PCA on standardized data
pca <- prcomp(iris[, 1:4], scale. = TRUE)

# Summary of variance explained
summary(pca)

# PCA loading, how original variables contribute to PCs
pca$rotation

# PCA scores, new coordinates for each data point
iris_pca <- as.data.frame(pca$x)

# Plot PCA components colored by hierarchical clusters
iris_pca$hc_cluster <- as.factor(hc_clusters)
ggplot(iris_pca, aes(PC1, PC2, color = hc_cluster)) +
  geom_point(alpha = 0.5) +
  labs(title = "Heirarchical Clustering Visualization on PCA Components") +
  theme_minimal()

# Scree plot, show how much variance each PC explains
fviz_eig(pca, addlabels = TRUE, ylim = c(0, 80))

# Biplot, scores and loading in one plot (observations and original variables) 
# Arrows show variable contributions
fviz_pca_biplot(pca,
                label = "var", # Only label the variables
                habillage = iris$Species, # Color points by species
                addEllipses = TRUE)

# Or PCA plot (2D projection of data) or using gglot2
# Personally, I prefer biplot
fviz_pca_ind(pca,
             geom.ind = "point",
             col.ind = iris$Species,
             palette = TRUE, # use default color palette
             legend.title = "Species")



#-------------------------------
# dist() Distance Calculation
#-------------------------------


# Calculate Euclidean distance between observations
dist_matrix <- dist(iris_data)

# Convert to matrix to review
distance_matrix_matrix <- as.matrix(dist_matrix)

# View top left corner of matrix
round(distance_matrix_matrix[1:5, 1:5], 2)

# Calculate Manhattan distance, more robust to outliers
dist_matrix <- dist(iris_data, method = "manhattan")



#----------------------------------------------------------------------
# PCA + K-Means Clustering, often better than raw features + K-Means
#----------------------------------------------------------------------


# We should use PCA before clustering for high dimensional and noisy data
# Use only first two PCs for clustering
pca_scores <- pca$x[, 1:2]

# Run kmeans on PCA reduced data
km_pca <- kmeans(pca_scores, centers = 3, nstart = 25)

# Visualized clusters
pca_df <- as.data.frame(pca_scores)
pca_df$km_cluster <- as.factor(km_pca$cluster)

ggplot(pca_df, aes(PC1, PC2, color = km_cluster)) +
  geom_point(alpha = 0.5) +
  labs(title = "K-Means Clustering on PCA-Reduced Data") +
  theme_minimal()

# Add actual species to see how well clustering match true classes
pca_df$Species <- iris$Species

ggplot(pca_df, aes(PC1, PC2, color = km_cluster, shape = Species)) +
  geom_point(size = 3, alpha = 0.5) +
  labs(title = "K-Means Clustering on PCA-Reduced Data  vs. True Species") +
  theme_minimal()

# Compare table
table(pca_df$km_cluster, iris$Species)



#---------------------------------
# PCA + Hierarchical Clustering
#---------------------------------


# Compute distance matrix
dist_matrix <- dist(pca_scores)

# Run hierarchical clusting on PCA reduced data
# method = "complete", uses the maximum distance between cluster points
hc_pca <- hclust(dist_matrix, method = "complete") 

# Plot dendrogram
plot(hc_pca, main = "Dendrogram on PCA-Reduced Data")

# Cut the tree into clusters
hc_cluster <- cutree(hc_pca, k = 3)

# Add to PCA data
pca_df$hc_cluster <- as.factor(hc_cluster) 

# Visualize hierarchical clusters in PCA space
ggplot(pca_df, aes(PC1, PC2, color = hc_cluster)) +
  geom_point(alpha = 0.5) +
  labs(title = "Hierarchical Clustering on PCA-Reduce Data") +
  theme_minimal()

# Compare to actual species
ggplot(pca_df, aes(PC1, PC2, color = hc_cluster, shape = Species)) +
  geom_point(size = 3, alpha = 0.5) +
  labs(title = "Hierarchical Clustering on PCA-Reduced Data vs. True Species") +
  theme_minimal()

# Compare table
table(pca_df$hc_cluster, iris$Species)



#--------------------------------------
# t-SNE and how to implement it in R
#--------------------------------------


# t-SNE t-distributed Stochastic Neighbor Embedding
# t-SNE is a nonlinear dimensional reduction technique, local structure only
# t-SNE expects a numeric matrix without missing value and no duplicates
# scale data first iris_data <- scale(iris[, 1:4])

# Remove duplicates
iris_unique <- unique(iris_data)

iris_labels <- iris$Species

# Run t-SNE
library(Rtsne)
set.seed(79)
tsne <- Rtsne(
  iris_unique, # scaled and unique
  dims = 2, # reduce to 2 dimensions
  perplexity = 30, # balances attention between local and global aspects (range 5-50)
  verbose = TRUE # show process
)


# Cluster the tsne-reduced data using hierarchical clustering
# Compute distance metrix
dist_matrix <- dist(tsne$Y)

# Perform hierarchical clustering
# ward.D2 works well with Euclidean distances for compact clusters
hc_tsne <- hclust(dist_matrix, method = "ward.D2")

# Cut the dendrogram tree into 3 clusters
hc_cluster <- cutree(hc_tsne, k = 3)

# Visualize the clusters on the t-SNE plot
tsne_df <- data.frame(
  x = tsne$Y[, 1],
  y = tsne$Y[, 2],
  hc_cluster = factor(hc_cluster)
  )

ggplot(tsne_df, aes(x, y, color = hc_cluster)) +
  geom_point(alpha = 0.5) +
  labs(title = "Hierarchical Clustering on tSNE-Reduce Data") +
  theme_minimal()

# Compare with actual species
# Note that we unique the iris_data for Rtsne()
table(tsne_df$hc_cluster, iris$Species[!duplicated(iris_data)])



#-------------------------------------
# UMAP and how to implement it in R
#-------------------------------------

# UMAP Uniform Manifold Approximation and Projection
# Balance between structure (local and global), speed, and clustering

if(!require(umap)) install.packages("umap")
library(umap)
set.seed(79)
umap <- umap(iris_data) # return a matrix of 2D coordinates (2 columns)

# Cluster the umap-reduced data using k-means clustering
# Automatically determine the optimal number of clusters using NbClust package
if(!require(NbClust)) install.packages("NbClust")
library(NbClust)

set.seed(79)
nb <- NbClust(data = umap$layout,
              distance = "euclidean",
              min.nc = 2, # try number of clusters from 2 to 10
              max.nc = 10,
              method = "kmeans"
              )

# Check the best number of clusters
best_k <- nb$Best.nc[1] # Majority rule result
print(paste("Best number of clusters according to Nbclust:", best_k))

# Apply k-means with optimal k
set.seed(79)
km_umap <- kmeans(umap$layout, centers = best_k, nstart = 25)
km_cluster <- km_umap$cluster

# Visualize the result
umap_df <- data.frame(
  x = umap$layout[, 1],
  y = umap$layout[, 2],
  km_cluster = factor(km_cluster)
)

ggplot(umap_df, aes(x, y, color = km_cluster)) +
  geom_point(alpha = 0.5) +
  labs(title = paste("K-Means Clustering on UMAP-Reduced Data, k =",best_k), 
       x = "UMAP 1", 
       y = "UMAP 2") +
  theme_minimal()

# Compare to actual species
table(umap_df$km_cluster, iris_labels)



#------------------------------------
# DBSCAN and how implement it in R
#------------------------------------


# DBSCAN Density-Based Spatial Clustering of Applications with Noise
# DBSCAN is a clustering algorithm that groups together points that are densely packed, 
# and labels low-density points as outliers.

if(!require(dbscan)) install.packages("dbscan")
library(dbscan)

# Run DBSCAN
db <- dbscan(iris_data, # scaled data
             eps = 0.5, # The radius (epsilon neighborhood) around a point to consider it a neighbor
             minPts = 5 # Minimum number of points required to form a dense region (core point)
             )

# View results
db$cluster # Cluster labels, 0 = noise

# Chose mintPts properly
minPts = ncol(iris_data) + 1 # or slightly higher

# Chose eps properly
# Use kNNdistplot() from dbscan package to find the "elbow"
kNNdistplot(iris_data, k = 5) # chose k = minPts
abline(h = 0.6, col = "red", lty = 2) # Example esp = 0.6

# Run DBSCAN with selected minPts and eps
db_2 <- dbscan(iris_data, eps = 0.6, minPts = 5)
db_cluster <- as.factor(db_2$cluster) # 0 = noise

# visualize using first two principal components (or t-SNE or UMAP)
pca_df$db_cluster <- db_cluster

ggplot(pca_df, aes(PC1, PC2, color = db_cluster)) +
  geom_point(alpha = 0.5) +
  labs(title = "DBSCAN Clustering on Iris Data, PCA Reduced",
       x = "PC1",
       y = "PC2") +
  theme_minimal()



#------------------------------------------------------------------------------
# Metrics are used to evaluate clustering (Silhouette score)
#------------------------------------------------------------------------------


# Measures how similar a point is to its own cluster compared to other clusters.
# ~1 = good fit within its cluster. 0 = between two clusters. Negative = wrong cluster

if(!require(cluster)) install.packages("cluster")
library(cluster)

# In k-means clustering
km <- kmeans(iris_data, centers = 3, nstart = 25)

sil_km <- silhouette(km$cluster, dist(iris_data)) # Silhouette score

plot(sil_km, main = "Silhouette Plot in K-Means")

mean(sil_km[, 3]) # Average silhouette width


# In DBSCAN clustering
db_2 <- dbscan(iris_data, eps = 0.6, minPts = 5)

db_cluster <- db_2$cluster
valid <- db_cluster != 0

sil_db <- silhouette(db_cluster[valid], dist(iris_data[valid, ])) 

plot(sil_db, main = "Silhouette Plot in DBSCAN")

mean(sil_db[, 3])
