| method                                | #clusters | prec, recall, fscore | time   |
|---------------------------------------|-----------|----------------------|--------|
| * kmeans (ncluster=2577)              | 2577      | 94.3, 74.98, 83.54   | 618.1s |
| kmeans (ncluster=2000)                | 2000      | 84.5, 84.46, 84.48   | 556.4s |
| * MiniBatchKMeans                     | 2577      | 89.98, 87.86, 88.91  | 122.8s |
| MiniBatchKMeans                       | 2000      | 74.37, 92.97, 82.63  | 90.2s  |
| * Spectral (ncluster=2577)            | 2577      |                      |        |
| Spectral (ncluster=2000)              | 2000      |                      |        |
| * HAC (ncluster=2577, knn=30)         | 2577      |                      |        |
| HAC (ncluster=2000, knn=30)           | 2000      |                      |        |
| FastHAC (distance=0.7, method=single) | 46767     | 99.79, 53.18, 69.38  | 1.66h  |
| DBSCAN (eps=0.75, nim_samples=10)     |           |                      |        |
| HDBSCAN (min_samples=10)              |           |                      |        |
| KNN DBSCAN (knn=80, min_samples=10)   | 2494      | 1.358, 78.99, 2.669  | 60.5s  |
| ApproxRankOrder (knn=20, th=10)       | 85150     | 52.96, 16.93, 25.66  | 86.4s  |
| ApproxRankOrder (knn=20, th=2)        | 97121     | 86.52, 9.495, 17.11  | 86.7s  |
