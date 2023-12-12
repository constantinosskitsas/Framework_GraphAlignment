1. Import ``helpers/pred.py`` which has the source implementation of FUGAL
2. Call ``predict_alignment()`` function in ``helpers/pred.py`` which takes arguments (queries, targets, mu, niter)
3. queries, targets should be a list of networkx graphs. Hyperparameters:
    1. For MultiMagma, use mu = 0.5, niter = 15
    2. For Euroroad, use mu = 2, niter = 10
    3. For NW, use mu = 4, niter = 15
4. For MultiMagma please comment lines 102-104 in ``helpers/pred.py`` as we used only first 4 features.
