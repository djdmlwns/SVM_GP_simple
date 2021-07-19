# Plot heatmap of uncertainty to check only for 2-D
if dim == 2:
    plot_heatmap_uncertainty(AS.GP_regressor)

# %%
# Plot SVM boundary of the proposed algorithm if needed only for 2-D
if dim == 2:
    plot_svm_boundary(AS.SVM_classifier, AS.X, AS.y)


