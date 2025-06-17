
def compare_pre(reconstruction_ref, reconstruction_test) :
    
    for i in range (6):
        reconstruction_ref[i, :, :, :] = np.where(reconstruction_ref[i, :, :, :] == 0, 0.0005, reconstruction_ref[i, :, :, :])

            
    pre = (reconstruction_ref[:, :, :, :]-reconstruction_test[:, :, :, :])*100/reconstruction_ref[:, :, :, :]
    diff = (reconstruction_ref[:, :, :, :]-reconstruction_test[:, :, :, :])

    reconstruction_test_nan = reconstruction_test.copy()
    reconstruction_ref_nan = reconstruction_ref.copy()

    reconstruction_test_nan[np.isnan(reconstruction_test_nan)] = 1
    reconstruction_ref_nan[np.isnan(reconstruction_ref_nan)] = 1


    pre_sum = np.sum(np.abs(reconstruction_ref_nan[:, :, :, :]-reconstruction_test_nan[:, :, :, :])*100/np.abs(reconstruction_ref_nan[:, :, :, :]), axis=(1,2,3))
    diff_sum = np.sum(np.abs(reconstruction_ref_nan[:, :, :, :]-reconstruction_test_nan[:, :, :, :]), axis=(1,2,3))

    pre_sum[2] = diff_sum[2]
    pre_sum[4] = diff_sum[4]

    return pre, diff, pre_sum