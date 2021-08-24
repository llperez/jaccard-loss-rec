import jac_data
import jac_model
import phik


X = jac_data.get_data()

phi_X = np.nan_to_num(phik.phik_matrix(pd.DataFrame(X, columns=[str(x) for x in range(X.shape[1])]), dropna=False, interval_cols=[]).to_numpy())

