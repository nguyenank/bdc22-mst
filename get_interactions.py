from sklearn.preprocessing import PolynomialFeatures

def get_interactions(x): 
    interactions = PolynomialFeatures(interaction_only=True, include_bias=True)
    x_w_inter = interactions.fit_transform(X=x)
    inter_vars_raw = interactions.get_feature_names_out()

    new_names = []

    for i in inter_vars_raw:
        n = i.replace(' ', '_')
        new_names.append(n)

    return x_w_inter, new_names, inter_vars_raw