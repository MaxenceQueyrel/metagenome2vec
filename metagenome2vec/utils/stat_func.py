from scipy.stats import normaltest, bartlett, levene, ttest_ind


def ttest_cv(score_cv_1, score_cv_2):
    if len(score_cv_1) < 8 or len(score_cv_2) < 8:
        return 0.0, 0.0
    normal = True
    same_var = True
    alpha = 0.1

    ## Test normal
    k2_1, p_1 = normaltest(score_cv_1)
    k2_2, p_2 = normaltest(score_cv_2)
    if p_1 < alpha or p_2 < alpha:
        normal = False

    ## Test same variance
    if normal:
        s, p = bartlett(score_cv_1, score_cv_2)
    else:
        s, p = levene(score_cv_1, score_cv_2)
    if p < alpha:
        same_var = False

    ## Test student
    s, p = ttest_ind(score_cv_1, score_cv_2, equal_var=same_var)
    return s, p
