"""Analisis deterministik; tidak memakai data uji untuk keputusan model/cutoff."""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data.download import COLUMNS, NUMERIC, load, map_target
from src.scorecard import FACTOR, Scorecard, decision_cost, probability_to_score, psi, select_cutoff

SEED = 2024


def booster():
    numeric = Pipeline([('impute', SimpleImputer(strategy='median'))])
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    prep = ColumnTransformer([('numeric', numeric, NUMERIC),
                              ('category', categorical, [c for c in COLUMNS if c not in NUMERIC])])
    return Pipeline([('prep', prep), ('model', GradientBoostingClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.05, random_state=SEED))])


def metrics(y, p):
    auc = roc_auc_score(y, p)
    return {'auc': float(auc), 'gini': float(2*auc-1),
            'ks': float(ks_2samp(p[np.asarray(y) == 0], p[np.asarray(y) == 1]).statistic),
            'brier': float(brier_score_loss(y, p))}


def fairness(y, p, groups, cutoff, attribute):
    rows = []
    for group in sorted(groups.unique()):
        mask = np.asarray(groups == group)
        yy, pp = np.asarray(y)[mask], p[mask]
        tn, fp, fn, tp = confusion_matrix(yy, pp >= cutoff, labels=[0, 1]).ravel()
        n = len(yy)
        rate = float((pp < cutoff).mean())
        # Wilson 95% interval untuk approval, termasuk kelompok kecil.
        center = (rate + 1.96**2/(2*n))/(1+1.96**2/n)
        half = 1.96*np.sqrt(rate*(1-rate)/n+1.96**2/(4*n*n))/(1+1.96**2/n)
        rows.append({'attribute': attribute, 'group': group, 'n': n,
                     'bad_n': int(yy.sum()), 'approval_rate': rate,
                     'approval_ci_low': center-half, 'approval_ci_high': center+half,
                     'false_reject_rate_good': fp/(fp+tn) if fp+tn else np.nan,
                     'false_approve_rate_bad': fn/(fn+tp) if fn+tp else np.nan,
                     'brier': brier_score_loss(yy, pp)})
    return pd.DataFrame(rows)


def run(data=None, output='.'):
    root = Path(output)
    for folder in ['reports', 'assets', 'artifacts']:
        (root/folder).mkdir(parents=True, exist_ok=True)
    data = load() if data is None else data.copy()
    X, y = data[COLUMNS], map_target(data.label)
    train, test = train_test_split(np.arange(len(data)), test_size=.2, stratify=y, random_state=SEED)
    Xtr, Xte, ytr, yte = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        oof = cross_val_predict(Scorecard(), Xtr, ytr, cv=cv, method='predict_proba')[:, 1]
        model = Scorecard().fit(Xtr, ytr)
    notices = sorted(set(str(w.message) for w in captured))
    bg = booster()
    oof_bg = cross_val_predict(bg, Xtr, ytr, cv=cv, method='predict_proba')[:, 1]
    bg.fit(Xtr, ytr)
    choice = select_cutoff(ytr, oof)
    p, pb = model.predict_proba(Xte)[:, 1], bg.predict_proba(Xte)[:, 1]
    score = probability_to_score(p)
    if not np.allclose(model.points(Xte).sum(axis=1), score):
        raise AssertionError('Poin atribut tidak konsisten dengan probabilitas.')
    result = {'seed': SEED, 'n': len(data), 'train_n': len(train), 'test_n': len(test),
              'bad_n': int(y.sum()), 'missing_cells': int(data.isna().sum().sum()),
              'exact_duplicate_rows': int(data.duplicated().sum()),
              'selected_attributes': model.woe_.selected_, 'oof_choice': choice,
              'scorecard': metrics(yte, p), 'boosting': metrics(yte, pb),
              'oof_scorecard': metrics(ytr, oof), 'oof_boosting': metrics(ytr, oof_bg),
              'test_decision': decision_cost(yte, p, choice['cutoff']),
              'psi_train_test': psi(model.predict_proba(Xtr)[:, 1], p),
              'logistic_converged': bool(model.result_.converged), 'fit_warnings': notices,
              'score_probability_max_error': float(np.max(np.abs(model.points(Xte).sum(axis=1)-score)))}
    iv = pd.DataFrame({'attribute': list(model.woe_.iv_), 'iv': list(model.woe_.iv_.values())})
    iv['selected'] = iv.attribute.isin(model.woe_.selected_)
    iv.sort_values('iv', ascending=False).to_csv(root/'reports/information_value.csv', index=False)
    model.woe_.table_.to_csv(root/'reports/woe.csv', index=False)
    points = model.woe_.table_.query('attribute in @model.woe_.selected_').copy()
    points['coefficient'] = points.attribute.map(model.result_.params)
    points['points'] = -FACTOR*points.coefficient*points.woe
    points.to_csv(root/'reports/scorecard_points.csv', index=False)
    result['base_points'] = float(model.points(Xte)['base'].iloc[0])
    summary = pd.DataFrame({'coefficient': model.result_.params, 'p_value': model.result_.pvalues,
                            'ci_low': model.result_.conf_int()[0], 'ci_high': model.result_.conf_int()[1]})
    summary.to_csv(root/'reports/coefficients.csv', index_label='attribute')
    pd.DataFrame({'row': train, 'split': 'train'}).pipe(lambda a: pd.concat([
        a, pd.DataFrame({'row': test, 'split': 'test'})])).to_csv(root/'reports/split.csv', index=False)
    pd.DataFrame({'row': test, 'default': yte.to_numpy(), 'probability': p, 'score': score,
                  'boosting_probability': pb}).to_csv(root/'reports/test_predictions.csv', index=False)
    pd.DataFrame([dict(model='scorecard', **result['scorecard']),
                  dict(model='boosting', **result['boosting'])]).to_csv(root/'reports/evaluation.csv', index=False)
    cutoffs = pd.DataFrame([decision_cost(yte, p, c) for c in sorted(set([.1, .2, .3, .5, choice['cutoff']]))])
    cutoffs.to_csv(root/'reports/cutoffs.csv', index=False)
    age = pd.Series(np.where(Xte.age < 25, '<25', np.where(Xte.age < 50, '25-49', '50+')), index=Xte.index)
    sex = Xte.personal_status_sex.map({'A91': 'male', 'A92': 'female', 'A93': 'male',
                                      'A94': 'male', 'A95': 'female'}).fillna('unknown')
    fair = pd.concat([fairness(yte, p, age, choice['cutoff'], 'age'),
                      fairness(yte, p, sex, choice['cutoff'], 'sex')], ignore_index=True)
    fair.to_csv(root/'reports/fairness.csv', index=False)
    data.describe(include='all').to_csv(root/'reports/eda.csv')
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for label, pred in [('Scorecard', p), ('Gradient boosting', pb)]:
        fpr, tpr, _ = roc_curve(yte, pred)
        ax[0].plot(fpr, tpr, label=f'{label} (AUC {roc_auc_score(yte, pred):.3f})')
        actual, mean = calibration_curve(yte, pred, n_bins=5, strategy='quantile')
        ax[1].plot(mean, actual, 'o-', label=label)
    for a in ax:
        a.plot([0, 1], [0, 1], '--', color='gray')
        a.legend(fontsize=8)
        a.grid(alpha=.2)
    ax[0].set(xlabel='False positive rate', ylabel='True positive rate', title='Diskriminasi pada data uji')
    ax[1].set(xlabel='Rata-rata prediksi', ylabel='Proporsi bad aktual', title='Kalibrasi: 5 kelompok kuantil')
    fig.tight_layout()
    fig.savefig(root/'assets/evaluation.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    iv.sort_values('iv').plot.barh(x='attribute', y='iv', ax=ax, legend=False, color='#087e8b')
    ax.set(title='Information Value, hanya data latih', xlabel='IV')
    fig.tight_layout()
    fig.savefig(root/'assets/information_value.png', dpi=150)
    plt.close(fig)
    (root/'reports/hasil.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    joblib.dump({'model': model, 'cutoff': choice['cutoff'],
                 'defaults': Xtr.iloc[0].to_dict(),
                 'categories': {c: sorted(Xtr[c].dropna().unique().tolist()) for c in COLUMNS if c not in NUMERIC}},
                root/'artifacts/scorecard.joblib')
    from src.reporting import write_reports
    write_reports(root, result, iv, summary, fair, cutoffs)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    run()
