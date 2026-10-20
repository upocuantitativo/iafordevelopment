# Code S1 - socsci-4597945 (revision). Robustness analyses: repeated/nested CV, permutation importance, PCA and cluster validation.
# Run from the repository root: python robustness/code_S1_robustness.py ; outputs go to robustness/results/ ; Python 3.13, scikit-learn 1.9, xgboost 3.4.

"""Robustness analyses requested by reviewers (socsci-4597945).
Leakage-free repeated cross-validation, permutation importance, cluster validity.
Data: DATA_GHAB.xlsx from github.com/upocuantitativo/iafordevelopment (62 countries).
"""
import json, warnings, sys
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold, KFold, LeaveOneOut, GridSearchCV, train_test_split, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import xgboost as xgb
warnings.filterwarnings("ignore")
import os
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'DATA_GHAB.xlsx')
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
os.chdir(os.path.join(HERE, 'results'))
SEED = 42
T = 'G_GPD_PCAP_SLOPE'
d = pd.read_excel(DATA)
X_all = d.drop(columns=['Pais', T]); y = d[T].values
cols = list(X_all.columns)
out = {'n_countries': len(d), 'n_predictors': len(cols)}

class TopKCorr(BaseEstimator, TransformerMixin):
    """Select k predictors with the largest |Pearson r| with y, fitted on training data only."""
    def __init__(self, k=15): self.k = k
    def fit(self, X, y):
        Xa = np.asarray(X, float)
        r = np.array([abs(np.corrcoef(Xa[:, j], y)[0, 1]) if Xa[:, j].std() > 0 else 0 for j in range(Xa.shape[1])])
        self.idx_ = np.argsort(-np.nan_to_num(r))[:self.k]; return self
    def transform(self, X): return np.asarray(X, float)[:, self.idx_]

def models():
    return {
        'Random forest': RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1),
        'Gradient boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED),
        'XGBoost': xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED, n_jobs=1, verbosity=0),
        'SVM (RBF)': Pipeline([('s', StandardScaler()), ('m', SVR(C=10, gamma='scale'))]),
        'Neural network (MLP)': Pipeline([('s', StandardScaler()), ('m', MLPRegressor(hidden_layer_sizes=(50,), alpha=0.1, max_iter=5000, random_state=SEED))]),
        'Ridge regression (linear benchmark)': Pipeline([('s', StandardScaler()), ('m', RidgeCV(alphas=np.logspace(-3, 3, 25)))]),
    }

# ---------- correlations (full sample, descriptive) ----------
cor = []
for c in cols:
    pr, pp = pearsonr(X_all[c], y); sr, sp = spearmanr(X_all[c], y)
    cor.append((c, pr, pp, sr, sp))
cor = pd.DataFrame(cor, columns=['var', 'pearson', 'p_pearson', 'spearman', 'p_spearman']).reindex()
cor['abs'] = cor.pearson.abs(); cor = cor.sort_values('abs', ascending=False)
cor.to_csv('res_correlations.csv', index=False)
out['top_corr'] = cor.head(15)[['var', 'pearson', 'p_pearson', 'spearman']].round(3).values.tolist()
out['n_sig_p05'] = int((cor.p_pearson < 0.05).sum())

# ---------- A. reproduce original single split ----------
top15 = cor['var'].head(15).tolist()
Xo = X_all[top15]
ts = min(0.3, 0.5 - 5 / len(d))
Xtr, Xte, ytr, yte = train_test_split(Xo, y, test_size=ts, random_state=SEED)
gs = GridSearchCV(RandomForestRegressor(random_state=SEED), {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}, cv=3, scoring='r2', n_jobs=-1).fit(Xtr, ytr)
out['orig_split'] = {'n_train': len(ytr), 'n_test': len(yte), 'test_r2': r2_score(yte, gs.predict(Xte)), 'inner_cv_r2': gs.best_score_, 'params': gs.best_params_}
# distribution of single-split test R2 over 100 different random splits (same procedure, fixed params)
sp = []
for s in range(100):
    a, b, c_, e = train_test_split(Xo, y, test_size=ts, random_state=s)
    m = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1).fit(a, c_)
    sp.append(r2_score(e, m.predict(b)))
out['split_r2_distribution'] = {'mean': np.mean(sp), 'p5': np.percentile(sp, 5), 'p95': np.percentile(sp, 95), 'min': min(sp), 'max': max(sp)}

# ---------- B. leakage-free repeated CV (selection inside folds) ----------
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED)
cvres = {}; oof_rf = None
for name, m in models().items():
    pipe = Pipeline([('sel', TopKCorr(15)), ('m', m)])
    r2s, maes, rep_r2 = [], [], []
    oof = np.zeros((10, len(y)))
    for i, (tr, te) in enumerate(rkf.split(X_all)):
        p = clone(pipe).fit(X_all.iloc[tr], y[tr]); pr = p.predict(X_all.iloc[te])
        oof[i // 5, te] = pr; r2s.append(r2_score(y[te], pr)); maes.append(mean_absolute_error(y[te], pr))
    rep_r2 = [r2_score(y, oof[k]) for k in range(10)]
    cvres[name] = {'fold_r2_mean': np.mean(r2s), 'fold_r2_sd': np.std(r2s), 'oof_r2_mean': np.mean(rep_r2), 'oof_r2_min': np.min(rep_r2), 'oof_r2_max': np.max(rep_r2), 'mae_mean': np.mean(maes)}
    if name == 'Random forest': oof_rf = oof.mean(0)
    print(name, cvres[name], flush=True)
out['cv_top15'] = cvres
np.save('oof_rf.npy', oof_rf)
out['rf_oof_pearson'] = pearsonr(oof_rf, y)[0]

# nested tuning for RF (inner 3-fold grid inside each outer fold), 5x5
r2n = []
for i, (tr, te) in enumerate(RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(X_all)):
    pipe = Pipeline([('sel', TopKCorr(15)), ('m', RandomForestRegressor(random_state=SEED, n_jobs=-1))])
    g = GridSearchCV(pipe, {'m__n_estimators': [100, 300], 'm__max_depth': [None, 5, 10], 'm__min_samples_leaf': [1, 3]}, cv=3, scoring='r2').fit(X_all.iloc[tr], y[tr])
    r2n.append(r2_score(y[te], g.predict(X_all.iloc[te])))
out['rf_nested'] = {'mean': np.mean(r2n), 'sd': np.std(r2n)}
# LOOCV RF
loo = cross_val_predict(Pipeline([('sel', TopKCorr(15)), ('m', RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1))]), X_all, y, cv=LeaveOneOut())
out['rf_loocv_r2'] = r2_score(y, loo)
# RF on all predictors (no screening)
r2a = []
for tr, te in rkf.split(X_all):
    m = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1).fit(X_all.iloc[tr], y[tr]); r2a.append(r2_score(y[te], m.predict(X_all.iloc[te])))
out['rf_all128'] = {'mean': np.mean(r2a), 'sd': np.std(r2a)}
# k sensitivity
out['rf_k_sens'] = {}
for k in [5, 10, 20, 30]:
    r = []
    for tr, te in RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(X_all):
        p = Pipeline([('sel', TopKCorr(k)), ('m', RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1))]).fit(X_all.iloc[tr], y[tr]); r.append(r2_score(y[te], p.predict(X_all.iloc[te])))
    out['rf_k_sens'][k] = np.mean(r)
# influential observations: drop 3 most extreme outcomes
keep = ~d.Pais.isin(['Angola', 'Samoa', 'Guinea']).values
r = []
for tr, te in RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED).split(X_all[keep]):
    Xa, ya = X_all[keep].iloc[tr], y[keep][tr]
    p = Pipeline([('sel', TopKCorr(15)), ('m', RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1))]).fit(Xa, ya); r.append(r2_score(y[keep][te], p.predict(X_all[keep].iloc[te])))
out['rf_drop_extremes'] = {'mean': np.mean(r), 'sd': np.std(r), 'n': int(keep.sum())}

# ---------- importance ----------
rf_full = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1).fit(Xo, y)
mdi = pd.Series(rf_full.feature_importances_, index=top15).sort_values(ascending=False)
out['mdi_full_top15'] = mdi.round(4).to_dict()
# out-of-fold permutation importance and selection frequency
perm = {c: [] for c in cols}; selfreq = {c: 0 for c in cols}; mdi_cv = {c: [] for c in cols}
for tr, te in rkf.split(X_all):
    sel = TopKCorr(15).fit(X_all.iloc[tr], y[tr]); names = [cols[j] for j in sel.idx_]
    for n in names: selfreq[n] += 1
    m = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1).fit(X_all.iloc[tr][names], y[tr])
    pi = permutation_importance(m, X_all.iloc[te][names], y[te], n_repeats=20, random_state=SEED, scoring='r2')
    for j, n in enumerate(names):
        perm[n].append(pi.importances_mean[j]); mdi_cv[n].append(m.feature_importances_[j])
imp = pd.DataFrame({'sel_freq': pd.Series(selfreq) / 50, 'perm_mean_when_selected': pd.Series({k: np.mean(v) if v else np.nan for k, v in perm.items()}),
                    'perm_mean_all50': pd.Series({k: np.sum(v) / 50 for k, v in perm.items()}), 'mdi_mean_when_selected': pd.Series({k: np.mean(v) if v else np.nan for k, v in mdi_cv.items()})})
imp = imp[imp.sel_freq > 0].sort_values('perm_mean_all50', ascending=False)
imp.to_csv('res_importance.csv')
out['imp_top'] = imp.head(15).round(4).reset_index().values.tolist()
common = imp.dropna().index
out['rank_corr_perm_vs_mdi'] = spearmanr(imp.loc[common, 'perm_mean_when_selected'], imp.loc[common, 'mdi_mean_when_selected'])[0]

# ---------- clustering ----------
Z = StandardScaler().fit_transform(X_all)
pca = PCA(n_components=10).fit(Z); P = pca.transform(Z)
out['pca_evr'] = pca.explained_variance_ratio_.round(4).tolist()
out['pca_cum10'] = float(pca.explained_variance_ratio_.sum())
load = pd.Series(pca.components_[0], index=cols)
out['pc1_top_loadings'] = load.reindex(load.abs().sort_values(ascending=False).index).head(10).round(3).to_dict()
load2 = pd.Series(pca.components_[1], index=cols)
out['pc2_top_loadings'] = load2.reindex(load2.abs().sort_values(ascending=False).index).head(10).round(3).to_dict()
sil, wss = {}, {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=SEED, n_init=50).fit(P); sil[k] = silhouette_score(P, km.labels_); wss[k] = km.inertia_
out['silhouette_excl_target'] = {k: round(v, 3) for k, v in sil.items()}
out['wss'] = {k: round(v, 1) for k, v in wss.items()}
lab = KMeans(n_clusters=2, random_state=SEED, n_init=50).fit_predict(P)
# original (target included in clustering, as in released code)
Zt = StandardScaler().fit_transform(d.drop(columns=['Pais']))
Pt = PCA(n_components=10).fit_transform(Zt)
labt = KMeans(n_clusters=2, random_state=SEED, n_init=50).fit_predict(Pt)
out['ari_with_vs_without_target'] = adjusted_rand_score(lab, labt)
silt = {k: round(silhouette_score(Pt, KMeans(n_clusters=k, random_state=SEED, n_init=50).fit_predict(Pt)), 3) for k in range(2, 7)}
out['silhouette_incl_target'] = silt
g = [y[lab == c] for c in (0, 1)]
out['cluster_sizes'] = [int((lab == c).sum()) for c in (0, 1)]
out['cluster_growth_mean'] = [float(np.mean(x)) for x in g]; out['cluster_growth_median'] = [float(np.median(x)) for x in g]
out['cluster_mw_p'] = float(mannwhitneyu(g[0], g[1]).pvalue)
out['cluster_members'] = {int(c): d.Pais[lab == c].tolist() for c in (0, 1)}
# PC1 correlation with growth
out['pc1_growth_spearman'] = spearmanr(P[:, 0], y)[0]
# bootstrap stability k=2
rng = np.random.default_rng(SEED); aris = []
for b in range(200):
    ix = rng.choice(len(y), len(y), replace=True)
    kb = KMeans(n_clusters=2, random_state=SEED, n_init=20).fit(P[ix]); aris.append(adjusted_rand_score(lab, kb.predict(P)))
out['boot_ari_k2'] = {'mean': np.mean(aris), 'p5': np.percentile(aris, 5)}
# k=3 profile growth
lab3 = KMeans(n_clusters=3, random_state=SEED, n_init=50).fit_predict(P)
out['k3_growth_means'] = [float(y[lab3 == c].mean()) for c in range(3)]; out['k3_sizes'] = [int((lab3 == c).sum()) for c in range(3)]
pd.DataFrame({'Pais': d.Pais, 'cluster': lab, 'PC1': P[:, 0], 'PC2': P[:, 1], 'growth': y, 'oof_rf': oof_rf}).to_csv('res_countries.csv', index=False)
json.dump(out, open('res.json', 'w'), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
print(json.dumps(out, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
import json,numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.stats import mannwhitneyu, spearmanr
d=pd.read_excel(DATA); y=d.G_GPD_PCAP_SLOPE.values
X=d.drop(columns=['Pais','G_GPD_PCAP_SLOPE'])
count=[c for c in X.columns if (c.startswith('P_Population ages') or c.startswith('P_Age population') or c in ('P_Population, female','L_Labor force, female','E_Children out of school, primary, female','E_Youth illiterate population, 15-24 years, female (number)','H_Number of infant deaths, female','H_Number of stillbirths','H_Number of under-five deaths, female'))]
Xr=X.drop(columns=count); print('excluded',len(count),'remaining',Xr.shape[1])
Z=StandardScaler().fit_transform(Xr); pca=PCA(10).fit(Z); P=pca.transform(Z)
out={'n_count_excluded':len(count),'n_rate':Xr.shape[1],'evr':pca.explained_variance_ratio_[:3].round(4).tolist()}
l1=pd.Series(pca.components_[0],index=Xr.columns); out['pc1_load']=l1.reindex(l1.abs().sort_values(ascending=False).index).head(10).round(3).to_dict()
sil={k:round(silhouette_score(P,KMeans(k,random_state=42,n_init=50).fit_predict(P)),3) for k in range(2,7)}; out['sil']=sil
for k in (2,3):
    lab=KMeans(k,random_state=42,n_init=50).fit_predict(P)
    out[f'k{k}']={'sizes':[int((lab==c).sum()) for c in range(k)],'growth_mean':[round(float(y[lab==c].mean()),1) for c in range(k)],'growth_median':[round(float(np.median(y[lab==c])),1) for c in range(k)]}
    if k==2: out['k2']['mw_p']=float(mannwhitneyu(y[lab==0],y[lab==1]).pvalue); out['k2']['members_small']=d.Pais[lab==np.argmin(np.bincount(lab))].tolist()
    if k==3:
        from scipy.stats import kruskal; out['k3']['kw_p']=float(kruskal(*[y[lab==c] for c in range(3)]).pvalue)
out['pc1_growth_spearman']=spearmanr(P[:,0],y)[0]; out['pc1_growth_p']=spearmanr(P[:,0],y)[1]
print(json.dumps(out,indent=1,default=float)); json.dump(out,open('res_clus2.json','w'),indent=1,default=float)
import json,numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
d=pd.read_excel(DATA); y=d.G_GPD_PCAP_SLOPE.values
X=d.drop(columns=['Pais','G_GPD_PCAP_SLOPE'])
count=[c for c in X.columns if (c.startswith('P_Population ages') or c.startswith('P_Age population') or c in ('P_Population, female','L_Labor force, female','E_Children out of school, primary, female','E_Youth illiterate population, 15-24 years, female (number)','H_Number of infant deaths, female','H_Number of stillbirths','H_Number of under-five deaths, female'))]
Xr=X.drop(columns=count); Z=StandardScaler().fit_transform(Xr); pca=PCA(10).fit(Z); P=pca.transform(Z)
o={'cum10':float(pca.explained_variance_ratio_.sum())}
o['wss']={k:round(KMeans(k,random_state=42,n_init=50).fit(P).inertia_,1) for k in range(1,7)}
lab=KMeans(2,random_state=42,n_init=50).fit_predict(P)
rng=np.random.default_rng(42); a=[]
for b in range(200):
    ix=rng.choice(62,62,replace=True); a.append(adjusted_rand_score(lab,KMeans(2,random_state=42,n_init=20).fit(P[ix]).predict(P)))
o['boot_ari']={'mean':float(np.mean(a)),'median':float(np.median(a)),'p5':float(np.percentile(a,5))}
small=np.argmin(np.bincount(lab))
for c in (0,1):
    m=lab==c
    o[f'cl{c}']={'n':int(m.sum()),'small':bool(c==small),'pc1':float(P[m,0].mean()),'prim_enrol':float(d.loc[m,'E_School enrollment, primary, female (% gross)'].mean()),'u5':float(d.loc[m,'H_Mortality rate, under-5, female (per 1,000 live births)'].mean()),'lifeexp':float(d.loc[m,'P_Life expectancy at birth, female (years)'].mean()),'growth_med':float(np.median(y[m]))}
# loadings S3
L=pd.DataFrame(pca.components_[:3].T,index=Xr.columns,columns=['PC1','PC2','PC3']); L.to_csv('res_loadings_rate.csv')
cc=pd.read_csv('res_countries.csv'); cc['PC1r']=P[:,0]; cc['PC2r']=P[:,1]; cc['cluster_r']=lab; cc['cluster_all128']=cc['cluster']; cc.to_csv('res_countries.csv',index=False)
o['evr2']=pca.explained_variance_ratio_[:2].tolist()
# histogram
o['hist']={'<0':int((y<0).sum()),'0-20':int(((y>=0)&(y<20)).sum()),'20-40':int(((y>=20)&(y<40)).sum()),'40-60':int(((y>=40)&(y<60)).sum()),'60-80':int(((y>=60)&(y<80)).sum()),'>=80':int((y>=80).sum())}
o['skew']=float(pd.Series(y).skew())
cc['res']=cc.growth-cc.oof_rf; o['big_res']=cc.reindex(cc.res.abs().sort_values(ascending=False).index)[['Pais','growth','oof_rf']].head(8).round(1).values.tolist()
o['oof_range']=[float(cc.oof_rf.min()),float(cc.oof_rf.max())]
print(json.dumps(o,indent=1)); json.dump(o,open('res_extra.json','w'),indent=1)
import json,urllib.request,time,numpy as np,pandas as pd
from scipy.stats import spearmanr
d=pd.read_excel(DATA)
cl=json.load(urllib.request.urlopen('https://api.worldbank.org/v2/country?format=json&per_page=400',timeout=90))[1]
n2i={x['name']:x['id'] for x in cl}
al={'Egypt, Arab Rep.':'EGY','Gambia, The':'GMB','Yemen, Rep.':'YEM','Congo, Dem. Rep.':'COD','Congo, Rep.':'COG','Kyrgyz Republic':'KGZ','Lao PDR':'LAO','Tanzania':'TZA','Syrian Arab Republic':'SYR'}
iso=[al.get(p,n2i.get(p)) for p in d.Pais]
def first(code):
    for t in range(4):
        try:
            j=json.load(urllib.request.urlopen(f'https://api.worldbank.org/v2/country/{";".join(iso)}/indicator/{code}?format=json&date=1960:2023&per_page=20000',timeout=120)); break
        except Exception: time.sleep(5)
    df=pd.DataFrame([(x['countryiso3code'],int(x['date']),x['value']) for x in j[1] if x['value'] is not None],columns=['iso','yr','v'])
    return df.sort_values('yr').groupby('iso').first()
o={}
for code,col,lab in [('SE.PRM.ENRR.FE','E_School enrollment, primary, female (% gross)','prim'),('SH.DYN.MORT.FE','H_Mortality rate, under-5, female (per 1,000 live births)','u5'),('NY.GDP.PCAP.CD','G_GPD_PCAP_SLOPE','gdp')]:
    f=first(code); x=pd.Series([f.v.get(i,np.nan) for i in iso]); yr=pd.Series([f.yr.get(i,np.nan) for i in iso])
    m=x.notna()
    r1=spearmanr(x[m],d.loc[m,col]); r2=spearmanr(x[m],d.loc[m,'G_GPD_PCAP_SLOPE'])
    o[lab]={'n':int(m.sum()),'median_first_year':float(yr.median()),'rho_initial_vs_own_slope':[round(r1[0],3),float(r1[1])],'rho_initial_vs_gdp_slope':[round(r2[0],3),float(r2[1])]}
    if lab=='gdp': o['gdp_first_level_vs_gdp_slope_note']='same as own slope'
print(json.dumps(o,indent=1)); json.dump(o,open('res_base.json','w'),indent=1)
