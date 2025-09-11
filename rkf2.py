import os
import joblib
import numpy as np

from pmlb import fetch_data
from itertools import product
from multiprocessing.pool import Pool

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, pairwise_distances
from sklearn.preprocessing import StandardScaler


from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.problem import ElementwiseProblem

from configs import PREFIX, CS, NS_COEFS, DSETS, NS_SPLITS, TEST_SIZES, SEEDS, POPS_SIZE, NS_GEN, SEEDS, SVC_MAX_ITER, SVC_MAX_ITER

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


class RKFP(ElementwiseProblem):
    def __init__(self, C, n_coefs, dset, n_splits, test_size, seed, **kwargs):        
        self.C = C
        self.n_coefs = n_coefs
        self.dset = dset
        self.n_splits = n_splits
        self.test_size = test_size
        self.seed = seed
        
        # --- OTIMIZAÇÃO: Carregar e dividir os dados APENAS UMA VEZ ---
        # Os dados são carregados e divididos no construtor e armazenados
        # para serem reutilizados em todas as chamadas de _evaluate.
        dset_X, dset_y = fetch_data(dataset_name=self.dset, return_X_y=True, local_cache_dir='datasets')
        self.Xn, self.X_test, self.yn, self.y_test = train_test_split(
            dset_X, dset_y, test_size=self.test_size, random_state=self.seed, shuffle=True, stratify=dset_y
        )
        
        # --- OTIMIZAÇÃO: O objeto KFold também é criado apenas uma vez ---
        self.skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        
        super().__init__(
            n_var=2*self.n_coefs, n_obj=1, n_ieq_constr=0, n_eq_constr=0,
            xl=-1.0, xu=+1.0, elementwise_evaluation=True, **kwargs
        )
        
    def arr2poly(self, a):
        if self.n_coefs == 3: return lambda x: (a[0]*x + a[1])*x + a[2]
        if self.n_coefs == 4: return lambda x: ((a[0]*x + a[1])*x + a[2])*x + a[3]
        if self.n_coefs == 5: return lambda x: (((a[0]*x + a[1])*x + a[2])*x + a[3])*x + a[4]
            
    def get_f(self, x):
        num = self.arr2poly(x[:self.n_coefs])
        den = self.arr2poly(x[self.n_coefs:])
        return lambda y: num(y) / den(y)
    
    @ignore_warnings(category=ConvergenceWarning)
    def _evaluate(self, x, out, *args, **kwargs):
        f = self.get_f(x=x.copy())
        
        # --- OTIMIZAÇÃO: Usa os dados pré-carregados (self.Xn, self.yn) ---
        # Remove a leitura de disco (fetch_data) e a divisão (train_test_split) de dentro do loop de avaliação.
        
        l_bas, l_itr, l_nsv = list(), list(), list()
        
        # --- OTIMIZAÇÃO: Usa o objeto skf pré-criado ---
        for train_index, test_index in self.skf.split(X=self.Xn, y=self.yn):
            X0, X1, y0, y1 = self.Xn[train_index], self.Xn[test_index], self.yn[train_index], self.yn[test_index]
            
            rs = StandardScaler()
            X0s = rs.fit_transform(X0)
            X1s = rs.transform(X1)
            
            pd00 = pairwise_distances(X0s, X0s)
            G0 = f(pd00) # train gram matrix
            pd10 = pairwise_distances(X1s, X0s)
            G1 = f(pd10) # valid gram matrix
            
            svc = SVC(C=self.C, kernel='precomputed', max_iter=SVC_MAX_ITER)
            svc.fit(G0, y0)
            yp = svc.predict(G1)
            
            l_bas.append( balanced_accuracy_score(y_true=y1, y_pred=yp) )
            l_itr.append( np.mean(svc.n_iter_) )
            l_nsv.append( svc.support_.shape[0] / X0s.shape[0] )
            
        bas = np.mean(l_bas)
        itr = np.mean(l_itr)
        nsv = np.mean(l_nsv)

        out['F'] = -1 * bas + (self.n_splits / self.yn.shape[0]) * ((itr / SVC_MAX_ITER) + nsv)

        out['valid_bas'], out['valid_itr'], out['valid_nsv'] = bas, itr, nsv
        # --- OTIMIZAÇÃO: Passa a função 'f' para o método de teste para evitar recalcular ---
        out['test_bas'], out['test_itr'], out['test_nsv'] = self._evaluate_on_test_set(f)
    
    @ignore_warnings(category=ConvergenceWarning)
    def _evaluate_on_test_set(self, f):
        # --- OTIMIZAÇÃO: Renomeado de _evaluate2 para maior clareza ---
        # Este método agora usa o conjunto de treino (Xn, yn) e teste (X_test, y_test)
        # definidos no construtor, eliminando a redundância de carregar e dividir os dados novamente.
        
        rs = StandardScaler()
        # Treina o scaler no conjunto de treino completo
        X0s = rs.fit_transform(self.Xn)
        # Aplica a transformação no conjunto de teste
        X1s = rs.transform(self.X_test)
        
        pd00 = pairwise_distances(X0s, X0s)
        G0 = f(pd00) # train gram matrix
        pd10 = pairwise_distances(X1s, X0s)
        G1 = f(pd10) # test gram matrix
        
        svc = SVC(C=self.C, kernel='precomputed', max_iter=SVC_MAX_ITER)
        # Treina no conjunto de treino completo
        svc.fit(G0, self.yn)
        # Prediz no conjunto de teste
        yp = svc.predict(G1)
        
        return balanced_accuracy_score(y_true=self.y_test, y_pred=yp), np.mean(svc.n_iter_), svc.support_.shape[0] / X0s.shape[0]


def job2bdone(p):
    C, n_coefs, dset, n_splits, test_size, seed, pop_size, n_gen = p
                    
    save_dir = os.path.join(f'{PREFIX}rkf-{n_splits}-{test_size}-{pop_size}-{n_gen}-v2', dset, str(C), str(n_coefs))
    hist_path = os.path.join(save_dir, f'hist-{seed}.joblib')
    
    if os.path.exists(hist_path):
        return 'already done'
    
    rkfp = RKFP(
        C=C, n_coefs=n_coefs, dset=dset,
        n_splits=n_splits, test_size=test_size, seed=seed
    )
    algo = DE(pop_size=pop_size, variant='DE/rand/1/bin', CR=0.9, F=0.8, dither='vector', jitter=False)
    resp = minimize(rkfp, algo, termination=('n_gen', n_gen), verbose=False, save_history=True)
    
    print(f'done path {save_dir}', f'execution time: {resp.exec_time}, best X: {resp.X}, best F: {resp.F}')
    
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(resp.history, hist_path)
    
    return 'done'


def main():
    products = product(CS, NS_COEFS, DSETS, NS_SPLITS, TEST_SIZES, SEEDS, POPS_SIZE, NS_GEN)

    results = None
    # --- OTIMIZAÇÃO: Definido 'maxtasksperchild=1' no Pool ---
    # Isso faz com que um processo de trabalho seja reiniciado após completar uma tarefa,
    # liberando memória e evitando potenciais memory leaks em execuções muito longas.
    with Pool(processes=os.cpu_count(), maxtasksperchild=1) as pool:
        results = pool.map(job2bdone, products)

    print(results)


if __name__ == '__main__':
    main()
