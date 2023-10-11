import locale
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from configs import SVC_MAX_ITER


locale.setlocale(locale.LC_ALL, "pt_BR.utf8")

plt.rcParams.update({
    'axes.formatter.use_locale' : True,
})

plt.style.use('classic')

def get_figsize(
    columnwidth=4, wf=1.0, hf_rel=(5.0 ** 0.5 - 1.0) / 2.0, hf_abs=None, unit="inch"
):

    unit = unit.lower()
    conversion = dict(inch=1.0, mm=25.4, cm=2.54, pt=72.0,)

    if unit in conversion.keys():
        fig_width = columnwidth / conversion[unit]
        if hf_abs is not None:
            fig_height = hf_abs / conversion[unit]
    else:
        raise ValueError(f"unit deve ser: {conversion.keys()}")

    fig_width *= wf

    if hf_abs is None:
        fig_height = fig_width * hf_rel

    return (fig_width, fig_height)

plt.rcParams.update({
    'figure.figsize' : get_figsize(columnwidth=455.0, unit='pt'),
    "axes.labelsize": 16,
    "font.size": 16,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

n2print = 5

rkf = pd.read_csv('basrkf-5-0.2-30-40.csv', sep=';')
rkf['X_valid_itr'] = rkf['X_valid_itr'].apply(lambda x: x / SVC_MAX_ITER )
rkf['X_test_itr'] = rkf['X_test_itr'].apply(lambda x: x / SVC_MAX_ITER )

rbf = pd.read_csv('basrbf-5-0.2-30-40.csv', sep=';')
rbf['valid_itr'] = rbf['valid_itr'].apply(lambda x: x / SVC_MAX_ITER )
rbf['test_itr'] = rbf['test_itr'].apply(lambda x: x / SVC_MAX_ITER )

rbfg = rbf.groupby(['dataset', 'C', 'gamma'], as_index=False).mean().sort_values(by='valid_bas', ascending=False)[:n2print]
print(rbfg)
print()

rkfg = rkf.groupby(['dataset', 'C', 'n_coefs'], as_index=False).mean().sort_values(by='X_valid_bas', ascending=False)[:n2print]
print(rkfg)
print()


dataset = 'pima'

rbfg_r = rbfg.iloc[0]
best_C_rbf, best_gamma_rbf = rbfg_r.C, rbfg_r.gamma

rkfg_r = rkfg.iloc[0]
best_C_rkf, best_n_coefs_rkf = rkfg_r.C, rkfg_r.n_coefs

best_rbf_bas_valid_values = rbf.loc[(rbf['dataset'] == dataset) & (rbf['C'] == best_C_rbf) & (rbf['gamma'] == best_gamma_rbf)]['valid_bas'].values
best_rbf_itr_valid_values = rbf.loc[(rbf['dataset'] == dataset) & (rbf['C'] == best_C_rbf) & (rbf['gamma'] == best_gamma_rbf)]['valid_itr'].values
best_rbf_nsv_valid_values = rbf.loc[(rbf['dataset'] == dataset) & (rbf['C'] == best_C_rbf) & (rbf['gamma'] == best_gamma_rbf)]['valid_nsv'].values

best_rkf_bas_valid_values = rkf.loc[(rkf['dataset'] == dataset) & (rkf['C'] == best_C_rkf) & (rkf['n_coefs'] == best_n_coefs_rkf)]['X_valid_bas'].values
best_rkf_itr_valid_values = rkf.loc[(rkf['dataset'] == dataset) & (rkf['C'] == best_C_rkf) & (rkf['n_coefs'] == best_n_coefs_rkf)]['X_valid_itr'].values
best_rkf_nsv_valid_values = rkf.loc[(rkf['dataset'] == dataset) & (rkf['C'] == best_C_rkf) & (rkf['n_coefs'] == best_n_coefs_rkf)]['X_valid_nsv'].values

best_rbf_bas_test_values = rbf.loc[(rbf['dataset'] == dataset) & (rbf['C'] == best_C_rbf) & (rbf['gamma'] == best_gamma_rbf)]['test_bas'].values
best_rbf_itr_test_values = rbf.loc[(rbf['dataset'] == dataset) & (rbf['C'] == best_C_rbf) & (rbf['gamma'] == best_gamma_rbf)]['test_itr'].values
best_rbf_nsv_test_values = rbf.loc[(rbf['dataset'] == dataset) & (rbf['C'] == best_C_rbf) & (rbf['gamma'] == best_gamma_rbf)]['test_nsv'].values

best_rkf_bas_test_values = rkf.loc[(rkf['dataset'] == dataset) & (rkf['C'] == best_C_rkf) & (rkf['n_coefs'] == best_n_coefs_rkf)]['X_test_bas'].values
best_rkf_itr_test_values = rkf.loc[(rkf['dataset'] == dataset) & (rkf['C'] == best_C_rkf) & (rkf['n_coefs'] == best_n_coefs_rkf)]['X_test_itr'].values
best_rkf_nsv_test_values = rkf.loc[(rkf['dataset'] == dataset) & (rkf['C'] == best_C_rkf) & (rkf['n_coefs'] == best_n_coefs_rkf)]['X_test_nsv'].values

def plot_double_boxplot(label1, data1, label2, data2, filename, xlabel, ylabel, loc=None):
    fig, ax = plt.subplots()

    # Create a list of data to plot
    data_to_plot = [data1, data2]

    ax.boxplot(data_to_plot, labels=[label1, label2])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f'{filename}.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
    plt.clf()

def plot_double_hist(label1, data1, label2, data2, filename, xlabel, ylabel, loc=None):

    plt.hist(data1, label=label1, bins=8, alpha=.7, color='red', edgecolor='black')
    plt.hist(data2, label=label2, bins=8, alpha=.7, color='yellow', edgecolor='black')
    plt.legend(loc=loc)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f'{filename}.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
    plt.clf()

def perform_shapiro_wilk_test(label, data, alpha=0.05):

    # Perform Shapiro-Wilk test
    statistic, p_value = stats.shapiro(data)

    print(f"{label} Shapiro statistic: {statistic}")
    print(f"{label} P-value: {p_value}")

    # Check p-value against alpha
    if p_value > alpha:
        print(f'Data {label} appears to be normally distributed (fail to reject H0)')
    else:
        print(f'Data {label} does not appear to be normally distributed (reject H0)')

    print()

def perform_paired_t_test(label1, data1, label2, data2, alpha=0.05):

    # Perform the paired t-test
    statistic, p_value = stats.ttest_rel(data1, data2)

    print(f'Paired t-test Statistic: {statistic}, p-value: {p_value}')

    # Interpret the result
    alpha = 0.05
    if p_value > alpha:
        print(f'{label1} and {label2} appears to have the same distribution (fail to reject H0)')
    else:
        print(f'{label1} and {label2} appears to have different distributions (reject H0)')
    
    print()

def perform_wilcoxon_signed_rank_test(label1, data1, label2, data2, alpha=0.05):

    # Perform wilcoxon
    statistic, p_value = stats.wilcoxon(data1, data2)

    print(f'Wilcoxon signed-rank test Statistic: {statistic}, p-value: {p_value}')

    if p_value < alpha:
        print(f"The difference between '{label1}' and '{label2}' measurements is statistically significant.")
    else:
        print(f"There is no statistically significant difference between '{label1}' and '{label2}' measurements.")
    
    print()

plot_double_hist('RBF', best_rbf_bas_valid_values, 'Racional', best_rkf_bas_valid_values, 'treino-bas', xlabel='Acurácia balanceada', ylabel='Frequência', loc='best')
perform_shapiro_wilk_test('RBF BAStr', best_rbf_bas_valid_values, alpha=0.05)
perform_shapiro_wilk_test('Racional BAStr', best_rkf_bas_valid_values, alpha=0.05)
perform_paired_t_test('RBF BAStr', best_rbf_bas_valid_values, 'Racional BAStr', best_rkf_bas_valid_values, alpha=0.05)

plot_double_boxplot('RBF', best_rbf_itr_valid_values, 'Racional', best_rkf_itr_valid_values, 'treino-itr', xlabel='Abordagem', ylabel='Porcentagem')
perform_shapiro_wilk_test('RBF ITRtr', best_rbf_itr_valid_values, alpha=0.05)
perform_shapiro_wilk_test('Racional ITRtr', best_rkf_itr_valid_values, alpha=0.05)
perform_wilcoxon_signed_rank_test('RBF ITRtr', best_rbf_itr_valid_values, 'Racional ITRtr', best_rkf_itr_valid_values, alpha=0.05)

plot_double_boxplot('RBF', best_rbf_nsv_valid_values, 'Racional', best_rkf_nsv_valid_values, 'treino-nsv', xlabel='Abordagem', ylabel='Porcentagem')
perform_shapiro_wilk_test('RBF NSVtr', best_rbf_nsv_valid_values, alpha=0.05)
perform_shapiro_wilk_test('Racional NSVtr', best_rkf_nsv_valid_values, alpha=0.05)
perform_wilcoxon_signed_rank_test('RBF NSVtr', best_rbf_nsv_valid_values, 'Racional NSVtr', best_rkf_nsv_valid_values, alpha=0.05)

plot_double_hist('RBF', best_rbf_bas_test_values, 'Racional', best_rkf_bas_test_values, 'teste-bas', xlabel='Acurácia balanceada', ylabel='Frequência')
perform_shapiro_wilk_test('RBF BASte', best_rbf_bas_test_values, alpha=0.05)
perform_shapiro_wilk_test('Racional BASte', best_rkf_bas_test_values, alpha=0.05)
perform_wilcoxon_signed_rank_test('RBF BASte', best_rbf_bas_test_values, 'Racional BASte', best_rkf_bas_test_values, alpha=0.05)

plot_double_boxplot('RBF', best_rbf_itr_test_values, 'Racional', best_rkf_itr_test_values, 'teste-itr', xlabel='Abordagem', ylabel='Porcentagem')
perform_shapiro_wilk_test('RBF ITRte', best_rbf_itr_test_values, alpha=0.05)
perform_shapiro_wilk_test('Racional ITRte', best_rkf_itr_test_values, alpha=0.05)
perform_wilcoxon_signed_rank_test('RBF ITRte', best_rbf_itr_test_values, 'Racional ITRte', best_rkf_itr_test_values, alpha=0.05)

plot_double_boxplot('RBF', best_rbf_nsv_test_values, 'Racional', best_rkf_nsv_test_values, 'teste-nsv', xlabel='Abordagem', ylabel='Porcentagem')
perform_shapiro_wilk_test('RBF NSVte', best_rbf_nsv_test_values, alpha=0.05)
perform_shapiro_wilk_test('Racional NSVte', best_rkf_nsv_test_values, alpha=0.05)
perform_wilcoxon_signed_rank_test('RBF NSVte', best_rbf_nsv_test_values, 'Racional NSVte', best_rkf_nsv_test_values, alpha=0.05)