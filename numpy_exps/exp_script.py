
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from methods import *
from loss import *

colors = sns.color_palette("colorblind")
colors = sns.color_palette("bright")
markers = ['o', '^', 'v', '<', '>', 's', 'D', 'd', 'p', 'h', 'H', '8', 'X', '*', '.', 'P', 'x', '+', '1', '2', '3', '4', '|', '_']
colors = sns.color_palette("bright", n_colors=len(markers))



def run_exp(prob_name, n, d, lambd_reg, cond_num, cons, epochs, trials, bs, lambd_list, dec_flag, imp_flag, figs, fs, save_flag, final):
    if prob_name == 'ridgereg':
        loss = RidgeRegression(n, d, lambd_reg, cond=cond_num, consistent=cons)
    if prob_name == 'leastsq':
        loss = LeastSquares(n, d, lambd_reg, cond=cond_num, consistent=cons)
    if prob_name == 'logreg':
        loss = LogisticRegression(n, d, lambd_reg, cond=cond_num)
    
    x0 = np.random.randn(loss.d)
    f0 = loss.func(x0, np.arange(loss.n))

    if loss.name in ["Ridge Regression", "Least Squares"] and cons and lambd_reg == 0:
        f_star = 0.0
    else:
        K = 20000
        name, f = GD(loss, K, x0)
        f_star = f[-1]

    mu = loss.mu
    if bs == n:
        L = loss.L
        print(f"{loss.name}: mu={loss.mu}, L={loss.L}")
    else:
        L = loss.L_max
        print(f"{loss.name}: mu={loss.mu}, L={loss.L}, L_avg={loss.L_avg}, L_max={loss.L_max}")
    
    A = (n-bs)*L/(bs*(n-1))
    B = (n*(bs-1))/(bs*(n-1))

    K = epochs * n
    record_f = np.arange(0, K+1, n)

    method_name = []
    f_hist = []

    for lambd in lambd_list:
        # rho = 0.5 * mu/(L*(mu+2*(B*mu+A)*(1-lambd)**2))
        # gamma = (mu-L*rho*(mu+2*(B*mu+A)*(1-lambd)**2))/(2*L*(B*mu+A)*(2*L**2*rho**2*(1-lambd)**2+1))
        if lambd == 1.0:
            rho = 1/(4*L)
        else:
            rho = min(1,mu/(2*(B*mu+A)*(1-lambd)**2))/(4*L)
        gamma = min(2,mu/(8*(B*mu+A)))/L
        name, ff = unifiedSAM(loss, trials, record_f, x0, gamma, rho, lambd, dec_flag, imp_flag, bs)
        method_name.append(name);f_hist.append(ff)
        print(f"lambd={lambd}, rho={rho}, gamma={gamma}")


    # Plotting
    fig, ax = plt.subplots(figsize=figs)
    scale = 1

    for j in range(len(lambd_list)):
        mean_f = np.mean((f_hist[j]-f_star)/(f0-f_star), 1)
        std_f = np.std(f_hist[j], 1)
        plt.fill_between(record_f, mean_f-scale*std_f, mean_f+scale*std_f, alpha=0.2, fc=colors[j])
        plt.plot(record_f, mean_f, color = colors[j], linewidth=2, label=r'$\lambda='+str(j/10)+r'$', markevery=np.linspace(0, len(record_f)-1, 5, dtype=int), marker=markers[j], markersize=7, markeredgewidth=1.5, markeredgecolor=[0,0,0,0.6])

    plt.yscale("log")
    plt.xscale("linear")
    plt.xlabel('Iterations', fontsize=30)
    plt.ylabel(r'$\frac{f(x^t)-f(x^*)}{f(x^0)-f(x^*)}$', fontsize=30)
    # plt.ylabel(r'$\frac{\ell(\theta^t)-\ell(\theta^*)}{\ell(\theta^0)-\ell(\theta^*)}$', fontsize=30)
    plt.legend(fontsize=fs)
    plt.grid(True)

    # Saving Name
    cond_name = ', A=Gaussian'
    cond_savename = '_A=G'
    if cond_num is not None:
        cond_name = ', cond(A)='+str(cond_num)
        cond_savename = '_A='+str(cond_num)

    cons_name = ''
    cons_savename = ''
    if cons:
        cons_name = ' (Consistent)'
        cons_savename = '_cons'

    dec_name = ''
    dec_savename = ''
    if dec_flag:
        dec_name = ' (Decreasing)'
        dec_savename = '_dec'

    imp_name = ''
    imp_savename = ''
    if imp_flag:
        imp_name = ' (Importance)'
        imp_savename = '_imp'
    
    lambdas_st = ''
    for lambd in lambd_list:
        lambdas_st += (str(lambd) + '_')

    title_name = loss.name+': n='+str(n)+', d='+str(d)+', lambda='+str(lambd_reg)+cond_name+cons_name+dec_name+imp_name
    save_name = loss.save_name+'_n='+str(n)+'_d='+str(d)+'_lambda='+str(lambd_reg)+cond_savename+cons_savename+'_epochs='+str(epochs)+'_trials='+str(trials)+'_bs='+str(bs)+'_lambdas='+lambdas_st+dec_savename+imp_savename
    if not final:
        plt.title(title_name)

    if save_flag:
        plt.savefig('figures/'+save_name+'.pdf', bbox_inches='tight')
    else:
        plt.show()
    
    print(save_name+" [DONE]")


def run_exp_(prob_name, n, d, lambd_reg, cond_num, cons, epochs, trials, lambd_list, dec_flag, imp_flag, figs, fs, save_flag, final):
    if prob_name == 'ridgereg':
        loss = RidgeRegression(n, d, lambd_reg, cond=cond_num, consistent=cons)
    if prob_name == 'ridgereg_':
        loss = RidgeRegression_(n, d, lambd_reg, mu=1.0, L=cond_num)
    if prob_name == 'leastsq':
        loss = LeastSquares(n, d, lambd_reg, cond=cond_num, consistent=cons)
    if prob_name == 'logreg':
        loss = LogisticRegression(n, d, lambd_reg, cond=cond_num)
    
    x0 = np.random.randn(loss.d)
    f0 = loss.func(x0, np.arange(loss.n))

    if loss.name in ["Ridge Regression", "Least Squares"] and cons and lambd_reg == 0:
        f_star = 0.0
    else:
        K = 20000
        name, f = GD(loss, K, x0)
        f_star = f[-1]
    print(f"{loss.name}: mu={loss.mu}, L={loss.L}, L_avg={loss.L_avg}, L_max={loss.L_max}")

    mu = loss.mu
    L = loss.L_max
    A = L
    B = 0
    rho_coeff = 0.5

    K = epochs * n
    record_f = np.arange(0, K+1, n)

    method_name = []
    f_hist = []
    for lambd in lambd_list:
        # rho = rho_coeff * mu/(L*(mu+2*(B*mu+A)*(1-lambd)**2))
        # gamma = (mu-L*rho*(mu+2*(B*mu+A)*(1-lambd)**2))/(2*L*(B*mu+A)*(2*L**2*rho**2*(1-lambd)**2+1))
        if lambd == 1.0:
            rho = 1/(4*L)
        else:
            rho = min(1,mu/(2*(B*mu+A)*(1-lambd)**2))/(4*L)
        gamma = min(2,mu/(8*(B*mu+A)))/L
        name, ff = unifiedSAM(loss, trials, record_f, x0, gamma, rho, lambd, False, False, 1)
        method_name.append(name);f_hist.append(ff)
        print(f"lambd={lambd}, rho={rho}, gamma={gamma}")
    
    if dec_flag:
        method_name_dec = []
        f_hist_dec = []
        for lambd in lambd_list:
            # rho = rho_coeff * mu/(L*(mu+2*(B*mu+A)*(1-lambd)**2))
            # gamma = (mu-L*rho*(mu+2*(B*mu+A)*(1-lambd)**2))/(2*L*(B*mu+A)*(2*L**2*rho**2*(1-lambd)**2+1))
            if lambd == 1.0:
                rho = 1/(4*L)
            else:
                rho = min(1,mu/(2*(B*mu+A)*(1-lambd)**2))/(4*L)
            gamma = min(2,mu/(8*(B*mu+A)))/L
            name, ff = unifiedSAM(loss, trials, record_f, x0, gamma, rho, lambd, True, False, 1)
            method_name_dec.append(name);f_hist_dec.append(ff)
            print(f"lambd={lambd}, rho={rho}, gamma={gamma} (Dec)")

    if imp_flag:
        A = loss.L_avg
        method_name_imp = []
        f_hist_imp = []
        for lambd in lambd_list:
            # rho = rho_coeff * mu/(L*(mu+2*(B*mu+A)*(1-lambd)**2))
            # gamma = (mu-L*rho*(mu+2*(B*mu+A)*(1-lambd)**2))/(2*L*(B*mu+A)*(2*L**2*rho**2*(1-lambd)**2+1))
            if lambd == 1.0:
                rho = 1/(4*L)
            else:
                rho = min(1,mu/(2*(B*mu+A)*(1-lambd)**2))/(4*L)
            gamma = min(2,mu/(8*(B*mu+A)))/L
            name, ff = unifiedSAM(loss, trials, record_f, x0, gamma, rho, lambd, False, True, 1)
            method_name_imp.append(name);f_hist_imp.append(ff)
            print(f"lambd={lambd}, rho={rho}, gamma={gamma} (Imp)")
    


    # Plotting
    fig, ax = plt.subplots(figsize=figs)
    scale = 1

    for j in range(len(lambd_list)):
        mean_f = np.mean((f_hist[j]-f_star)/(f0-f_star), 1)
        std_f = np.std(f_hist[j], 1)
        plt.fill_between(record_f, mean_f-scale*std_f, mean_f+scale*std_f, alpha=0.2, fc=colors[j])
        if dec_flag:
            plt.plot(record_f, mean_f, color = colors[j], linewidth=2, label='Constant Step-size', markevery=np.linspace(0, len(record_f)-1, 5, dtype=int), marker=markers[j], markersize=7, markeredgewidth=1.5, markeredgecolor=[0,0,0,0.6])
        if imp_flag:
            plt.plot(record_f, mean_f, color = colors[j], linewidth=2, label='Uniform Sampling', markevery=np.linspace(0, len(record_f)-1, 5, dtype=int), marker=markers[j], markersize=7, markeredgewidth=1.5, markeredgecolor=[0,0,0,0.6])
    
    if dec_flag:
        for j in range(len(lambd_list)):
            mean_f = np.mean((f_hist_dec[j]-f_star)/(f0-f_star), 1)
            std_f = np.std(f_hist_dec[j], 1)
            plt.fill_between(record_f, mean_f-scale*std_f, mean_f+scale*std_f, alpha=0.2, fc=colors[len(lambd_list)+j])
            plt.plot(record_f, mean_f, color = colors[len(lambd_list)+j], linewidth=2, label='Decreasing Step-size', markevery=np.linspace(0, len(record_f)-1, 5, dtype=int), marker=markers[j], markersize=7, markeredgewidth=1.5, markeredgecolor=[0,0,0,0.6])

    if imp_flag:
        for j in range(len(lambd_list)):
            mean_f = np.mean((f_hist_imp[j]-f_star)/(f0-f_star), 1)
            std_f = np.std(f_hist_imp[j], 1)
            plt.fill_between(record_f, mean_f-scale*std_f, mean_f+scale*std_f, alpha=0.2, fc=colors[len(lambd_list)+j])
            plt.plot(record_f, mean_f, color = colors[len(lambd_list)+j], linewidth=2, label='Importance Sampling', markevery=np.linspace(0, len(record_f)-1, 5, dtype=int), marker=markers[j+1], markersize=7, markeredgewidth=1.5, markeredgecolor=[0,0,0,0.6])

    plt.yscale("log")
    plt.xscale("linear")
    plt.xlabel('Iterations', fontsize=30)
    plt.ylabel(r'$\frac{f(x^t)-f(x^*)}{f(x^0)-f(x^*)}$', fontsize=30)
    plt.legend(fontsize=fs)
    plt.grid(True)

    # Saving Name
    cond_name = ', A=Gaussian'
    cond_savename = '_A=G'
    if cond_num is not None:
        cond_name = ', cond(A)='+str(cond_num)
        cond_savename = '_A='+str(cond_num)

    cons_name = ''
    cons_savename = ''
    if cons:
        cons_name = ' (Consistent)'
        cons_savename = '_cons'

    dec_name = ''
    dec_savename = ''
    if dec_flag:
        dec_name = ' (Decreasing)'
        dec_savename = '_dec'

    imp_name = ''
    imp_savename = ''
    if imp_flag:
        imp_name = ' (Importance)'
        imp_savename = '_imp'
    
    lambdas_st = ''
    for lambd in lambd_list:
        lambdas_st += (str(lambd) + '_')

    title_name = loss.name+': n='+str(n)+', d='+str(d)+', lambda='+str(lambd_reg)+cond_name+dec_name+imp_name
    save_name = loss.save_name+'_n='+str(n)+'_d='+str(d)+'_lambda='+str(lambd_reg)+cond_savename+cons_savename+'_epochs='+str(epochs)+'_trials='+str(trials)+'_lambdas='+lambdas_st+dec_savename+imp_savename
    if not final:
        plt.title(title_name)

    if save_flag:
        plt.savefig('figures/'+save_name+'.pdf', bbox_inches='tight')
    else:
        plt.show()

    print(save_name+" [DONE]")


# def run_exp_others(prob_name, n, d, lambd_reg, cond_num, cons, epochs, trials, lambd_list, figs, fs, save_flag, final):
#     if prob_name == 'ridgereg':
#         loss = RidgeRegression(n, d, lambd_reg, cond=cond_num, consistent=cons)
#     if prob_name == 'leastsq':
#         loss = LeastSquares(n, d, lambd_reg, cond=cond_num, consistent=cons)
#     if prob_name == 'logreg':
#         loss = LogisticRegression(n, d, lambd_reg, cond=cond_num)
    
#     x0 = np.random.randn(loss.d)
#     f0 = loss.func(x0, np.arange(loss.n))

#     if loss.name in ["Ridge Regression", "Least Squares"] and cons and lambd_reg == 0:
#         f_star = 0.0
#     else:
#         K = 20000
#         name, f = GD(loss, K, x0)
#         f_star = f[-1]
#     print(f"{loss.name}: mu={loss.mu}, L={loss.L}, L_avg={loss.L_avg}, L_max={loss.L_max}")

#     mu = loss.mu
#     L = loss.L_max
#     A = L
#     B = 0
#     rho_coeff = 0.5

#     K = epochs * n
#     record_f = np.arange(0, K+1, n)

#     method_name = []
#     f_hist = []
#     for lambd in lambd_list:
#         rho = rho_coeff * mu/(L*(mu+2*(B*mu+A)*(1-lambd)**2))
#         gamma = (mu-L*rho*(mu+2*(B*mu+A)*(1-lambd)**2))/(2*L*(B*mu+A)*(2*L**2*rho**2*(1-lambd)**2+1))
#         name, ff = unifiedSAM(loss, trials, record_f, x0, gamma, rho, lambd, False, False, 1)
#         method_name.append(name);f_hist.append(ff)
#         print(f"lambd={lambd}, rho={rho}, gamma={gamma}")
    
#     rho = 1/(L*K**(1/4))
#     gamma = 1/(L*K**(1/2))
#     name, ff = unifiedSAM(loss, trials, record_f, x0, gamma, rho, 0, False, False, 1)
#     method_name.append(name);f_hist.append(ff)
#     print(f"Andr: lambd={lambd}, rho={rho}, gamma={gamma}")


#     # Plotting
#     fig, ax = plt.subplots(figsize=figs)
#     scale = 1

#     for j in range(len(lambd_list)):
#         mean_f = np.mean((f_hist[j]-f_star)/(f0-f_star), 1)
#         std_f = np.std(f_hist[j], 1)
#         plt.fill_between(record_f, mean_f-scale*std_f, mean_f+scale*std_f, alpha=0.2, fc=colors[j])
#         plt.plot(record_f, mean_f, color = colors[j], linewidth=2, label=r'$\lambda='+str(lambd_list[j])+r'$', markevery=np.linspace(0, len(record_f)-1, 5, dtype=int), marker=markers[j], markersize=7, markeredgewidth=1.5, markeredgecolor=[0,0,0,0.6])

#     j += 1
#     mean_f = np.mean((f_hist[j]-f_star)/(f0-f_star), 1)
#     std_f = np.std(f_hist[j], 1)
#     plt.fill_between(record_f, mean_f-scale*std_f, mean_f+scale*std_f, alpha=0.2, fc=colors[j])
#     plt.plot(record_f, mean_f, color = colors[j], linewidth=2, label='Andr', markevery=np.linspace(0, len(record_f)-1, 5, dtype=int), marker=markers[j], markersize=7, markeredgewidth=1.5, markeredgecolor=[0,0,0,0.6])
    
#     plt.yscale("log")
#     plt.xscale("linear")
#     plt.xlabel('Iterations', fontsize=fs)
#     plt.ylabel(r'$\frac{f(x^k)-f(x^*)}{f(x^0)-f(x^*)}$', fontsize=fs)
#     plt.legend(fontsize=fs)

#     cond_name = ', A=Gaussian'
#     cond_savename = '_A=G'
#     if cond_num is not None:
#         cond_name = ', cond(A)='+str(cond_num)
#         cond_savename = '_A='+str(cond_num)

#     cons_name = ''
#     cons_savename = ''
#     if cons:
#         cons_name = ' (Consistent)'
#         cons_savename = '_cons'

#     title_name = loss.name+': n='+str(n)+', d='+str(d)+', lambda='+str(lambd_reg)+cond_name
#     save_name = loss.save_name+'_n='+str(n)+'_d='+str(d)+'_lambda='+str(lambd_reg)+cond_savename+cons_savename
#     if not final:
#         plt.title(title_name)

#     if save_flag:
#         plt.savefig('figures/'+save_name+'.pdf', bbox_inches='tight')
#     else:
#         plt.show()

#     print(save_name+" [DONE]")
