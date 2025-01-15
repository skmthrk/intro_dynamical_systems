import os, time, math

import numpy as np
from numpy import eye, zeros, ones, sqrt, diag, exp, pi, log, isinf, isnan, kron, inf, nan
import scipy
from scipy.linalg import expm, inv, det
import numdifftools as nd
import pybobyqa

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import multiprocessing

# constants
SAMPLE_SIZE = 250

class KalmanFilter(object):

    def __init__(self, dim_x, dim_y, dim_u=0, x0=None, P0=None):

        self.dim_x = dim_x # dimension of state variables
        self.dim_y = dim_y # dimension of measurement variables
        self.dim_u = dim_u # dimension of control variables (defaulted at 0)

        # initial state estimate, x ~ N(x0, P0)
        # use diffuse prior unless explicitly provided
        self.kappa = 1e+7 # parameter for diffuse prior
        self.x0 = x0 if x0 else zeros((dim_x, 1))
        self.P0 = P0 if P0 else self.kappa * eye(dim_x)

        # placeholder for measurement
        self.Y = []

        # favorite inverse/determinant operation 
        self.inv = inv
        self.det = det
    
        # clear up model parameters
        self._reset_model()

    def _reset_model(self):

        dim_x = self.dim_x
        dim_y = self.dim_y
        dim_u = self.dim_u

        # initialize state estimate, x ~ N(x,P)
        self.x = self.x0
        self.P = self.P0

        # placeholder for state transition equations: x' = Ax + v + nu where v = Bu and nu ~ N(0,V)
        self.A = eye(dim_x)
        self.B = zeros((dim_x, dim_u))
        self.v = self.B @ zeros((dim_u, 1))
        self.V = eye(dim_x)

        # placeholder for measurement equations: y' = Cx' + w + omega, omega ~ N(0,W)
        self.C = zeros((dim_y, dim_x))
        self.w = zeros((dim_y, 1))
        self.W = eye(dim_y)

        # placeholder for prior on next state x'|y ~ N(x_prior, P_prior)
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # placeholder for forecast on measurement y'|y ~ N(y,Q)
        self.y = zeros((dim_y, 1))
        self.Q = eye(dim_y)
        self.Qinv = self.inv(self.Q)

        # placeholder for Kalman gain K and prediction error q
        self.K = zeros((dim_x, dim_y))
        self.q = zeros((dim_y, 1))

        # placeholder for posterior x'|y' ~ N(x_post, P_post)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict(self, A, B, u, V, C, w, W):
        """
        Calculate prior x'|y and forecast y'|y
        """
        #
        # compute state prior x_{t}|Y_{t-1} ~ N(x_{t|t-1}, P_{t|t-1})
        #
        # - self.x: posterior mean from previous period
        # - self.P: posterior covariance from previous period

        # prior mean: x_{t|t-1} = A_{t}x_{t-1|t-1} + v_{t}
        self.x_prior = A @ self.x + B @ u

        # prior covariance: P_{t|t-1} = AP_{t-1|t-1}A.T + V_{t}
        self.P_prior = A @ self.P @ A.T + V

        #
        # compute forecast y_{t}|Y_{t-1} ~ N(y_{t|t-1}, Q_{t|t-1})
        #

        # mean: y_{t|t-1} = C_{t}x_{t|t-1} + w_{t}
        self.y = C @ self.x_prior + w

        # covariance: Q_{t|t-1} = C_{t}P_{t|t-1}C_{t}.T + W_{t}
        self.Q = C @ self.P_prior @ C.T + W
        self.Qinv = self.inv(self.Q)

        # Kalman gain: K_{t} = P_{t|t-1} * C_{t}.T * Q_{t|t-1}^{-1}
        self.K = self.P_prior @ C.T @ self.Qinv

    def update(self, y):
        """
        Calculate posterior x'|y'
        - y in the input is observed value for y'
        """
        # Kalman gain calculated in the self.predict() step
        K = self.K

        #
        # compute posterior x_{t}|Y_{t} ~ N(x_{t|t}, P_{t|t})
        #

        # prediction error
        self.q = y - self.y

        # posterior mean: x_{t|t} = x_{t|t-1} + K_{t}q_{t}
        self.x = self.x_prior + K @ self.q

        # posterior covariance: P_{t|t} = P_{t|t-1} - KQ_{t|t-1}K.T
        self.P = self.P_prior - K @ self.Q @ K.T

        # save
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def Pt(self):
        """
        Calculate Prob(y_{t}|Y_{t-1})
        """
        #
        value = exp(-1/2 * (self.q.T @ self.Qinv @ self.q)[0][0])
        value = value / (2*pi)**(self.dim_y/2)
        value = value / (abs(self.det(self.Q)))**(1/2)
        return value

    def log_Pt(self, checkError=True):
        """
        Calculate ln(Prob(y_{t}|Y_{t-1}))
        """
        #
        val = log(self.Pt())

        if checkError and (isinf(val) or isnan(val)):
            raise ValueError("log_Pt gives an invalid value (inf or nan).")

        return val

    def log_likelihood(self, A, B, u, V, C, w, W, verbose=True, checkError=True):

        # reset model (clear x, y, A, B, u, V, C, w, W previously computed if any)
        # use x0, P0 stored at self.x0, self.P0
        self._reset_model()

        # load measurement data
        Y = self.Y
        
        value = 0 # initialize output value
        for y in Y:

            # set self.y, self.x_prior, self.Q, self.Qinv, and self.K
            self.predict(A, B, u, V, C, w, W) 

            # set self.x, self.P, self.y, self.q
            self.update(y) 
        
            # compute Pt and add log(Pt) to output value
            value += self.log_Pt(checkError)

        # to track changes in output value
        if verbose:
            print(value)

        return value


class Model(object):

    def __init__(self):

        # kalman filter instance
        dim_x, dim_y, dim_u  = 3, 2, 1
        self.kf = KalmanFilter(dim_x, dim_y, dim_u)

        # True parameter values
        a11 = 0.150
        a21 = 0.050
        a22 = 0.575
        a23 = 0.200
        a32 = 0.005
        a33 = 0.990
        b = 8
        sigma1 = 0.375
        sigma2 = 0.150
        sigma3 = 0.050
        self.parameters_true = [a11, a21, a22, a23, a32, a33, b, sigma1, sigma2, sigma3]

        #self.optimization_methods = ['BFGS', 'Powell', 'COBYQA', 'BOBYQA', 'SLSQP', 'Nelder-Mead']
        self.optimization_methods = ['BFGS', 'SLSQP', 'Nelder-Mead']

        # optimization results
        self.results = {}

        self.output_dir = 'output_ex2'

    def build_matrices(self, parameters):
        
        a11, a21, a22, a23, a32, a33, b, sigma1, sigma2, sigma3 = parameters

        # A, B, V
        A = np.array(
            [[a11, 0.0, 0.0],
             [a21, a22, a23],
             [0.0, a32, a33]])
        m = A.shape[0]
        B = zeros((m,1))
        B[0] = b
        V = zeros((m, m))
        V[0,0] = sigma1**2
        V[1,1] = sigma2**2
        V[2,2] = sigma3**2

        # x0, P0
        u0 = zeros((1,1)) # no external forcing up until t=0
        x0 = np.linalg.solve((eye(m) - A), B @ u0)
        vecP0 = np.linalg.solve(eye(m*m) - kron(A, A), V.ravel(order='F'))
        P0 = vecP0.reshape(m, m, order='F')

        # C, W
        C = np.array(
            [[0.0, 1.0, 0.0],
             [1.0, 1.0, 1.0]])
        W = eye(C.shape[0])*1e-12 # no measurement errors

        # u, v, w
        u = eye(1)
        v = B * u
        w = zeros((W.shape[0],1))

        return A, B, u, V, C, w, W, x0, P0

    def generate_sample(self, n, seed=None):

        if seed is not None:
            np.random.seed(seed=seed)

        A, B, u, V, C, w, W, x0, P0 = self.build_matrices(self.parameters_true)
       
        # draw initial state x0 from N(x0, P0)
        P0_sqrt = np.linalg.cholesky(P0)
        x0 = x0 + P0_sqrt @ np.random.randn(P0_sqrt.shape[-1]).reshape(-1,1)
    
        # draw state disturbance nu from N(0, V)
        V_sqrt = np.linalg.cholesky(V)
        Nu = [V_sqrt @ np.random.randn(V_sqrt.shape[-1]).reshape(-1,1) for _ in range(n)]
    
        # draw measurement error omega from N(0, W)
        W_sqrt = np.linalg.cholesky(W)
        Omega = [W_sqrt @ np.random.randn(W_sqrt.shape[-1]).reshape(-1,1) for _ in range(n)]

        # generate a sample
        X, Y = [], []
        x_prev = x0
        for nu, omega in zip(Nu, Omega):
    
            # state transition
            x = A @ x_prev + B @ u + nu
            X.append(x)
    
            # measurement
            y = C @ x + w + omega
            Y.append(y)
    
            # update previous state
            x_prev = x

        self.kf.Y = Y

    def objfun(self, input_values, log_input=True, verbose=False, checkError=True):
        '''
        objective function to minimize (negative of log likelihood)
        '''
        if log_input:
            # if input is log(parameters)
            parameters = exp(input_values)
        else:
            parameters = input_values
        A, B, u, V, C, w, W, x0, P0 = self.build_matrices(parameters)

        # initial state distribution x ~ N(x0, P0)
        self.kf.x0, self.kf.P0 = x0, P0

        return -self.kf.log_likelihood(A, B, u, V, C, w, W, verbose, checkError)

    def estimate(self, verbose=False):

        tol = 1e-5
        maxiter = 3000
        num_attempts = 10
        methods = self.optimization_methods

        # initial guess
        parameters = self.parameters_true
    
        # initialize best fvalue and best_parameters
        best_fvalue = inf
        best_parameters = parameters

        results = {} # store result for each method

        for attempt in range(num_attempts):

            fvalue = inf

            print(f"### Attempt {attempt+1}/{num_attempts} ###\n")

            # randomly selecting initial guess around the current best estimate
            print('Searching for a good initial guess...\n')
            scale = attempt/10 if attempt else 1.0e-3
            for _ in range(250):
                log_parameters_candidate = log(best_parameters) + np.random.randn(10) * scale
                try:
                    fvalue0 = self.objfun(log_parameters_candidate, verbose=False)
                except Exception as e:
                    continue
                if fvalue0 < fvalue:
                    parameters = exp(log_parameters_candidate)
                    fvalue = fvalue0

            bounds = [(None, None) for _ in parameters]
    
            # find the minimizing point using multiple methods
            for method in methods:
                success = False
                try:
                    print(f"Solving for MLE using {method}...")
                    if method == 'BOBYQA':
                        lower = []
                        upper = []
                        bounds_bobyqa = [(log(1.0e-4), log(1.0e+4)) for _ in parameters]
                        for bound in bounds_bobyqa:
                            lower.append(bound[0])
                            upper.append(bound[1])
                        seek_global_minimum = False # takes longer time if True
                        start_time = time.time()
                        res = pybobyqa.solve(objfun=self.objfun, x0=log(parameters), maxfun=maxiter, bounds=(lower, upper), scaling_within_bounds=True, seek_global_minimum=seek_global_minimum)
                        elapsed_time = time.time() - start_time
                        success = res.flag == res.EXIT_SUCCESS
                        fvalue = res.f
                        message = res.msg
                        num_iter = res.nf
                        parameters = exp(res.x)
                    else:
                        start_time = time.time()
                        options = {
                            'maxiter': maxiter,
                        }
                        if method == 'Nelder-Mead':
                            options['adaptive'] = True
                        if method == 'BFGS':
                            options['gtol'] = 1.0e-3
                        if method == 'COBYQA':
                            options['scale'] = True
                        res = scipy.optimize.minimize(fun=self.objfun, x0=log(parameters), method=method, bounds=bounds, tol=tol, options=options)
                        elapsed_time = time.time() - start_time
                        success = res.success
                        fvalue = res.fun
                        message = res.message
                        num_iter = res.nit

                except Exception as e:
                    print(f"===> Error in {method}: {e}\n")
                    continue
            

                status = 'Success' if success else 'Failure'
                print(f"===> {status} in {elapsed_time:.3f} seconds: {message} ({num_iter} iterations)")

                if success:

                    parameters = exp(res.x)

                    # compute 95% confidence intervals
                    if method == 'BFGS':
                        hess_inv = res.hess_inv
                        # get the standard errors of parameters from standard errors of log(parameters)
                        jacobian = diag(parameters) # Jacobian of the exp(parameters) evaluated at the MLE value
                        covariance = jacobian @ hess_inv @ jacobian.T # covariance of MLE parameters
                        std_errs = sqrt(diag(covariance))
                        confidence_intvls = 1.959 * std_errs # 95% interval
                    else: # need to compute hess_inv
                        try:
                            print(".... numerically computing hess_inv...")
                            hessian = nd.Hessian(lambda parameters: self.objfun(parameters, log_input=False, checkError=False))
                            hess_inv = inv(hessian(parameters))
                            covariance = hess_inv # covariance of MLE parameters
                            std_errs = sqrt(diag(covariance))
                            confidence_intvls = 1.959 * std_errs # 95% interval
                        except Exception as e:
                            print(f".... failed to compute std_errs: {e})")
                            confidence_intvls = nan * ones(len(res.x))

                    # update the best estimate for a given method
                    if (method not in results) or (fvalue < results[method]['fvalue']):
                        results[method] = {
                            'attempt': attempt,
                            'elapsed_time': elapsed_time,
                            'parameters': parameters,
                            'fvalue': fvalue,
                            'message': message,
                            'status': status,
                            'confidence_intvls': confidence_intvls,
                            'res': res,
                        }
                        print(f'===> Best estimate for {method} updated')
        
                    # update the best estimate among all methods
                    if fvalue < best_fvalue:
                        print(f'===> Best estimate updated')
                        best_method = method
                        best_parameters = parameters # update min_parameters
                        best_fvalue = fvalue # update min_fvalue
                print()
    
            if set(results.keys()) == set(methods):
                break

        self.results = results
        print('=== Summary ===\n')
        print(f" - sample size n: {len(self.kf.Y)}")
        print()
        for method in results:
            fvalue = results[method]['fvalue']
            status = results[method]['status']
            attempt = results[method]['attempt']
            elapsed_time = results[method]['elapsed_time']
            message = results[method]['message']
            parameters = list(results[method]['parameters'])
            confidence_intvls = results[method]['confidence_intvls']
            if method == best_method:
                print(f"*** {method} (Best method)")
                self.res = results[method]
            else:
                print(f"*** {method}")
            print(f" - fvalue: {fvalue} (attempt {attempt+1})")
            print(f" - status: {status} in {elapsed_time} seconds")
            #print(f" - message: {message}")
            print(f" - estimated parameters (vs true parameter values):")
            for parameter, parameter_true, ce in zip(parameters, self.parameters_true, confidence_intvls):
                print(f"  {parameter:.4f} +-{ce:.4f} ({parameter_true:.4f})")
        
            print()

def run_optimization(seed, n=SAMPLE_SIZE):
    ''' Funciton called in the multi processing loop
    '''
    model = Model()
    model.generate_sample(n=n, seed=seed)
    model.estimate()
    output_dir = model.output_dir

    # save results
    for method, result  in model.results.items():
        fvalue = result['fvalue']
        status = result['status']
        attempt = result['attempt']
        elapsed_time = result['elapsed_time']
        message = result['message']
        parameters = list(result['parameters'])
        confidence_intvls = result['confidence_intvls']

        out = [
            f"{fvalue}",
            f"{status}",
            f"{message}",
        ]
        for para, ce in zip(parameters, confidence_intvls):
            out.append(f"{para:.4f},{ce:.4f}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(f"{output_dir}/n{n}_seed{seed}_{method}.csv", 'w') as f:
            f.write('\n'.join(out))

def plot_results(model, n):
    
    labels = [
        r'$\hat{a}_{11}$',
        r'$\hat{a}_{21}$',
        r'$\hat{a}_{22}$',
        r'$\hat{a}_{23}$',
        r'$\hat{a}_{32}$',
        r'$\hat{a}_{33}$',
        r'$\hat{b}$',
        r'$\hat{\sigma}_{1}$',
        r'$\hat{\sigma}_{2}$',
        r'$\hat{\sigma}_{3}$']
    
    data_dir = model.output_dir
    parameters_true = model.parameters_true
    
    seed_methods = {}
    for f in os.listdir(data_dir):
        if f.startswith(f"n{n}_") and f.endswith('.csv'):
            # e.g., n250_seed1_BFGS.csv
            fname, ext = os.path.splitext(f)
            seed = int(fname.split('_')[1][4:])
            method = fname.split('_')[-1]
            seed_methods.setdefault(seed, []).append(method)
    
    seeds = sorted(seed_methods.keys())
    data = []
    for seed in seeds:
        methods = seed_methods[seed]
        fvalue = np.inf
        for method in methods:
            fname = f"n{n}_seed{seed}_{method}.csv"
            with open(os.path.join(data_dir, fname), 'r') as f:
                lines = [line.strip() for line in f]
                if float(lines[0]) < fvalue:
                    parameters = [float(line.split(',')[0]) for line in lines[3:]]
                    ces = [float(line.split(',')[1]) for line in lines[3:]]
                    
        # exclude extreme outliers
        outlier = False
        for parameter, parameter_true in zip(parameters, parameters_true):
            deviation = abs((parameter - parameter_true)/parameter_true)
            if deviation > 100:
                outlier = True
        if outlier:
            print(f"Skipped: {seed} {parameters}")
            continue
        data.append(parameters)
    
    df = pd.DataFrame(data, columns=labels)
    print(df.describe())
    
    num_rows, num_cols = df.shape
    
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(num_cols, num_cols, wspace=0.10, hspace=0.10)
    fontsize = 12
    
    statistics = []
    
    for i in range(num_cols):
        for j in range(num_cols):
            ax = plt.Subplot(fig, gs[i, j])
            fig.add_subplot(ax)
            
            if i == j: # If same column (diagonal) plot histogram
                bins = 50
                freqs, bins, patches =ax.hist(df.iloc[:, i], edgecolor = 'white', bins=35)
                for patch in patches:
                    patch.set_linewidth(0.5)  # Set linewidth for each patch
                
                ax.spines['top'].set_visible(False)    # Removing spines from top
                ax.spines['right'].set_visible(False) # Removing spines from right
                ax.spines['left'].set_visible(False) # Removing spines from left
                true_value = parameters_true[i]
                ymax = max(freqs)*1.2
                ax.vlines(true_value, ymin=0, ymax=ymax, linestyle='solid', color='k')
                ax.text(true_value, ymax*1.05, f'{true_value}', ha='center', fontsize = fontsize)
                mean_val = df.iloc[:, j].mean()
                std_val = df.iloc[:, j].std()
                label = df.columns[j]
                statistics.append((label, mean_val, std_val))
            elif i > j : # if column is below diagonal plot scatter
                ax.scatter(df.iloc[:,j], df.iloc[:,i], edgecolor='w')
                for spine in ax.spines.values(): # Setting linewidth for all spines for scatter
                    spine.set_linewidth(0.3)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_frame_on(False) # remove the frame
    
            #Remove x and y axis labels for all cells except bottom and right end
            if i != num_cols-1:
                ax.set_xticks([])
            ax.set_yticks([])
    
            # Add labels on the first row and column
            if j == 0 and i != 0:
                ax.set_ylabel(df.columns[i], rotation = 0, ha='right', fontsize=fontsize)
                ax.yaxis.set_label_coords(-0.1,0.5)
            if i == num_cols-1:
                ax.set_xlabel(f'{df.columns[j]}\n', fontsize=fontsize)
            
            # Add x ticks at the bottom of figure
            if i == num_cols -1:
                ax.set_xticks(np.linspace(df.iloc[:,j].min(), df.iloc[:,j].max(), 5)) # 5 ticks
                ax.set_xticklabels(np.round(np.linspace(df.iloc[:,j].min(), df.iloc[:,j].max(), 5),2), fontsize = fontsize, rotation=45) # set labels
    
    ax = fig.add_subplot(1, 1, 1, facecolor='none')
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    x0 = 0.15
    fsize = 18
    ax.text(x0 + 0.615, 0.97, r"MLE$^{*}$", ha='center', fontsize=fsize)
    y = 0.94
    ax.text(x0 + 0.575, y, f"mean", ha='center', fontsize=fsize)
    ax.text(x0 + 0.650, y, f"(std)", ha='center', fontsize=fsize)
    ax.text(x0 + 0.735, y, "true", ha='center', fontsize=fsize)
    for idx, ((label, mean_val, std_val), true_value) in enumerate(zip(statistics, parameters_true)):
        y = 0.9*(1-(idx/(len(statistics)-1))) + 0.6*idx/(len(statistics)-1)
        ax.text(x0 + 0.500, y, f"{label}", ha='center', fontsize=fsize)
        ax.text(x0 + 0.600, y, f"{mean_val:5.3f}", ha='right', fontsize=fsize)
        ax.text(x0 + 0.680, y, f"({std_val:5.3f})", ha='right', fontsize=fsize)
        ax.text(x0 + 0.760, y, f"{true_value:5.3f}", ha='right', fontsize=fsize)
    ax.text(x0 + 0.490, 0.550, f"* based on {num_rows} simulated samples", ha='left', fontsize=fsize)
    ax.text(x0 + 0.490, 0.525, f"   (size of each sample is n = {n})", ha='left', fontsize=fsize)
    
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.06, top=0.98, wspace = 0.2, hspace = 0.2)
    filename = 'fig_state_space_ex2'
    fig.savefig(f"{filename}.png", dpi=300)
    fig.savefig(f"{filename}.svg")


if __name__ == "__main__":

    seeds = range(0, 1000)
    parallel_processing = False

    if parallel_processing:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(run_optimization, seeds)
    else:
        for seed in seeds:
            run_optimization(seed, n=SAMPLE_SIZE)

    model = Model()
    plot_results(model, n=SAMPLE_SIZE)
