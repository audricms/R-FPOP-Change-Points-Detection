import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import t as student_t
from scipy import integrate
from scipy.stats import norm
from sklearn.metrics import adjusted_rand_score  
import matplotlib.pyplot as plt
import math
from typing import Callable, Tuple, List, Dict, Literal, Optional
from tqdm import tqdm
from statsmodels import robust 
import matplotlib.pyplot as plt
import pandas as pd 



QuadPiece = Tuple[float, float, float, float, float, int]
#quadpiece is a tuple containing:
#first float a = left bound of the interval
#second flat b: right bound of the interval
#3 other floats A,B,C = coefficients of the quadratic polynom on interval [a,b]: AX^2+BX=c 
#last int tau = index of last changepoint 


#algorithm 3 from pseudocode 
def rfpop_algorithm3_min_over_theta(Qt_pieces: List[QuadPiece]):
    best_val = float('inf')
    best_tau = 0
    for (a,b,A,B,C,tau) in Qt_pieces:
        # computing min of Q_t(theta) over segment (a,b]
        if abs(A) < 1e-16:
            #if the coefficient associated with x^2 of the quadratic function is =0 then the function is linear (B!=0) or constant (B=0)
            #hence in this case min value of Q_t(theta) is on the boundaries of the segment
            left = a + 1e-12 #because segment does not contain a 
            right = b
            vleft = A*left*left + B*left + C
            vright = A*right*right + B*right + C
            theta_star = left if vleft <= vright else right
        else: #if A!=0 then Q_t(theta) =A theta^2 + B theta + C on segment (a,b] 
            #so minimizer is -B/2A   (each quadratic is assumed convex, not concave so not a maximizer)
            theta_star = -B / (2.0 * A)
            #we need to ensure the minimizer is in (a,b]
            if theta_star <= a: 
                theta_star = a + 1e-12
            elif theta_star > b:
                theta_star = b
        val = A*theta_star*theta_star + B*theta_star + C
        if val < best_val:
            best_val = val
            best_tau = tau
    return float(best_val), int(best_tau)

#algorithm 2 from appendix D 
def rfpop_algorithm2_add_Qstar_and_gamma(Qstar_pieces: List[QuadPiece], 
                                         #Q_star est représenté sous forme [ (a1,b1,A1,B1,C1,tau1), (a2,b2,A2,B2,C2,tau2), ..., ]
                                         gamma_pieces: List[QuadPiece]):

    out: List[QuadPiece] = []
    i = 0
    j = 0
    while i < len(Qstar_pieces) and j < len(gamma_pieces):
        pa, pb, pA, pB, pC, p_tau = Qstar_pieces[i]
        ga, gb, gA, gB, gC, g_tau = gamma_pieces[j]

        #new interval 
        a = max(pa, ga)
        b = min(pb, gb)
        
        # sum of Q_star and gamma on the new interval
        newA = pA + gA
        newB = pB + gB
        newC = pC + gC
        # tau comes from Q* (p_tau) 
        out.append((a, b, newA, newB, newC, p_tau))
        # avancer indices selon bornes
        if abs(b - pb) < 1e-12:
            i += 1
        if abs(b - gb) < 1e-12:
            j += 1
        # to avoid infinite loop
        if (i < len(Qstar_pieces) and j < len(gamma_pieces)) and a >= b - 1e-14 and abs(b - Qstar_pieces[i][1]) > 1e-12 and abs(b - gamma_pieces[j][1]) > 1e-12:
            break
    # to do as they say on page 12 and merge identical neighbouring intervals
    if not out:
        return []
    merged: List[QuadPiece] = [out[0]]
    for pc in out[1:]:
        a,b,A,B,C,tau = pc
        ma,mb,mA,mB,mC,mtau = merged[-1]
        if (abs(A - mA) < 1e-14 and abs(B - mB) < 1e-14 and abs(C - mC) < 1e-9 and tau == mtau and abs(a - mb) < 1e-9):
            merged[-1] = (ma, b, mA, mB, mC, mtau)
        else:
            merged.append(pc)
    return merged



#algorithm 4

def rfpop_algorithm4_prune_compare_to_constant(Qt_pieces: List[QuadPiece],
                                               Qt_val: float, #Qt_val + beta = the constant C in appendix
                                               beta: float,
                                               t_index_for_new: int):
   
    thr = Qt_val + beta
    out: List[QuadPiece] = []

    #goes through the Nt intervals of Qt
    for (a,b,A,B,C,tau) in Qt_pieces:
        #finds the roots of Q_t(x) - C ie the roots of A x^2 + B x + (C - thr) = 0 on [a,b]
        # BEWARE: source of confuction: thr is the equivalent of the constant C in the pseudocode of the paper
        # C is the last coefficient of the quadratic function, not the constant C defined in the pseudocode 
        roots: List[float] = []
        if abs(A) < 1e-16:
            # if the quadratic function is linear or constant: B x + (C - thr) = 0
            if abs(B) > 1e-16:
                x = -(C - thr) / B
                #we need to ensure root x is in interval [a,b] otherwise we don't care about it 
                if x + 1e-12 >= a and x - 1e-12 <= b:
                    x_clamped = max(min(x, b), a) #to ensure the root is indeed within the interval and have no problems of numerical errors later: we need to sort the roots it could do weird things later if we don't do that
                    roots.append(x_clamped)
        else:
            #if the function is indeed quadratic we compute delta and roots 
            D = B*B - 4.0*A*(C - thr)
            if D >= -1e-14:
                D = max(D, 0.0)
                sqrtD = math.sqrt(D)
                x1 = (-B - sqrtD) / (2.0*A)
                x2 = (-B + sqrtD) / (2.0*A)
                for x in (x1, x2):
                    if x + 1e-12 >= a and x - 1e-12 <= b:
                        x_clamped = max(min(x, b), a)
                        # avoid duplicates if delta D = 0 we get 2 times the same root 
                        if not any(abs(x_clamped - r) < 1e-9 for r in roots):
                            roots.append(x_clamped)
        roots.sort()
        # create vector [a, r1, r2,  b]
        breaks = [a] + roots + [b]
        for k in range(len(breaks)-1):
            lo = breaks[k]
            hi = breaks[k+1]
            mid = (lo + hi) / 2.0
            #here we evaluate the polynom A*mid*mid + B*mid + C - thr on each subsegment [a,root_1], [root_1,root_2],[root_2,b]
            #on each of these subsegment this polynom has the same sign as each bound of a subsegment is a root 
            #so we can pick an arbitrary value like the middle to see the sign of the polynom on this subsegment
            val_mid = A*mid*mid + B*mid + C - thr 

            #if polynom is negative then Q_t(theta)<constant so we keep Q_t(theta) = A*theta^2 + B*theta + C
            if val_mid <= 1e-12:
                # keep Q_t on this subsegment
                out.append((lo, hi, A, B, C, tau))
            else:
                # keep constant thr in this subsegment and set a changepoint at t_index_for_new 
                out.append((lo, hi, 0.0, 0.0, thr, t_index_for_new))
    # fusionner adjacences égales
    if not out:
        return []
    merged: List[QuadPiece] = [out[0]]
    for pc in out[1:]:
        a,b,A,B,C,tau = pc
        ma,mb,mA,mB,mC,mtau = merged[-1]
        if (abs(A - mA) < 1e-14 and abs(B - mB) < 1e-14 and abs(C - mC) < 1e-9 and tau == mtau and abs(a - mb) < 1e-9):
            merged[-1] = (ma, b, mA, mB, mC, mtau)
        else:
            merged.append(pc)
    return merged



#algorithm 1
def rfpop_algorithm1_main(y: List[float], gamma_builder, beta: float):

    y_arr = np.asarray(y, dtype=float)
    n = len(y_arr)

    lo = float(np.min(y_arr)) - 1.0 #we add -1 and +1 to ensure there is no numeric problem with python approximations for foats 
    hi = float(np.max(y_arr)) + 1.0
    Qstar = [(lo, hi, 0.0, 0.0, 0.0, 0)]  # Q*_1
    cp_tau = [0] * n
    Qt_vals = [0.0] * n
    for t_idx in range(n):
        # Sub-routine 2 : Qt = Q*_t + gamma(y_t)
        gamma_pcs = gamma_builder(float(y_arr[t_idx]), t_idx)
        Qt_pcs = rfpop_algorithm2_add_Qstar_and_gamma(Qstar, gamma_pcs)
        # Sub-routine 3 : min over theta Qt
        Qt_val, tau_t = rfpop_algorithm3_min_over_theta(Qt_pcs)
        cp_tau[t_idx] = tau_t
        Qt_vals[t_idx] = Qt_val
        # Sub-routine 4 : compute Q*_{t+1}
        Qstar = rfpop_algorithm4_prune_compare_to_constant(Qt_pcs, Qt_val, beta, t_idx)
    return cp_tau, Qt_vals, Qstar



QuadPiece = Tuple[float, float, float, float, float, int]
INF = 1e18  # borne pratique pour "infty"
EPS = 1e-12

def gamma_builder_L2(y: float, tau_for_new: int) -> List[QuadPiece]:
    """
    gamma(y,theta) = (y - theta)^2 = theta^2 - 2*y*theta + y^2
    Single quadratic piece on (-INF, INF].
    """
    A = 1.0
    B = -2.0 * y
    C = y * y
    return [(-INF, INF, A, B, C, tau_for_new)]

def gamma_builder_biweight(y: float, K: float, tau_for_new: int) -> List[QuadPiece]:
    """ :
      gamma(y,theta) = (y-theta)^2   if |y-theta| <= K
                      = K^2         otherwise
    Can be defined as 3 different quadratic functions over 3 segments :
      (-INF, y-K]       : constant K^2
      (y-K, y+K]        : quadratic (y-theta)^2
      (y+K, INF)        : constant K^2
    """
    A_q = 1.0
    B_q = -2.0 * y
    C_q = y * y
    constC = float(K * K)
    return [
        (-INF, y - K, 0.0, 0.0, constC, tau_for_new),
        (y - K, y + K, A_q, B_q, C_q, tau_for_new),
        (y + K, INF, 0.0, 0.0, constC, tau_for_new)
    ]


def gamma_builder_huber(y: float, K: float, tau_for_new: int) -> List[QuadPiece]:
    """
    Huber loss with threshold K:
      gamma(y,theta) = (y-theta)^2                     if |y-theta| <= K
                      = 2*K*|y-theta| - K^2           if |y-theta| > K
    
    Can be defined as 3 quadratic functions over 3 segments  
    (-INF, y-K] :  2*K*(y-theta) - K^2 = -2K*theta + (2K*y - K^2)
    (y-K, y+K]: (y-theta)^2 = theta^2 - 2y theta + y^2
    (y+K, INF): 2*K*(theta-y) - K^2 = 2K*theta + (-2K*y - K^2)
    """
    # central quadratic
    A_q = 1.0
    B_q = -2.0 * y
    C_q = y * y
    # left linear: B_left * theta + C_left  where B_left = -2K, C_left = 2K*y - K^2
    B_left = -2.0 * K
    C_left = 2.0 * K * y - K * K
    # right linear: B_right * theta + C_right where B_right = 2K, C_right = -2K*y - K^2
    B_right = 2.0 * K
    C_right = -2.0 * K * y - K * K
    return [
        (-INF, y - K, 0.0, B_left, C_left, tau_for_new),   # linear = A=0
        (y - K, y + K, A_q, B_q, C_q, tau_for_new),       # quadratic
        (y + K, INF, 0.0, B_right, C_right, tau_for_new) # linear = A=0
    ]

def gamma_builder_L1(y: float, tau_for_new: int) -> List[QuadPiece]:
    """
    L1 loss: gamma(y,theta) = |y - theta|
    Piecewise-linear
      (-INF, y) :  y - theta = (-1)*theta + y
      (y, INF): theta - y = (+1)*theta + (-y)

    """
    # left piece: (-INF, y] -> -theta + y  => A=0, B=-1, C=y
    # right piece: (y, INF] -> theta - y  => A=0, B=1,  C=-y
    return [
        (-INF, y, 0.0, -1.0, float(y), tau_for_new),
        (y, INF, 0.0,  1.0, float(-y), tau_for_new)
    ]


def generate_scenarios():

    np.random.seed(42) 

    # Fonction auxiliaire pour le bruit t-student
    def t_noise(n, df=5, sigma=1.0):
        return np.random.standard_t(df, size=n) * sigma

    #scenario 1
    n1 = 2048
    t = np.arange(n1) / n1
    pos = np.array([0.1, 0.13, 0.15, 0.23, 0.25, 0.40, 0.44, 0.65, 0.76, 0.78, 0.81])
    h = np.array([4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2])
    sig1 = np.zeros(n1)
    for p, hi in zip(pos, h):
        sig1[int(p * n1):] += hi
    sig1 = sig1 * 3.5 
    y1 = sig1 + t_noise(n1, df=5, sigma=5.0) 

    #Scenario 2
    n2 = 512
    sig2 = np.zeros(n2)
    segments_fms = [
        (0, 200, -0.2), (200, 220, 0.2), (220, 240, 1.2), (240, 260, -0.8),
        (260, 280, 0.5), (280, 400, -0.1), (400, 420, 0.8), (420, 512, -0.3)
    ]
    for start, end, val in segments_fms:
        sig2[start:end] = val
    y2 = sig2 + t_noise(n2, df=5, sigma=0.3)


    # --- Scenario 2'
    # same signal but less noise
    y2_prime = sig2 + t_noise(n2, df=5, sigma=0.2)

    #Scenario 3
    # short and long segments
    n3 = 512
    sig3 = np.zeros(n3)
    curr = 0
    mu = 0
    #alternate randomly between long and short segments 
    while curr < n3:
        length = np.random.choice([10, 20, 80, 100])
        val = np.random.normal(0, 2)
        end = min(curr + length, n3)
        sig3[curr:end] = val
        curr = end

    y3 = sig3 + t_noise(n3, df=5, sigma=0.3)


    #Scenario 4 saw tooth 
    n4 = 240
    sig4 = np.zeros(n4)
    n_teeth = 10
    len_tooth = n4 // n_teeth 
    for i in range(n_teeth):
        if i % 2 == 1:
            sig4[i*len_tooth : (i+1)*len_tooth] = 1.0
    y4 = sig4 + t_noise(n4, df=5, sigma=0.3)

    #Scenario 5: Stairs 
    n5 = 150
    sig5 = np.zeros(n5)
    n_steps = 10
    len_step = n5 // n_steps 
    for i in range(n_steps):
        sig5[i*len_step : (i+1)*len_step] = (i + 1) * 1.5
    # Bruit
    y5 = sig5 + t_noise(n5, df=5, sigma=0.3)

    dic1 = {
        "Scenario 1": y1,
        "Scenario 2": y2,
        "Scenario 2'": y2_prime,
        "Scenario 3": y3,
        "Scenario 4": y4,
        "Scenario 5": y5}
    dic2 = {
        "Scenario 1": sig1,
        "Scenario 2": sig2,
        "Scenario 2'": sig2,
        "Scenario 3": sig3,
        "Scenario 4": sig4,
        "Scenario 5": sig5}
    return dic1,dic2


def biweight_phi(z, K_std):
    return 2*z if abs(z) <= K_std else 0.0

def huber_phi(z, K_std):
    return 2*z if abs(z) <= K_std else 2*K_std*np.sign(z)

def compute_penalty_beta(y, loss):
    ys = pd.Series(y)
    sigma = robust.mad(ys.diff().dropna())/np.sqrt(2)
    n = len(y)
    # return 2 * sigma**2 * np.log(n)

    if loss == 'l2':
        return 2 * sigma**2 * np.log(n)

    elif loss == 'biweight':
        K_std = 3.0  # K / sigma
        E_phi2, _ = integrate.quad(
            lambda z: (biweight_phi(z, K_std)**2) * norm.pdf(z),
            -np.inf, np.inf
        )
        return 2 * sigma**2 * np.log(n)   * E_phi2

    elif loss == 'huber':
        K_std = 1.345
        E_phi2, _ = integrate.quad(
            lambda z: (huber_phi(z, K_std)**2) * norm.pdf(z),
            -np.inf, np.inf
        )
        return 2 * sigma**2 * np.log(n) * E_phi2

    elif loss == 'l1':
        return np.log(n)
    

    # #function to compute parameter K
def compute_loss_bound_K(y,
             loss: Literal['huber','biweight']):
    ys =pd.Series(y)
    mad = robust.mad(ys.diff().dropna())/np.sqrt(2)
    if loss == "biweight":
        return 3*mad
    elif loss == 'huber':
        return 1.345*mad 

# =============================================================================
# CROSS-VALIDATION SUR DONNÉES RÉELLES
# =============================================================================
def cross_validate_rfpop(
    y: np.ndarray,
    loss: Literal['huber', 'biweight', 'l2', 'l1'] = 'biweight',
    beta_range: Optional[np.ndarray] = None,
    K_range: Optional[np.ndarray] = None,
    criterion: Literal['elbow', 'penalized_cost', 'bic'] = 'elbow',
    verbose: bool = True
) -> dict:
    """
    Cross-validation pour RFPOP avec critère "elbow" (stabilité).
    
    Le critère 'elbow' trouve la première région stable du nombre de changepoints
    après que le nombre soit descendu sous max/2. Cela évite l'oversegmentation.

    Parameters:
    -----------
    y : np.ndarray
        Série temporelle
    loss : str
        Type de loss: 'huber', 'biweight', 'l2', ou 'l1'
    beta_range : np.ndarray, optional
        Grille de valeurs pour β (multipliers of paper beta)
    K_range : np.ndarray, optional  
        Grille de valeurs pour K (multipliers of paper K)
    criterion : str
        'elbow' (trouve région stable), 'penalized_cost', ou 'bic'
    verbose : bool
        Affiche les informations de progression

    Returns:
    --------
    dict : Résultats incluant best_beta, best_K, changepoints, etc.
    """
    n = len(y)
    y_list = list(y)
    
    # Paramètres théoriques du papier
    ys = pd.Series(y)
    sigma_hat = robust.mad(ys.diff().dropna()) / np.sqrt(2)
    if sigma_hat == 0:
        sigma_hat = np.std(y)
    
    # Calculer les paramètres de référence
    if loss in ['huber', 'biweight']:
        K_paper = compute_loss_bound_K(y, loss)
        beta_paper = compute_penalty_beta(y, loss)
    else:  # l2 ou l1
        K_paper = None
        beta_paper = compute_penalty_beta(y, 'l2')
    
    # Définir les grilles de recherche ÉTENDUES
    if beta_range is None:
            beta_mults = np.array([0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 20, 50, 100, 1000])
            beta_range = beta_mults * beta_paper
    
    if loss in ['huber', 'biweight']:
        if K_range is None:
            K_mults = np.array([0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
            K_range = K_mults * K_paper
    else:
        K_range = [None]
    
    results = []
    
    if verbose:
        total = len(beta_range) * len(K_range)
        print(f"Cross-validation ({criterion}) pour {loss} loss")
        print(f"Exploration de {len(beta_range)} × {len(K_range)} = {total} combinaisons...")
    
    for beta in beta_range:
        for K in K_range:
            try:
                if loss == 'huber':
                    def make_gamma(K_val):
                        return lambda y_t, t: gamma_builder_huber(y_t, K_val, t)
                    gamma_builder = make_gamma(K)
                elif loss == 'biweight':
                    def make_gamma(K_val):
                        return lambda y_t, t: gamma_builder_biweight(y_t, K_val, t)
                    gamma_builder = make_gamma(K)
                elif loss == 'l2':
                    gamma_builder = lambda y_t, t: gamma_builder_L2(y_t, t)
                else:  # l1
                    gamma_builder = lambda y_t, t: gamma_builder_L1(y_t, t)
                
                cp_tau, Qt_vals, _ = rfpop_algorithm1_main(y_list, gamma_builder, beta)
                
                changepoints = list(set(cp_tau))
                if 0 in changepoints:
                    changepoints.remove(0)
                n_cp = len(changepoints)
                
                penalized_cost = Qt_vals[-1]
                
                results.append({
                    'beta': beta,
                    'beta_mult': beta / beta_paper,
                    'K': K,
                    'K_mult': K / K_paper if K else None,
                    'n_changepoints': n_cp,
                    'penalized_cost': penalized_cost,
                    'cp_tau': cp_tau,
                    'changepoints': sorted(changepoints)
                })
                
            except Exception as e:
                if verbose:
                    print(f"  Erreur: {e}")
    
    if not results:
        raise ValueError("Aucun résultat valide obtenu")
    
    # Sélection selon le critère
    if criterion == 'elbow':
        best = _find_elbow_point(results)
    elif criterion == 'bic':
        # BIC: minimiser cost + log(n) * n_cp
        for r in results:
            r['bic'] = r['penalized_cost'] + np.log(n) * r['n_changepoints']
        best = min(results, key=lambda x: x['bic'])
    else:  # penalized_cost
        best = min(results, key=lambda x: x['penalized_cost'])
    
    return {
        'best_beta': best['beta'],
        'best_K': best['K'],
        'best_ratio': best['beta'] / best['K'] if best['K'] else None,
        'best_n_cp': best['n_changepoints'],
        'changepoints': best['changepoints'],
        'cp_tau': best['cp_tau'],
        'all_results': results,
        'paper_params': {
            'beta': beta_paper,
            'K': K_paper,
            'sigma_hat': sigma_hat
        },
        'loss': loss,
        'criterion': criterion
    }


def _find_elbow_point(results):
    """
    Trouve le point "elbow" - la première région stable après la descente initiale.
    
    Algorithme:
    1. Trier par beta croissant
    2. Trouver max_ncp (nombre max de changepoints)
    3. Chercher la PREMIÈRE région stable (2+ points consécutifs avec même n_cp)
       où n_cp <= max_ncp/2 et n_cp > 0
    4. Si pas trouvé, prendre la première région stable avec n_cp > 0
    
    Cela évite de sélectionner des régions sur-segmentées (trop de CP) 
    ou sous-segmentées (trop peu de CP).
    """
    # Trier par beta croissant
    sorted_results = sorted(results, key=lambda x: x['beta'])
    n_cps = [r['n_changepoints'] for r in sorted_results]
    
    if not n_cps:
        return results[0]
    
    max_ncp = max(n_cps)
    threshold = max_ncp / 2
    
    # Chercher la première région stable sous le seuil
    for i in range(len(n_cps) - 1):
        n_cp = n_cps[i]
        # Vérifier: n_cp > 0, n_cp <= seuil, et stable (même que suivant)
        if n_cp > 0 and n_cp <= threshold and n_cps[i + 1] == n_cp:
            return sorted_results[i]
    
    # Fallback 1: première région stable avec n_cp > 0 (sans condition de seuil)
    for i in range(len(n_cps) - 1):
        if n_cps[i] > 0 and n_cps[i + 1] == n_cps[i]:
            return sorted_results[i]
    
    # Fallback 2: premier résultat avec n_cp > 0
    for i, r in enumerate(sorted_results):
        if r['n_changepoints'] > 0:
            return r
    
    # Fallback 3: premier résultat
    return sorted_results[0]


def extract_changepoints_backtrack(cp_tau):
    """Extrait les changepoints par backtracking."""
    n = len(cp_tau)
    changepoints = []
    t = n - 1
    
    while t > 0:
        tau = cp_tau[t]
        if tau > 0:
            changepoints.append(tau)
        t = tau
    
    changepoints.reverse()
    return changepoints


def get_segments_from_cp_tau(cp_tau, y):
    """Extrait les segments à partir de cp_tau."""
    n = len(cp_tau)
    segments = []
    t = n - 1
    
    while t > 0:
        t_prev = int(cp_tau[t])
        if isinstance(y, pd.Series):
            seg_mean = y.iloc[t_prev:t+1].mean()
        else:
            seg_mean = np.mean(y[t_prev:t+1])
        segments.append((t_prev, t, seg_mean))
        t = t_prev
    
    segments.reverse()
    return segments





def plot_sensitivity_tobeta(df,
                            name, 
                            scaling_list = [1,5,10,50,100,500,1000,5000,10000,50000]): 
    y = df[name].dropna()
    y = y[y.index>'2000']

    fig, axes = plt.subplots(1,3,figsize=(15,5))
    axes = axes.flatten()

    for i,loss in tqdm(enumerate(['huber','biweight','l2'])):
        
        beta = compute_penalty_beta(y,loss)
        K=compute_loss_bound_K(y,loss)
        
        list_scaling = np.array(scaling_list)*beta
        nb_changepoints = []

        for scaling in list_scaling:
            if loss == 'huber':
                cp_tau, Qt_vals, Qstar_final = rfpop_algorithm1_main(y, lambda y_t, t: gamma_builder_huber(y_t, K, t), scaling)
            elif loss == 'biweight':
                cp_tau, Qt_vals, Qstar_final = rfpop_algorithm1_main(y, lambda y_t, t: gamma_builder_biweight(y_t, K, t), scaling) 
            elif loss == 'l2':
                cp_tau, Qt_vals, Qstar_final = rfpop_algorithm1_main(y, lambda y_t, t: gamma_builder_L2(y_t, t), scaling)

            nb_changepoints.append(len(set(cp_tau)))

        ax = axes[i]
        ax.plot(scaling_list,nb_changepoints)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("beta scaling factor (logscale)")
        ax.set_ylabel("number of detected changepoints (logscale)")
        ax.set_title(f'{name} - {loss} loss: number of changepoints detected')

    plt.tight_layout()
    plt.show()


def plot_segments(df,name,scaling_huber,scaling_biweight,scaling_l2):
    
    y = df[name].dropna()
    y = y[y.index>'2000']

    fig, axes = plt.subplots(1,3,figsize=(15,5))
    axes = axes.flatten()

    for i,loss in tqdm(enumerate(['huber','biweight','l2'])):
        
        beta = compute_penalty_beta(y,loss)
        K=compute_loss_bound_K(y,loss)
        
        if loss == 'huber':
            cp_tau, Qt_vals, Qstar_final = rfpop_algorithm1_main(y, lambda y_t, t: gamma_builder_huber(y_t, K, t), beta*scaling_huber)
        elif loss == 'biweight':
            cp_tau, Qt_vals, Qstar_final = rfpop_algorithm1_main(y, lambda y_t, t: gamma_builder_biweight(y_t, K, t), beta*scaling_biweight) 
        elif loss == 'l2':
            cp_tau, Qt_vals, Qstar_final = rfpop_algorithm1_main(y, lambda y_t, t: gamma_builder_L2(y_t, t), beta*scaling_l2)

        ax = axes[i]
        ax.plot(y, '.', markersize=2)
        changepoints = np.unique(cp_tau)
        t = len(y) - 1
        segments = []
        
        while t > 0:
            # Le début du segment actuel est indiqué par cp_tau[t]
            t_prev = int(cp_tau[t])
            # On enregistre le segment (start_index, end_index)
            segments.append((t_prev, t))
            # On recule
            t = t_prev
            
        segments.reverse() # On remet dans l'ordre chronologique

        # 3. Calcul de la moyenne de chaque segment
        for start_idx, end_idx in segments:
        
            segment_data = y.iloc[start_idx : end_idx + 1]
            seg_mean = segment_data.mean()
            
            # Récupération des dates pour l'axe X
            date_start = y.index[start_idx]
            date_end = y.index[end_idx] # La date du dernier point du segment

        
            ax.plot([date_start, date_end], [seg_mean, seg_mean], 
                    color='black', linewidth=2, label='Mean on each segment' if start_idx == 0 else "")
            
        
            if end_idx < len(y) - 1:
                ax.axvline(x=date_end, color='r', linestyle='--', alpha=0.5)

        if loss == 'l2':
            ax.set_title(f'{name} - {loss} loss\nbeta = {round(beta*scaling_l2, 1)}')
        elif loss == 'huber':
            ax.set_title(f'{name} - {loss} loss\nK = {round(K, 1)} | beta = {round(beta*scaling_huber, 1)}')
        elif loss == 'biweight':
            ax.set_title(f'{name} - {loss} loss\nK = {round(K, 1)} | beta = {round(beta*scaling_biweight, 1)}')
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.show()



def online_most_recent_changepoint(
    y,
    loss: str = 'biweight',
    beta: float = None,
    K: float = None,
    step: int = 50,  # Calculer tous les 'step' points
    min_obs: int = 100  # Minimum d'observations avant de commencer
):
    """
    Analyse online: pour chaque t, estime le changepoint le plus récent
    étant donné les données y[0:t].
    
    Returns:
    --------
    dict avec 't_values' et 'most_recent_cp'
    """
    n = len(y)
    y_list = list(y) if not isinstance(y, list) else y
    
    # Paramètres par défaut
    if beta is None:
        beta = compute_penalty_beta(np.array(y_list), loss)
    if K is None and loss in ['huber', 'biweight']:
        K = compute_loss_bound_K(np.array(y_list), loss)
    
    # Gamma builder
    if loss == 'huber':
        gamma_builder = lambda y_t, t, K=K: gamma_builder_huber(y_t, K, t)
    elif loss == 'biweight':
        gamma_builder = lambda y_t, t, K=K: gamma_builder_biweight(y_t, K, t)
    else:
        gamma_builder = lambda y_t, t: gamma_builder_L2(y_t, t)
    
    t_values = []
    most_recent_cp = []
    all_changepoints = []
    
    # Pour chaque t de min_obs à n
    for t in tqdm(range(min_obs, n + 1, step), desc=f"Online analysis ({loss})"):
        y_partial = y_list[:t]
        
        cp_tau, Qt_vals, _ = rfpop_algorithm1_main(y_partial, gamma_builder, beta)
        
        # Extraire les changepoints
        changepoints = []
        idx = t - 1
        while idx > 0:
            tau = cp_tau[idx]
            if tau > 0:
                changepoints.append(tau)
            idx = tau
        changepoints = sorted(changepoints)
        
        # Le plus récent = le dernier de la liste (ou 0 si aucun)
        recent_cp = changepoints[-1] if changepoints else 0
        
        t_values.append(t)
        most_recent_cp.append(recent_cp)
        all_changepoints.append(changepoints.copy())
    
    return {
        't_values': t_values,
        'most_recent_cp': most_recent_cp,
        'all_changepoints': all_changepoints,
        'params': {'beta': beta, 'K': K, 'loss': loss}
    }


def plot_most_recent_changepoint(online_result, y=None, dates=None, 
                                  true_changepoints=None, title=""):
    """
    Plot comme dans le papier: Most Recent Changepoint vs Time
    """
    t_vals = online_result['t_values']
    recent_cp = online_result['most_recent_cp']
    recent_cp = np.maximum.accumulate(recent_cp)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot step function du most recent changepoint
    ax.step(t_vals, recent_cp, where='post', linewidth=2, color='black', 
            label='Estimated most recent CP')
    
    # Si on a les vrais changepoints, les afficher en lignes horizontales rouges
    if true_changepoints is not None:
        for cp in true_changepoints:
            ax.axhline(y=cp, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.plot([], [], color='red', linestyle='--', label='True changepoints')
    
    ax.set_xlabel('Time (t)', fontsize=12)
    ax.set_ylabel('Estimated Most Recent Changepoint', fontsize=12)
    ax.set_title(f'{title}\nOnline Analysis - {online_result["params"]["loss"].upper()} loss', 
                 fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
