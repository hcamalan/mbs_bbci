import numpy as np
from scipy.ndimage.filters import gaussian_filter

def add_noise(source_signal, intensity=0.1, uniform = False):
    '''
    Adds gaussian or uniform noise to a signal.
    Takes:
        source_signal (1-dim array) - source signal to add noise to
        intensity (float) - proportion of the intensity of the function that will be added as noise
        uniform (boolean) - parameter that picks distribution of the noise. With False a Gaussian distribution is picked, and with True a uniform one
    Returns:
        (1-dim array) signal with noise added
    '''
    if uniform == True:
        return source_signal + np.random.uniform(-1*intensity, intensity, len(sig))
    noise = np.random.rand(len(source_signal)) - 0.5
    noise = (noise/np.max(np.abs(noise))) *intensity
    return source_signal + noise
    
def warp_tf_linear(t, scale = 1, phase = 0):
    return np.linspace(0, len(t)/scale, len(t)) - phase    

def warp_tf_nonlinear(t, intensity = 20, smoothing_factor = 10):
    warped_t = add_noise(t, intensity = intensity)
    return gaussian_filter(warped_t, smoothing_factor)

def create_curves(source_signal, scales, phases, noise_args = {}, nonlinear_args = {}):
    '''
    Generates curves with various properties like linear and nonlinear time warping as well as noise. 
    '''
    noise, nonlinear = False, False
    if noise_args.keys() != []:
        noise = True
    if nonlinear_args.keys() != []:
        nonlinear = True
    
    n_signals = len(scales)
    t = np.arange(len(source_signal))
    
    warped_signals = np.zeros((len(t), n_signals))
    times = warped_signals.copy()
    for i in range(n_signals):
        w_t = warp_tf_linear(t.copy(), scales[i], phases[i])
        if nonlinear == True:
            w_t = warp_tf_nonlinear(w_t, nonlinear_args["intensity"], nonlinear_args["sf"])
        w_s = np.interp(w_t, t, source_signal)
        if noise == True:
            w_s = add_noise(w_s, intensity = noise_args["intensity"])
        times[:, i], warped_signals[:, i] = w_t, w_s
    
    return times, warped_signals

def I_max(source_signal, r):
    '''
    Feature function "max"
    Takes:
        source_signal - source signal whose feature function will be taken
        r - parameter that determines the sharpness of the feature function. As r approaches infinity, the feature function turns into a Dirac delta function.
    Returns:
        I - feature function of the given source signal(1-dim array)
    '''
    I = np.power(source_signal - np.min(source_signal), r)
    return I/np.sum(I)

def I_local(source_signal, r):
    '''
    Feature function "local"
    Takes:
        source_signal - source signal whose feature function will be taken
        r - parameter that determines the sharpness of the feature function. As r approaches infinity, the feature function turns into a Dirac delta function.
    Returns:
        I - feature function of the given source signal(1-dim array)
    '''
    u = np.abs(np.gradient(source_signal))
    l = np.sqrt(np.abs(np.gradient(np.gradient(source_signal))))
    I = np.exp(-1*r*(u/l))
    return I/np.sum(I)

def I_local_scaled(source_signal, r):
    '''
    Feature function "local 2"
    Takes:
        source_signal - source signal whose feature function will be taken
        r - parameter that determines the sharpness of the feature function. As r approaches infinity, the feature function turns into a Dirac delta function.
    Returns:
        I - feature function of the given source signal(1-dim array)
    '''
    u = np.abs(np.gradient(source_signal))
    l = np.sqrt(np.abs(np.gradient(np.gradient(source_signal))))
    I = np.exp(-1*r*(u/l))
    I = np.multiply(I, np.abs(source_signal))# - np.average(source_signal)))
    I = np.power(I, r)
    return I/np.sum(I)



def I_min(source_signal, r):
    '''
    Feature function "min"
    Takes:
        source_signal - source signal whose feature function will be taken
        r - parameter that determines the sharpness of the feature function. As r approaches infinity, the feature function turns into a Dirac delta function.
    Returns:
        I - feature function of the given source signal(1-dim array)
    '''
    I = np.power(np.max(source_signal) - source_signal, r)
    return I/np.sum(I)

def get_moment(I, t, m=1):
    '''
    Computes first or later moments.
    Takes:
        I - feature function,
        t - time function
        m - order of moment
    Returns:
        (float) a moment value
    '''
    #Eq. 4 from the paper
    first_moment = np.sum(I * t)/len(t)
    if m == 1:
        return first_moment
    
    #Eq.5
    t_new = np.power(t - (first_moment*len(t)), m)
    kth_moment = np.sum(I * t_new)/len(t_new) 
    return kth_moment

def smoothen(curves, std = 2):
    '''
    Smoothens a number of curves using a Gaussian kernel
    Takes:
        curves (2-dim array): first index contains the values of the individual curves, second index refers to which curve
        std: standard deviation of the Gaussian kernel used for smoothing
    Returns:
        smoothened_curves (2-dim array)
    '''
    smoothened_curves = np.zeros(curves.shape)
    for i in range(curves.shape[1]):
        smoothened_curves[:, i] = gaussian_filter(curves[:, i], std)
    return smoothened_curves

def synchronize(curves, fun, r = 20, s = True, s_std = 2):
    '''
    Synchronizes a number of curves using Moment-Based Synchronization
    
    Returns:
        X_t (2-dim array) -  dewarping functions of each curve
        synchronized_curves (2-dim array) - synchronized curves
    '''
    n_curves = curves.shape[1]
    
    if s == True:
        curves = smoothen(curves, s_std)
    
    t = np.arange(curves.shape[0])
    first_moments, second_moments = np.zeros(n_curves), np.zeros(n_curves)
    
    for i in range(n_curves):
        feature_function = fun(curves[:, i], r)
        first_moments[i] = get_moment(feature_function, t, 1)
        second_moments[i] = get_moment(feature_function, t, 2)
    
    first_moments, second_moments = first_moments*curves.shape[0], second_moments*curves.shape[0]    
    
    z_first_moment = np.average(first_moments)
    z_second_moment = np.power(np.sum(np.sqrt(second_moments)) / n_curves, 2)

    betas = np.sqrt(second_moments/z_second_moment)
    alphas = first_moments - betas*z_first_moment

    X_t = np.zeros(curves.shape)
    synchronized_curves = np.zeros(curves.shape)

    #print len(t)
    for i in range(n_curves):
        X_t[:, i] = t * betas[i] + alphas[i]
        synchronized_curves[:, i] = np.interp(X_t[:, i], t, curves[:, i])

    return X_t, synchronized_curves


