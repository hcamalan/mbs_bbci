def warp_nonlinear(s, t, intensity = 20, smoothing_factor = 10):
    '''
    Warps a signal in the time domain in a nonlinear manner. Uses the "add_noise" function, then smoothens this noise to create nonlinear time function. 
    Takes:
        s (1-dim array) - source signal to warp
        t (1-dim array) - time function of the same signal, default value is a range from zero to the size of "s"
        intensity (int, float) - proportion of the intensity that will be added as noise to the time function
        smoothing_factor (int, float) - SD of the Gaussian kernel that will be used for smoothing
    Returns:
        warped_t (1-dim array) - time function of the warped signal
        warped_s (1-dim array) - warped signal
    '''

    t = np.arange(len(s))
    warped_t = add_noise(t, intensity)
    warped_t = gaussian_filter(warped_t, smoothing_factor)
    warped_s = np.interp(warped_t, t, s)
    return warped_t, warped_s

def warp_linear(s, scale, phase):
    '''
    Warps a signal in the time domain in a linear manner. Creates a time function and interpolates the values of the source signal to match this time function.
    Takes:
        s (1-dim array) - source signal to warp
        scale - scale (or slope) of the time function
        phase - phase of the time function
    Returns:
        warped_t (1-dim array) - time function of the warped signal
        warped_s (1-dim array) - warped signal
    '''
    t = np.arange(len(s))
    warped_t = np.linspace(0, len(t)/scale, len(t)) - phase
    warped_s = np.interp(warped_t, t, s)
    return warped_t, warped_s


def create_source_signal(T = 1000):
    '''
    Generates a source signal by combining some Gaussians.
    '''
    
    
    #T = 1000
    stdevs = [50, 10, 20, 30]
    signs = [-1, -1, 1, 1]
    t = np.arange(T)
    s = np.zeros(T)

    for ind, it in enumerate(stdevs):
        g = np.roll(gaussian(T, it), np.int((np.random.rand()-0.5)*120))
        s = s + g*signs[ind]
    
    
    source_signal = s / np.max(np.abs(s))
    
    # add noise
    pure_signal = source_signal.copy()
    source_signal = source_signal + (np.random.rand(len(source_signal)) - 0.5)
    
    # whiten
    source_signal = source_signal - np.average(source_signal)#normalize
    return t, source_signal, pure_signal


#Data generation



def create_curves(source_signal, scales, phases):
    '''
    Generates curves with warped time given phase and scale differences. 
    '''
    n_signals = len(scales)
    t = np.arange(len(source_signal))
    
    warped_signals = np.zeros((len(t), n_signals))
    times = warped_signals.copy()
    for i in range(n_signals):
        times[:, i] = np.linspace(0, len(t)/scales[i], len(t)) - phases[i]
        warped_signals[:, i] = np.interp(times[:, i], t, source_signal)
    return times, warped_signals


# Feature functions

def I_min(source_signal, r):
    '''
    Feature function "min" 
    '''
    I = np.power(np.max(source_signal) - source_signal, r)
    return I/np.sum(I)

def I_max(source_signal, r):
    '''
    Feature function "max"
    '''
    I = np.power(source_signal - np.min(source_signal), r)
    return I/np.sum(I)

def I_local(source_signal, r):
    '''
    Feature function "local"
    '''
    u = np.abs(np.gradient(source_signal))
    l = np.sqrt(np.abs(np.gradient(np.gradient(source_signal))))
    I = np.exp(-1*r*(u/l))
    return I/np.sum(I)

# Moment calculation

def get_moment(I, t, m=1):
    '''
    Computes first or later moments. I: feature function, t: time function, m:order of moment 
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
    smoothened_curves = np.zeros(curves.shape)
    for i in range(curves.shape[1]):
        smoothened_curves[:, i] = gaussian_filter(curves[:, i], std)
    return smoothened_curves

def synchronize(curves, fun, r = 20, s = True, s_std = 2):
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



