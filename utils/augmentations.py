import torch
import numpy as np

def remove_frequency(x_f, frac = 0.1):
    remove_freqs = torch.rand_like(x_f) > frac
    return x_f*remove_freqs

def remove_frequency_abs_budget(x_f, E = 20):
    remove_freqs = torch.rand(x_f.shape).argsort(2)[:,:,:E]
    return x_f.scatter(-1, remove_freqs, 0)

def add_frequency(x_f, alpha = 0.5, frac = 0.1):
    max_freq = torch.max(x_f, axis = 2)[0].unsqueeze(2)
    freq_mask = x_f < max_freq*alpha
    rand_mask = torch.rand_like(x_f) < frac 
    final_mask = freq_mask*rand_mask
    new_freqs = final_mask*max_freq*alpha

    return x_f*~final_mask + new_freqs

def add_frequency_abs_budget(x_f, alpha = 0.5, E = 20):
    
    for i, sample in enumerate(x_f):
        for j, row in enumerate(sample):
            max_freq = torch.max(row)*alpha
            low_freqs = torch.nonzero(row < max_freq).squeeze()
            add_freqs = torch.rand(low_freqs.shape).argsort()[:E]
            x_f[i,j,low_freqs[add_freqs]] = max_freq

    return x_f

def frequency_augmentation(freq_cont, keep_all = True, return_ifft = True, abs_budget = True):
    x_f = freq_cont.abs()

    if abs_budget:
        x_f_add = add_frequency_abs_budget(x_f)
        x_f_rem = remove_frequency_abs_budget(x_f)
    else:
        x_f_add = add_frequency(x_f)
        x_f_rem = remove_frequency(x_f)

    if keep_all:
        collect_x_f = torch.cat((x_f.unsqueeze(1), x_f_add.unsqueeze(1), x_f_rem.unsqueeze(1)), axis = 1)
        if return_ifft:
            x_f_phase = freq_cont.imag
            collect_x_t = torch.fft.ifft(collect_x_f + x_f_phase.unsqueeze(1), axis = -1).real
            return collect_x_f, collect_x_t
    else:
        collect_x_f = torch.cat((x_f_add.unsqueeze(1), x_f_rem.unsqueeze(1)), axis = 1)
    
    return collect_x_f

# Temporal augmentations

def jitter(x, sigma = 0.01):
    return x + np.random.normal(0, sigma, size = x.shape)

def scaling(x, sigma = 1.1):
    factor = np.random.normal(2, sigma, size = [x.shape[0], 1, 1])
    return x * factor

def permutation(x, max_seg = 8):
    seg = np.random.randint(1, max_seg, size = x.shape[0])

    for i, signal in enumerate(x):
        sig_points = np.random.choice(np.arange(signal.shape[-1]), size = seg[i], replace = False)
        sig_points = [0, *np.sort(sig_points), signal.shape[-1]-1]
        #sig_points = [point for i, point in enumerate(sig_points[:-1]) if sig_points[i+1]-point > min_length][1:]
        sig_split = np.split(signal.squeeze().numpy(), sig_points, axis = 0)
        new_sig = np.concatenate(np.random.permutation(sig_split), axis = 0)
        x[i,0,:] = torch.Tensor(new_sig)

    return x

def time_augmentation(x, keep_all = True, return_fft = True):
    x_jitter = jitter(x.clone())
    x_scale = scaling(x.clone())
    x_perm = permutation(x.clone())

    if keep_all:
        collect_x_t = torch.cat((x.unsqueeze(1), x_jitter.unsqueeze(1), x_scale.unsqueeze(1), x_perm.unsqueeze(1)), axis = 1)
        if return_fft:
            collect_x_f = torch.fft.fft(collect_x_t, axis = -1).abs()
            return collect_x_t, collect_x_f
    else:
        collect_x_t = torch.cat((x_jitter.unsqueeze(1), x_scale.unsqueeze(1), x_perm.unsqueeze(1)), axis = 1)
    
    return collect_x_t