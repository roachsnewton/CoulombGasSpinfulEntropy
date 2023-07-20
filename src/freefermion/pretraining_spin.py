import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import time

from ..sampler_spin import make_autoregressive_sampler_spin

def make_loss(log_prob, Es, beta):
    
    def loss_fn(params, state_indices):
        logp = log_prob(params, state_indices)
        E = Es[state_indices].sum(axis=-1)
        F = jax.lax.stop_gradient(logp / beta + E)

        ### quantities with energy dimensions are in the units of Ry/rs^2
        E_mean = E.mean()
        F_mean = F.mean()
        S_mean = (-logp).mean()
        E_std = E.std()
        F_std = F.std()
        S_std = (-logp).std()

        gradF = (logp * (F - F_mean)).mean()

        ### <E^2>, <E>^2
        E2_mean = (E**2).mean()
        E_mean2 = (E.mean())**2

        auxiliary_data = {"F_mean": F_mean, "F_std": F_std,
                          "E_mean": E_mean, "E_std": E_std,
                          "S_mean": S_mean, "S_std": S_std,
                          "E2_mean": E2_mean, "E_mean2": E_mean2,
                         }

        return gradF, auxiliary_data

    return loss_fn

def pretrain_spin(van, params_van,
             nup, ndown, dim, Theta, Emax, twist, L, beta,
             path, key,
             lr, sr, damping, max_norm,
             batch, epoch=5000):
    
    n = nup + ndown

    from ..orbitals import sp_orbitals, twist_sort
    sp_indices, _ = sp_orbitals(dim, Emax)
    sp_indices_twist, Es_twist = twist_sort(sp_indices, twist)
    del sp_indices
    sp_indices_twist = jnp.array(sp_indices_twist)[::-1]
    Es_twist = (2*jnp.pi/L)**2 * jnp.array(Es_twist)[::-1]

    from mpmath import mpf, mp
    from .analytic import Z_E
    F, E, S = Z_E(nup, dim, mpf(str(Theta)), [mpf(twist_i) for twist_i in np.array(twist)], Emax)
    ### Twice of the spin polarized case
    print("Analytic results in the unit Ry/rs^2: "
            "F: %s, E: %s, S: %s" % (mp.nstr(F*2), mp.nstr(E*2), mp.nstr(S*2)))
    print("Quantities per particle: "
            "F: %s, E: %s, S: %s" % (mp.nstr(F/nup), mp.nstr(E/nup), mp.nstr(S/nup)))
    
    num_states = Es_twist.size
    sampler, log_prob_novmap = make_autoregressive_sampler_spin(van, 
                                sp_indices_twist, nup, ndown, num_states)
    log_prob = jax.vmap(log_prob_novmap, (None, 0), 0)

    loss_fn = make_loss(log_prob, Es_twist, beta)

    import optax
    if sr:
        from ..sampler import make_classical_score
        score_fn = make_classical_score(log_prob_novmap)
        from ..sr import fisher_sr
        optimizer = fisher_sr(score_fn, damping, max_norm)
        print("Optimizer fisher_sr: damping = %.5f, max_norm = %.5f." % (damping, max_norm))
    else:
        optimizer = optax.adam(lr)
        print("Optimizer adam: lr = %.3f." % lr)
    opt_state = optimizer.init(params_van)

    @jax.jit
    def update(params_van, opt_state, key):
        key, subkey = jax.random.split(key)
        state_indices = sampler(params_van, subkey, batch)

        grads, aux = jax.grad(loss_fn, argnums=0, has_aux=True)(params_van, state_indices)
        updates, opt_state = optimizer.update(grads, opt_state,
                                params=(params_van, state_indices) if sr else None)
        params_van = optax.apply_updates(params_van, updates)
        
        return params_van, opt_state, key, aux

    import os
    log_filename = os.path.join(path, "data.txt")
    f = open(log_filename, "w", buffering=1, newline="\n")

    for i in range(1, epoch+1):
        t0 = time.time()
        params_van, opt_state, key, aux = update(params_van, opt_state, key)
        E, E_std, F, F_std, S, S_std = aux["E_mean"], aux["E_std"], \
                                       aux["F_mean"], aux["F_std"], \
                                       aux["S_mean"], aux["S_std"]
        
        ### capacity heat Cv, Cv1 = beta^2 * <E^2>, Cv2 = beta^2 * <E>^2
        E_mean2 = aux["E_mean2"]
        E2_mean = aux["E2_mean"]
        Cv1 = (beta**2) * E2_mean
        Cv2 = (beta**2) * E_mean2
        Cv = Cv1 - Cv2
        Cv, Cv1, Cv2 = Cv/n, Cv1/n, Cv2/n
        
        # Quantities per particle
        F, F_std, E, E_std, S, S_std = F/n, F_std/n, E/n, E_std/n, S/n, S_std/n
        F_std = F_std / jnp.sqrt(batch)
        E_std = E_std / jnp.sqrt(batch)
        S_std = S_std / jnp.sqrt(batch)
        
        t1 = time.time()
        dt = t1 - t0
        print("iter: %06d" % i,
                " F: %.6f" % F, "(%.6f)" % F_std,
                " E: %.6f" % E, "(%.6f)" % E_std,
                " S: %.6f" % S, "(%.6f)" % S_std,
                " Cv: %.6f"  % Cv, "=(%.6f ,%.6f)"  % (Cv1, Cv2),
                " dt: %.3f" % dt)

        f.write( ("%6d" + "  %.6f"*9 + "  %.3f\n") % (i, 
                    F, F_std, E, E_std, S, S_std, 
                    Cv, Cv1, Cv2, dt) )

    return params_van