import jax
import jax.numpy as jnp

from functools import partial

from .MCMC import mcmc

@partial(jax.pmap, axis_name="p",
                   in_axes=(0, None, 0, None, 0, 0, None, None, None),
                   static_broadcasted_argnums=(1, 3),
                   donate_argnums=4)
def sample_stateindices_and_x(key,
                              sampler, params_van,
                              logp, x, params_flow,
                              mc_steps, mc_stddev, L):
    """
        Generate new state_indices of shape (batch, n), as well as coordinate sample
    of shape (batch, n, dim), from the sample of last optimization step.
    """
    key, key_state, key_MCMC = jax.random.split(key, 3)
    batch = x.shape[0]
    state_indices = sampler(params_van, key_state, batch)
    x, accept_rate = mcmc(lambda x: logp(x, params_flow, state_indices), x, key_MCMC, mc_steps, mc_stddev)
    x -= L * jnp.floor(x/L)
    return key, state_indices, x, accept_rate

####################################################################################

from .potential import potential_energy

def make_loss(log_prob, logpsi, logpsi_grad_laplacian, kappa, G, L, rs, Vconst, beta):

    def observable_and_lossfn(params_van, params_flow, state_indices, x, key):
        logp_states = log_prob(params_van, state_indices)
        grad, laplacian = logpsi_grad_laplacian(x, params_flow, state_indices, key)
        print("grad.shape:", grad.shape)
        print("laplacian.shape:", laplacian.shape)

        kinetic = -laplacian - (grad**2).sum(axis=(-2, -1))
        potential = potential_energy(x, kappa, G, L, rs) + Vconst
        
        # Quantities with energy dimensions are in the units of Ry/rs^2
        Ekloc = kinetic.real
        Eploc = potential.real
        Eloc = Ekloc + Eploc
        Sloc = -logp_states.real
        Floc = -Sloc / beta + Eloc

        # Calculate H, partialH, H * partialH
        H    =  (Ekloc + Eploc) / (rs**2)
        pH   = -(2*Ekloc + Eploc) / (rs**3)
        H_pH = H * pH
        
        H_mean    = jax.lax.pmean(H.mean(), axis_name="p")
        pH_mean   = jax.lax.pmean(pH.mean(), axis_name="p")
        H_pH_mean = jax.lax.pmean(H_pH.mean(), axis_name="p")
        
        H2_mean    = jax.lax.pmean((H**2).mean(), axis_name="p")
        pH2_mean   = jax.lax.pmean((pH**2).mean(), axis_name="p")
        H_pH2_mean = jax.lax.pmean((H_pH**2).mean(), axis_name="p")
        
        # Beta is in the unit of rs^2/Ry
        beta_rs = beta * (rs**2)
        # Calculate partial derivatives partialS
        pS_mean = (beta_rs**2) * (H_mean * pH_mean - H_pH_mean)
        
        # Changed the energy units Ry/rs^2 -> Ry
        Ekloc = Ekloc/(rs**2)
        Eploc = Eploc/(rs**2)
        Eloc = Eloc/(rs**2)
        Floc = Floc/(rs**2)
        
        # Calculate mean and variance of observables
        K_mean, K2_mean, V_mean, V2_mean, \
        E_mean, E2_mean, F_mean, F2_mean, S_mean, S2_mean = \
        jax.tree_map(lambda x: jax.lax.pmean(x, axis_name="p"), 
                     (Ekloc.mean(), (Ekloc**2).mean(),   # Ek: kinetic
                      Eploc.mean(), (Eploc**2).mean(),   # Ep: potential
                      Eloc.mean(),  (Eloc**2).mean(),    # E:  energy
                      Floc.mean(),  (Floc**2).mean(),    # F:  free energy
                      Sloc.mean(),  (Sloc**2).mean()     # S:  entropy
                     ))
        observable = {"K_mean": K_mean, "K2_mean": K2_mean,
                      "V_mean": V_mean, "V2_mean": V2_mean,
                      "E_mean": E_mean, "E2_mean": E2_mean,
                      "F_mean": F_mean, "F2_mean": F2_mean,
                      "S_mean": S_mean, "S2_mean": S2_mean,
                      "pS_mean":   pS_mean,
                      "H_mean":    H_mean,    "H2_mean":    H2_mean,
                      "pH_mean":   pH_mean,   "pH2_mean":   pH2_mean,
                      "H_pH_mean": H_pH_mean, "H_pH2_mean": H_pH2_mean,
                      }

        def classical_lossfn(params_van):
            logp_states = log_prob(params_van, state_indices)

            tv = jax.lax.pmean(jnp.abs(Floc - F_mean).mean(), axis_name="p")
            Floc_clipped = jnp.clip(Floc, F_mean - 5.0*tv, F_mean + 5.0*tv)
            gradF_phi = (logp_states * Floc_clipped).mean()
            classical_score = logp_states.mean()
            return gradF_phi, classical_score

        def quantum_lossfn(params_flow):
            logpsix = logpsi(x, params_flow, state_indices)

            tv = jax.lax.pmean(jnp.abs(Eloc - E_mean).mean(), axis_name="p")
            Eloc_clipped = jnp.clip(Eloc, E_mean - 5.0*tv, E_mean + 5.0*tv)
            gradF_theta = 2 * (logpsix * Eloc_clipped.conj()).real.mean()
            quantum_score = 2 * logpsix.real.mean()
            return gradF_theta, quantum_score

        return observable, classical_lossfn, quantum_lossfn

    return observable_and_lossfn
