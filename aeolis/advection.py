'''This file is part of AeoLiS.
   
AeoLiS is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
   
AeoLiS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
   
You should have received a copy of the GNU General Public License
along with AeoLiS.  If not, see <http://www.gnu.org/licenses/>.
   
AeoLiS  Copyright (C) 2015 Bas Hoonhout

bas.hoonhout@deltares.nl         b.m.hoonhout@tudelft.nl
Deltares                         Delft University of Technology
Unit of Hydraulic Engineering    Faculty of Civil Engineering and Geosciences
Boussinesqweg 1                  Stevinweg 1
2629 HVDelft                     2628CN Delft
The Netherlands                  The Netherlands

'''

from __future__ import absolute_import, division

import logging
import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse.linalg
from numba import njit

# import AeoLiS modules
import aeolis.transport
from aeolis.utils import prevent_tiny_negatives, format_log, rotate

# initialize logger
logger = logging.getLogger(__name__)

def solve_steadystate(self) -> dict:
    '''Implements the steady state solution
    '''
    # upwind scheme:
    beta = 1. 
    
    l = self.l
    s = self.s
    p = self.p

    Ct = s['Ct'].copy()
    pickup = s['pickup'].copy()

    # compute transport weights for all sediment fractions
    w_init, w_air, w_bed = aeolis.transport.compute_weights(s, p)

    if self.t == 0.:
        # use initial guess for first time step
        if p['grain_dist'] != None:
            w = p['grain_dist'].reshape((1,1,-1))
            w = w.repeat(p['ny']+1, axis=0)
            w = w.repeat(p['nx']+1, axis=1)
        else:
            w = w_init.copy()
    else:
        w = w_init.copy()

    # set model state properties that are added to warnings and errors
    logprops = dict(minwind=s['uw'].min(),
                    maxdrop=(l['uw']-s['uw']).max(),
                    time=self.t,
                    dt=self.dt)
        
    nf = p['nfractions']     
            
    us = np.zeros((p['ny']+1,p['nx']+1))
    un = np.zeros((p['ny']+1,p['nx']+1))
    
    us_plus = np.zeros((p['ny']+1,p['nx']+1))
    un_plus = np.zeros((p['ny']+1,p['nx']+1))
    
    us_min = np.zeros((p['ny']+1,p['nx']+1))
    un_min = np.zeros((p['ny']+1,p['nx']+1))

    Cs = np.zeros(us.shape)
    Cn = np.zeros(un.shape)
    
    Cs_plus = np.zeros(us.shape)
    Cn_plus = np.zeros(un.shape)
    
    Cs_min = np.zeros(us.shape)
    Cn_min = np.zeros(un.shape)
    
    for i in range(nf):
        us[:,:] = s['us'][:,:,i] 
        un[:,:] = s['un'][:,:,i] 
        
        us_plus[:,1:] = s['us'][:,:-1,i] 
        un_plus[1:,:] = s['un'][:-1,:,i] 
        
        us_min[:,:-1] = s['us'][:,1:,i]
        un_min[:-1,:] = s['un'][1:,:,i]
    
        #boundary values
        us[:,0]  = s['us'][:,0,i]
        un[0,:]  = s['un'][0,:,i]
        
        us_plus[:,0]  = s['us'][:,0,i]
        un_plus[0,:]  = s['un'][0,:,i]
        
        us_min[:,-1]  = s['us'][:,-1,i]
        un_min[-1,:]  = s['un'][-1,:,i]
        
        
        # define matrix coefficients to solve linear system of equations        
        Cs = s['dn'] * s['dsdni'] * us[:,:]  
        Cn = s['ds'] * s['dsdni'] * un[:,:] 

        Cs_plus = s['dn'] * s['dsdni'] * us_plus[:,:]  
        Cn_plus = s['ds'] * s['dsdni'] * un_plus[:,:]
        
        Cs_min = s['dn'] * s['dsdni'] * us_min[:,:]  
        Cn_min = s['ds'] * s['dsdni'] * un_min[:,:]
        
        
        Ti = 1 / p['T']
        
        beta = abs(beta)
        if beta >= 1.:
            # define upwind direction
            ixs = np.asarray(us[:,:] >= 0., dtype=float)
            ixn = np.asarray(un[:,:] >= 0., dtype=float)
            sgs = 2. * ixs - 1.
            sgn = 2. * ixn - 1.
        
        else:
            # or centralizing weights
            ixs = beta + np.zeros(us)
            ixn = beta + np.zeros(un)
            sgs = np.zeros(us)
            sgn = np.zeros(un)

        # initialize matrix diagonals
        A0 = np.zeros(s['zb'].shape)
        Apx = np.zeros(s['zb'].shape)
        Ap1 = np.zeros(s['zb'].shape)
        Ap2 = np.zeros(s['zb'].shape)
        Amx = np.zeros(s['zb'].shape)
        Am1 = np.zeros(s['zb'].shape)
        Am2 = np.zeros(s['zb'].shape)

        # populate matrix diagonals
        A0  = sgs * Cs + sgn * Cn + Ti
        Apx = Cn_min * (1. - ixn)
        Ap1 = Cs_min * (1. - ixs)
        Amx = -Cn_plus * ixn
        Am1 = -Cs_plus * ixs    

        # add boundaries
        A0[:,0] = 1.
        Apx[:,0] = 0.
        Amx[:,0] = 0.
        Am2[:,0] = 0.
        Am1[:,0] = 0.

        A0[:,-1] = 1.
        Apx[:,-1] = 0.
        Ap1[:,-1] = 0.
        Ap2[:,-1] = 0.
        Amx[:,-1] = 0.

        if p['boundary_offshore'] == 'flux':
            Ap2[:,0] = 0.
            Ap1[:,0] = 0.
        elif p['boundary_offshore'] == 'constant':
            Ap2[:,0] = 0.
            Ap1[:,0] = 0.
        elif p['boundary_offshore'] == 'uniform':
            Ap2[:,0] = 0.
            Ap1[:,0] = -1.
        elif p['boundary_offshore'] == 'gradient':
            Ap2[:,0] = s['ds'][:,1] / s['ds'][:,2]
            Ap1[:,0] = -1. - s['ds'][:,1] / s['ds'][:,2]
        elif p['boundary_offshore'] == 'circular':
            logger.log_and_raise('Cross-shore cricular boundary condition not yet implemented', exc=NotImplementedError)
        else:
            logger.log_and_raise('Unknown offshore boundary condition [%s]' % self.p['boundary_offshore'], exc=ValueError)

        if p['boundary_onshore'] == 'flux':                              
            Am2[:,-1] = 0.
            Am1[:,-1] = 0.            
        elif p['boundary_onshore'] == 'constant':                              
            Am2[:,-1] = 0.
            Am1[:,-1] = 0.
        elif p['boundary_onshore'] == 'uniform':
            Am2[:,-1] = 0.
            Am1[:,-1] = -1.
        elif p['boundary_onshore'] == 'gradient':
            Am2[:,-1] = s['ds'][:,-2] / s['ds'][:,-3]
            Am1[:,-1] = -1. - s['ds'][:,-2] / s['ds'][:,-3]
        elif p['boundary_offshore'] == 'circular':
            logger.log_and_raise('Cross-shore cricular boundary condition not yet implemented', exc=NotImplementedError)
        else:
            logger.log_and_raise('Unknown onshore boundary condition [%s]' % self.p['boundary_onshore'], exc=ValueError)

        if p['boundary_lateral'] == 'constant':
            A0[0,:] = 1.
            Apx[0,:] = 0.
            Ap1[0,:] = 0.
            Amx[0,:] = 0.
            Am1[0,:] = 0.
            
            A0[-1,:] = 1.
            Apx[-1,:] = 0.
            Ap1[-1,:] = 0.
            Amx[-1,:] = 0.
            Am1[-1,:] = 0.
        
            #logger.log_and_raise('Lateral constant boundary condition not yet implemented', exc=NotImplementedError)
        elif p['boundary_lateral'] == 'uniform':
            logger.log_and_raise('Lateral uniform boundary condition not yet implemented', exc=NotImplementedError)
        elif p['boundary_lateral'] == 'gradient':
            logger.log_and_raise('Lateral gradient boundary condition not yet implemented', exc=NotImplementedError)
        elif p['boundary_lateral'] == 'circular':
            pass
        else:
            logger.log_and_raise('Unknown lateral boundary condition [%s]' % self.p['boundary_lateral'], exc=ValueError)

        # construct sparse matrix
        if p['ny'] > 0:
            j = p['nx']+1
            A = scipy.sparse.diags((Apx.flatten()[:j],
                                    Amx.flatten()[j:],
                                    Am2.flatten()[2:],
                                    Am1.flatten()[1:],
                                    A0.flatten(),
                                    Ap1.flatten()[:-1],
                                    Ap2.flatten()[:-2],
                                    Apx.flatten()[j:],
                                    Amx.flatten()[:j]),
                                    (-j*p['ny'],-j,-2,-1,0,1,2,j,j*p['ny']), format='csr')
        else:
            A = scipy.sparse.diags((Am2.flatten()[2:],
                                    Am1.flatten()[1:],
                                    A0.flatten(),
                                    Ap1.flatten()[:-1],
                                    Ap2.flatten()[:-2]),
                                    (-2,-1,0,1,2), format='csr')

        # solve transport for each fraction separately using latest
        # available weights

        # renormalize weights for all fractions equal or larger
        # than the current one such that the sum of all weights is
        # unity
        w = aeolis.transport.renormalize_weights(w, i)

        # iteratively find a solution of the linear system that
        # does not violate the availability of sediment in the bed
        for n in range(p['max_iter']):
            self._count('matrixsolve')

            # compute saturation levels
            ix = s['Cu'] > 0.
            S_i = np.zeros(s['Cu'].shape)
            S_i[ix] = s['Ct'][ix] / s['Cu'][ix]
            s['S'] = S_i.sum(axis=-1)

            # create the right hand side of the linear system
            y_i = np.zeros(s['zb'].shape)
            
            y_i[:,1:-1] = (
                (w[:,1:-1,i] * s['Cuf'][:,1:-1,i] * Ti) * (1. - s['S'][:,1:-1]) +
                (w[:,1:-1,i] * s['Cu'][:,1:-1,i] * Ti) * s['S'][:,1:-1]
                )

            # add boundaries
            if p['boundary_offshore'] == 'flux':
                y_i[:,0] = p['offshore_flux'] * s['Cu0'][:,0,i] 
            if p['boundary_onshore'] == 'flux':
                y_i[:,-1] = p['onshore_flux'] * s['Cu0'][:,-1,i] 
                
            if p['boundary_offshore'] == 'constant':
                y_i[:,0] = p['constant_offshore_flux'] / s['u'][:,0,i] 
            if p['boundary_onshore'] == 'constant':
                y_i[:,-1] = p['constant_onshore_flux'] / s['u'][:,-1,i]

            # solve system with current weights
            Ct_i = scipy.sparse.linalg.spsolve(A, y_i.flatten())
            Ct_i = prevent_tiny_negatives(Ct_i, p['max_error'])
                
            # check for negative values
            if Ct_i.min() < 0.:
                ix = Ct_i < 0.

                logger.warning(format_log('Removing negative concentrations',
                                            nrcells=np.sum(ix),
                                            fraction=i,
                                            iteration=n,
                                            minvalue=Ct_i.min(),
                                            coords=np.argwhere(ix.reshape(y_i.shape)),
                                            **logprops))

                Ct_i[~ix] *= 1. + Ct_i[ix].sum() / Ct_i[~ix].sum()
                Ct_i[ix] = 0.

            # determine pickup and deficit for current fraction
            Cu_i = s['Cu'][:,:,i].flatten()
            mass_i = s['mass'][:,:,0,i].flatten()
            w_i = w[:,:,i].flatten()
            pickup_i = (w_i * Cu_i - Ct_i) / p['T'] * self.dt
            deficit_i = pickup_i - mass_i
            ix = (deficit_i > p['max_error']) \
                    & (w_i * Cu_i > 0.)

            # quit the iteration if there is no deficit, otherwise
            # back-compute the maximum weight allowed to get zero
            # deficit for the current fraction and progress to
            # the next iteration step
            if not np.any(ix):
                logger.debug(format_log('Iteration converged',
                                        steps=n,
                                        fraction=i,
                                        **logprops))
                pickup_i = np.minimum(pickup_i, mass_i)
                break
            else:
                w_i[ix] = (mass_i[ix] * p['T'] / self.dt \
                            + Ct_i[ix]) / Cu_i[ix]
                w[:,:,i] = w_i.reshape(y_i.shape)

        # throw warning if the maximum number of iterations was reached
        if np.any(ix):
            logger.warning(format_log('Iteration not converged',
                                        nrcells=np.sum(ix),
                                        fraction=i,
                                        **logprops))

        # check for unexpected negative values
        if Ct_i.min() < 0:
            logger.warning(format_log('Negative concentrations',
                                        nrcells=np.sum(Ct_i<0.),
                                        fraction=i,
                                        minvalue=Ct_i.min(),
                                        **logprops))
        if w_i.min() < 0:
            logger.warning(format_log('Negative weights',
                                        nrcells=np.sum(w_i<0),
                                        fraction=i,
                                        minvalue=w_i.min(),
                                        **logprops))

        Ct[:,:,i] = Ct_i.reshape(y_i.shape)
        pickup[:,:,i] = pickup_i.reshape(y_i.shape)

    # check if there are any cells where the sum of all weights is
    # smaller than unity. these cells are supply-limited for all
    # fractions. Log these events.
    ix = 1. - np.sum(w, axis=2) > p['max_error']
    if np.any(ix):
        self._count('supplylim')
        logger.warning(format_log('Ran out of sediment',
                                    nrcells=np.sum(ix),
                                    minweight=np.sum(w, axis=-1).min(),
                                    **logprops))
        
    
    qs = Ct * s['us'] 
    qn = Ct * s['un'] 
    q = np.hypot(qs, qn)


    return dict(Ct=Ct,
                qs=qs,
                qn=qn,
                pickup=pickup,
                w=w,
                w_init=w_init,
                w_air=w_air,
                w_bed=w_bed,
                q=q)
    
    
def solve(self, alpha:float=.5, beta:float=1.) -> dict:
    '''Implements the explicit Euler forward, implicit Euler backward and semi-implicit Crank-Nicolson numerical schemes

    Determines weights of sediment fractions, sediment pickup and
    instantaneous sediment concentration. Returns a partial
    spatial grid dictionary that can be used to update the global
    spatial grid dictionary.

    Parameters
    ----------
    alpha :
        Implicitness coefficient (0.0 for Euler forward, 1.0 for Euler backward or 0.5 for Crank-Nicolson, default=0.5)
    beta : 
        Centralization coefficient (1.0 for upwind or 0.5 for centralized, default=1.0)

    Returns
    -------
        Partial spatial grid dictionary

    Examples
    --------
    >>> model.s.update(model.solve(alpha=1., beta=1.) # euler backward

    >>> model.s.update(model.solve(alpha=.5, beta=1.) # crank-nicolson

    See Also
    --------
    model.AeoLiS.euler_forward
    model.AeoLiS.euler_backward
    model.AeoLiS.crank_nicolson
    transport.compute_weights
    transport.renormalize_weights

    '''

    l = self.l
    s = self.s
    p = self.p

    Ct = s['Ct'].copy()
    pickup = s['pickup'].copy()

    # compute transport weights for all sediment fractions
    w_init, w_air, w_bed = aeolis.transport.compute_weights(s, p)

    if self.t == 0.:
        if type(p['bedcomp_file']) == np.ndarray:
            w = w_init.copy()
        else:
            # use initial guess for first time step
            w = p['grain_dist'].reshape((1,1,-1))
            w = w.repeat(p['ny']+1, axis=0)
            w = w.repeat(p['nx']+1, axis=1)
    else:
        w = w_init.copy()

    # set model state properties that are added to warnings and errors
    logprops = dict(minwind=s['uw'].min(),
                    maxdrop=(l['uw']-s['uw']).max(),
                    time=self.t,
                    dt=self.dt)
        
    nf = p['nfractions']
    
    us = np.zeros((p['ny']+1,p['nx']+1))
    un = np.zeros((p['ny']+1,p['nx']+1))
    
    us_plus = np.zeros((p['ny']+1,p['nx']+1))
    un_plus = np.zeros((p['ny']+1,p['nx']+1))
    
    us_min = np.zeros((p['ny']+1,p['nx']+1))
    un_min = np.zeros((p['ny']+1,p['nx']+1))

    Cs = np.zeros(us.shape)
    Cn = np.zeros(un.shape)
    
    Cs_plus = np.zeros(us.shape)
    Cn_plus = np.zeros(un.shape)
    
    Cs_min = np.zeros(us.shape)
    Cn_min = np.zeros(un.shape)
    
    
    for i in range(nf):
        
        us[:,:] = s['us'][:,:,i] 
        un[:,:] = s['un'][:,:,i] 
        
        us_plus[:,1:] = s['us'][:,:-1,i] 
        un_plus[1:,:] = s['un'][:-1,:,i] 
        
        us_min[:,:-1] = s['us'][:,1:,i]
        un_min[:-1,:] = s['un'][1:,:,i]
    
        #boundary values            
        us_plus[:,0]  = s['us'][:,0,i]
        un_plus[0,:]  = s['un'][0,:,i]
        
        us_min[:,-1]  = s['us'][:,-1,i]
        un_min[-1,:]  = s['un'][-1,:,i]
        
        
        # define matrix coefficients to solve linear system of equations        
        Cs = self.dt * s['dn'] * s['dsdni'] * us[:,:]  
        Cn = self.dt * s['ds'] * s['dsdni'] * un[:,:] 

        Cs_plus = self.dt * s['dn'] * s['dsdni'] * us_plus[:,:]  
        Cn_plus = self.dt * s['ds'] * s['dsdni'] * un_plus[:,:]
        
        Cs_min = self.dt * s['dn'] * s['dsdni'] * us_min[:,:]  
        Cn_min = self.dt * s['ds'] * s['dsdni'] * un_min[:,:]
        
        Ti = self.dt / p['T']          

        
        beta = abs(beta)
        if beta >= 1.:
            # define upwind direction
            ixs = np.asarray(s['us'][:,:,i] >= 0., dtype=float)
            ixn = np.asarray(s['un'][:,:,i] >= 0., dtype=float)
            sgs = 2. * ixs - 1.
            sgn = 2. * ixn - 1.
        
        else:
            # or centralizing weights
            ixs = beta + np.zeros(Cs.shape)
            ixn = beta + np.zeros(Cn.shape)
            sgs = np.zeros(Cs.shape)
            sgn = np.zeros(Cn.shape)
            
        # initialize matrix diagonals
        A0 = np.zeros(s['zb'].shape)
        Apx = np.zeros(s['zb'].shape)
        Ap1 = np.zeros(s['zb'].shape)
        Ap2 = np.zeros(s['zb'].shape)
        Amx = np.zeros(s['zb'].shape)
        Am1 = np.zeros(s['zb'].shape)
        Am2 = np.zeros(s['zb'].shape)

        # populate matrix diagonals
        A0  = 1. + (sgs * Cs + sgn * Cn + Ti) * alpha
        Apx = Cn_min * alpha * (1. - ixn)
        Ap1 = Cs_min * alpha * (1. - ixs)
        Amx = -Cn_plus * alpha * ixn
        Am1 = -Cs_plus * alpha * ixs    

        # add boundaries
        A0[:,0] = 1.
        Apx[:,0] = 0.
        Amx[:,0] = 0.
        Am2[:,0] = 0.
        Am1[:,0] = 0.

        A0[:,-1] = 1.
        Apx[:,-1] = 0.
        Ap1[:,-1] = 0.
        Ap2[:,-1] = 0.
        Amx[:,-1] = 0.

        if (p['boundary_offshore'] == 'flux') | (p['boundary_offshore'] == 'noflux'):
            Ap2[:,0] = 0.
            Ap1[:,0] = 0.
        elif p['boundary_offshore'] == 'constant':
            Ap2[:,0] = 0.
            Ap1[:,0] = 0.
        elif p['boundary_offshore'] == 'uniform':
            Ap2[:,0] = 0.
            Ap1[:,0] = -1.
        elif p['boundary_offshore'] == 'gradient':
            Ap2[:,0] = s['ds'][:,1] / s['ds'][:,2]
            Ap1[:,0] = -1. - s['ds'][:,1] / s['ds'][:,2]
        elif p['boundary_offshore'] == 'circular':
            logger.log_and_raise('Cross-shore cricular boundary condition not yet implemented', exc=NotImplementedError)
        else:
            logger.log_and_raise('Unknown offshore boundary condition [%s]' % self.p['boundary_offshore'], exc=ValueError)

        if (p['boundary_onshore'] == 'flux') | (p['boundary_offshore'] == 'noflux'):                              
            Am2[:,-1] = 0.
            Am1[:,-1] = 0.            
        elif p['boundary_onshore'] == 'constant':                              
            Am2[:,-1] = 0.
            Am1[:,-1] = 0.
        elif p['boundary_onshore'] == 'uniform':
            Am2[:,-1] = 0.
            Am1[:,-1] = -1.
        elif p['boundary_onshore'] == 'gradient':
            Am2[:,-1] = s['ds'][:,-2] / s['ds'][:,-3]
            Am1[:,-1] = -1. - s['ds'][:,-2] / s['ds'][:,-3]
        elif p['boundary_offshore'] == 'circular':
            logger.log_and_raise('Cross-shore cricular boundary condition not yet implemented', exc=NotImplementedError)
        else:
            logger.log_and_raise('Unknown onshore boundary condition [%s]' % self.p['boundary_onshore'], exc=ValueError)

        if p['boundary_lateral'] == 'constant':
            A0[0,:] = 1.
            Apx[0,:] = 0.
            Ap1[0,:] = 0.
            Amx[0,:] = 0.
            Am1[0,:] = 0.
            
            A0[-1,:] = 1.
            Apx[-1,:] = 0.
            Ap1[-1,:] = 0.
            Amx[-1,:] = 0.
            Am1[-1,:] = 0.
        
            #logger.log_and_raise('Lateral constant boundary condition not yet implemented', exc=NotImplementedError)
        elif p['boundary_lateral'] == 'uniform':
            logger.log_and_raise('Lateral uniform boundary condition not yet implemented', exc=NotImplementedError)
        elif p['boundary_lateral'] == 'gradient':
            logger.log_and_raise('Lateral gradient boundary condition not yet implemented', exc=NotImplementedError)
        elif p['boundary_lateral'] == 'circular':
            pass
        else:
            logger.log_and_raise('Unknown lateral boundary condition [%s]' % self.p['boundary_lateral'], exc=ValueError)

        # construct sparse matrix
        if p['ny'] > 0:
            j = p['nx']+1
            A = scipy.sparse.diags((Apx.flatten()[:j],
                                    Amx.flatten()[j:],
                                    Am2.flatten()[2:],
                                    Am1.flatten()[1:],
                                    A0.flatten(),
                                    Ap1.flatten()[:-1],
                                    Ap2.flatten()[:-2],
                                    Apx.flatten()[j:],
                                    Amx.flatten()[:j]),
                                    (-j*p['ny'],-j,-2,-1,0,1,2,j,j*p['ny']), format='csr')
        else:
            A = scipy.sparse.diags((Am2.flatten()[2:],
                                    Am1.flatten()[1:],
                                    A0.flatten(),
                                    Ap1.flatten()[:-1],
                                    Ap2.flatten()[:-2]),
                                    (-2,-1,0,1,2), format='csr')

        # solve transport for each fraction separately using latest
        # available weights

        # renormalize weights for all fractions equal or larger
        # than the current one such that the sum of all weights is
        # unity
        # Christa: seems to have no significant effect on weights, 
        # numerical check to prevent any deviation from unity
        w = aeolis.transport.renormalize_weights(w, i)            

        # iteratively find a solution of the linear system that
        # does not violate the availability of sediment in the bed
        for n in range(p['max_iter']):
            self._count('matrixsolve')

            # compute saturation levels
            ix = s['Cu'] > 0.
            S_i = np.zeros(s['Cu'].shape)
            S_i[ix] = s['Ct'][ix] / s['Cu'][ix]
            s['S'] = S_i.sum(axis=-1)

            # create the right hand side of the linear system
            y_i = np.zeros(s['zb'].shape)
            y_im = np.zeros(s['zb'].shape)  # implicit terms
            y_ex = np.zeros(s['zb'].shape)  # explicit terms
            
            y_im[:,1:-1] = (
                (w[:,1:-1,i] * s['Cuf'][:,1:-1,i] * Ti) * (1. - s['S'][:,1:-1]) +
                (w[:,1:-1,i] * s['Cu'][:,1:-1,i] * Ti) * s['S'][:,1:-1]
                )
            
            y_ex[:,1:-1] = (
                (l['w'][:,1:-1,i] * l['Cuf'][:,1:-1,i] * Ti) * (1. - s['S'][:,1:-1]) \
                + (l['w'][:,1:-1,i] * l['Cu'][:,1:-1,i] * Ti) * s['S'][:,1:-1] \
                - (
                    sgs[:,1:-1] * Cs[:,1:-1] +\
                    sgn[:,1:-1] * Cn[:,1:-1] + Ti
                ) * l['Ct'][:,1:-1,i] \
                + ixs[:,1:-1] * Cs_plus[:,1:-1] * l['Ct'][:,:-2,i] \
                - (1. - ixs[:,1:-1]) * Cs_min[:,1:-1] * l['Ct'][:,2:,i] \
                + ixn[:,1:-1] * Cn_plus[:,1:-1] * np.roll(l['Ct'][:,1:-1,i], 1, axis=0) \
                - (1. - ixn[:,1:-1]) * Cn_min[:,1:-1] * np.roll(l['Ct'][:,1:-1,i], -1, axis=0) \
                )
            
            y_i[:,1:-1] = l['Ct'][:,1:-1,i] + alpha * y_im[:,1:-1] + (1. - alpha) * y_ex[:,1:-1]

            # add boundaries
            if p['boundary_offshore'] == 'flux':
                y_i[:,0] = p['offshore_flux'] * s['Cu0'][:,0,i] 
            if p['boundary_onshore'] == 'flux':
                y_i[:,-1] = p['onshore_flux'] * s['Cu0'][:,-1,i] 
                
            if p['boundary_offshore'] == 'constant':
                y_i[:,0] = p['constant_offshore_flux'] / s['u'][:,0,i] 
            if p['boundary_onshore'] == 'constant':
                y_i[:,-1] = p['constant_onshore_flux'] / s['u'][:,-1,i]

            # solve system with current weights
            Ct_i = scipy.sparse.linalg.spsolve(A, y_i.flatten())
            Ct_i = prevent_tiny_negatives(Ct_i, p['max_error'])
            
            # check for negative values
            if Ct_i.min() < 0.:
                ix = Ct_i < 0.

                logger.warning(format_log('Removing negative concentrations',
                                            nrcells=np.sum(ix),
                                            fraction=i,
                                            iteration=n,
                                            minvalue=Ct_i.min(),
                                            coords=np.argwhere(ix.reshape(y_i.shape)),
                                            **logprops))

                if Ct_i[~ix].sum() != 0:
                    Ct_i[~ix] *= 1. + Ct_i[ix].sum() / Ct_i[~ix].sum()
                else:
                    Ct_i[~ix] = 0

                #Ct_i[~ix] *= 1. + Ct_i[ix].sum() / Ct_i[~ix].sum()
                Ct_i[ix] = 0.

            # determine pickup and deficit for current fraction
            Cu_i = s['Cu'][:,:,i].flatten()
            mass_i = s['mass'][:,:,0,i].flatten()
            w_i = w[:,:,i].flatten()
            pickup_i = (w_i * Cu_i - Ct_i) / p['T'] * self.dt
            deficit_i = pickup_i - mass_i
            ix = (deficit_i > p['max_error']) \
                    & (w_i * Cu_i > 0.)

            # quit the iteration if there is no deficit, otherwise
            # back-compute the maximum weight allowed to get zero
            # deficit for the current fraction and progress to
            # the next iteration step
            if not np.any(ix):
                logger.debug(format_log('Iteration converged',
                                        steps=n,
                                        fraction=i,
                                        **logprops))
                pickup_i = np.minimum(pickup_i, mass_i)
                break
            else:
                w_i[ix] = (mass_i[ix] * p['T'] / self.dt \
                            + Ct_i[ix]) / Cu_i[ix]
                w[:,:,i] = w_i.reshape(y_i.shape)

        # throw warning if the maximum number of iterations was reached
        if np.any(ix):
            logger.warning(format_log('Iteration not converged',
                                        nrcells=np.sum(ix),
                                        fraction=i,
                                        **logprops))

        # check for unexpected negative values
        if Ct_i.min() < 0:
            logger.warning(format_log('Negative concentrations',
                                        nrcells=np.sum(Ct_i<0.),
                                        fraction=i,
                                        minvalue=Ct_i.min(),
                                        **logprops))
        if w_i.min() < 0:
            logger.warning(format_log('Negative weights',
                                        nrcells=np.sum(w_i<0),
                                        fraction=i,
                                        minvalue=w_i.min(),
                                        **logprops))

        Ct[:,:,i] = Ct_i.reshape(y_i.shape)
        pickup[:,:,i] = pickup_i.reshape(y_i.shape)

    # check if there are any cells where the sum of all weights is
    # smaller than unity. these cells are supply-limited for all
    # fractions. Log these events.
    ix = 1. - np.sum(w, axis=2) > p['max_error']
    if np.any(ix):
        self._count('supplylim')
        # logger.warning(format_log('Ran out of sediment',
        #                           nrcells=np.sum(ix),
        #                           minweight=np.sum(w, axis=-1).min(),
        #                           **logprops))
        
    qs = Ct * s['us'] 
    qn = Ct * s['un'] 

    return dict(Ct=Ct,
                qs=qs,
                qn=qn,
                pickup=pickup,
                w=w,
                w_init=w_init,
                w_air=w_air,
                w_bed=w_bed)
    
#@njit
def solve_EF(self, alpha:float=0., beta:float=1.) -> dict:
    '''Implements the explicit Euler forward, implicit Euler backward and semi-implicit Crank-Nicolson numerical schemes

    Determines weights of sediment fractions, sediment pickup and
    instantaneous sediment concentration. Returns a partial
    spatial grid dictionary that can be used to update the global
    spatial grid dictionary.

    Parameters
    ----------
    alpha :
        Implicitness coefficient (0.0 for Euler forward, 1.0 for Euler backward or 0.5 for Crank-Nicolson, default=0.5)
    beta : 
        Centralization coefficient (1.0 for upwind or 0.5 for centralized, default=1.0)

    Returns
    -------
        Partial spatial grid dictionary

    Examples
    --------
    >>> model.s.update(model.solve(alpha=1., beta=1.) # euler backward

    >>> model.s.update(model.solve(alpha=.5, beta=1.) # crank-nicolson

    See Also
    --------
    model.AeoLiS.euler_forward
    model.AeoLiS.euler_backward
    model.AeoLiS.crank_nicolson
    transport.compute_weights
    transport.renormalize_weights

    '''

    l = self.l
    s = self.s
    p = self.p

    Ct = s['Ct'].copy()
    pickup = s['pickup'].copy()
    Ts = p['T']

    # compute transport weights for all sediment fractions
    w_init, w_air, w_bed = aeolis.transport.compute_weights(s, p)

    if self.t == 0.:
        if type(p['bedcomp_file']) == np.ndarray:
            w = w_init.copy()
        else:
            # use initial guess for first time step
            w = p['grain_dist'].reshape((1,1,-1))
            w = w.repeat(p['ny']+1, axis=0)
            w = w.repeat(p['nx']+1, axis=1)
    else:
        w = w_init.copy()

    # set model state properties that are added to warnings and errors
    logprops = dict(minwind=s['uw'].min(),
                    maxdrop=(l['uw']-s['uw']).max(),
                    time=self.t,
                    dt=self.dt)
        
    nf = p['nfractions']
        
    
    for i in range(nf):

        if 1:
            #define 4 quadrants based on wind directions
            ix1 = ((s['us'][:,:,0]>=0) & (s['un'][:,:,0]>=0))
            ix2 = ((s['us'][:,:,0]<0) & (s['un'][:,:,0]>=0))
            ix3 = ((s['us'][:,:,0]<0) & (s['un'][:,:,0]<0))
            ix4 = ((s['us'][:,:,0]>0) & (s['un'][:,:,0]<0))
            
            # initiate solution matrix including ghost cells to accomodate boundaries
            Ct_s = np.zeros((Ct.shape[0]+2,Ct.shape[1]+2))
            # populate solution matrix with previous concentration results
            Ct_s[1:-1,1:-1] = Ct[:,:,i]
            
            #set upwind boundary condition
            Ct_s[:,0:2]=0
            #circular boundary condition in lateral directions
            Ct_s[0,:]=Ct_s[-2,:]
            Ct_s[-1,:]=Ct_s[1,:]
            # using the Euler forward scheme we can calculate pickup first based on the previous timestep
            # there is no need for iteration
            pickup[:,:,i] = self.dt*(np.minimum(s['Cu'][:,:,i],s['mass'][:,:,0,i]+Ct[:,:,i])-Ct[:,:,i])/Ts
            
            #solve for all 4 quadrants in one step using logical indexing
            Ct_s[1:-1,1:-1] = Ct_s[1:-1,1:-1] + \
                ix1*(-self.dt*s['us'][:,:,i]*(Ct_s[1:-1,1:-1]-Ct_s[1:-1,:-2])/s['ds'] \
                        -self.dt*s['un'][:,:,i]*(Ct_s[1:-1,1:-1]-Ct_s[:-2,1:-1])/s['dn']) +\
                ix2*(+self.dt*s['us'][:,:,i]*(Ct_s[1:-1,1:-1]-Ct_s[1:-1,2:])/s['ds'] \
                        -self.dt*s['un'][:,:,i]*(Ct_s[1:-1,1:-1]-Ct_s[:-2,1:-1])/s['dn']) +\
                ix3*(+self.dt*s['us'][:,:,i]*(Ct_s[1:-1,1:-1]-Ct_s[1:-1,2:])/s['ds'] \
                        +self.dt*s['un'][:,:,i]*(Ct_s[1:-1,1:-1]-Ct_s[2:,1:-1])/s['dn']) +\
                ix4*(-self.dt*s['us'][:,:,i]*(Ct_s[1:-1,1:-1]-Ct_s[1:-1,:-2])/s['ds'] \
                        +self.dt*s['un'][:,:,i]*(Ct_s[1:-1,1:-1]-Ct_s[2:,1:-1])/s['dn']) \
                + pickup[:,:,i]
            
            # define Ct as a subset of Ct_s (eliminating the boundaries)
            Ct[:,:,i] = Ct_s[1:-1,1:-1] 
        
    qs = Ct * s['us'] 
    qn = Ct * s['un'] 

    return dict(Ct=Ct,
                qs=qs,
                qn=qn,
                pickup=pickup,
                w=w,
                w_init=w_init,
                w_air=w_air,
                w_bed=w_bed)
    
#@njit
def solve_SS(self, alpha:float=0., beta:float=1.) -> dict:
    '''Implements the explicit Euler forward, implicit Euler backward and semi-implicit Crank-Nicolson numerical schemes

    Determines weights of sediment fractions, sediment pickup and
    instantaneous sediment concentration. Returns a partial
    spatial grid dictionary that can be used to update the global
    spatial grid dictionary.

    Parameters
    ----------
    alpha :
        Implicitness coefficient (0.0 for Euler forward, 1.0 for Euler backward or 0.5 for Crank-Nicolson, default=0.5)
    beta : 
        Centralization coefficient (1.0 for upwind or 0.5 for centralized, default=1.0)

    Returns
    -------
        Partial spatial grid dictionary

    Examples
    --------
    >>> model.s.update(model.solve(alpha=1., beta=1.) # euler backward

    >>> model.s.update(model.solve(alpha=.5, beta=1.) # crank-nicolson

    See Also
    --------
    model.AeoLiS.euler_forward
    model.AeoLiS.euler_backward
    model.AeoLiS.crank_nicolson
    transport.compute_weights
    transport.renormalize_weights

    '''

    l = self.l
    s = self.s
    p = self.p

    Ct = s['Ct'].copy()
    pickup = s['pickup'].copy()
    Ts = p['T']

    # compute transport weights for all sediment fractions
    w_init, w_air, w_bed = aeolis.transport.compute_weights(s, p)

    if self.t == 0.:
        if type(p['bedcomp_file']) == np.ndarray:
            w = w_init.copy()
        else:
            # use initial guess for first time step
            # when p['grain_dist'] has 2 dimensions take the first row otherwise take the only row
            if len(p['grain_dist'].shape) == 2:
                w = p['grain_dist'][0,:].reshape((1,1,-1))
            else:
                w = p['grain_dist'].reshape((1,1,-1))
                
            w = w.repeat(p['ny']+1, axis=0)
            w = w.repeat(p['nx']+1, axis=1)
    else:
        w = w_init.copy()

    # set model state properties that are added to warnings and errors
    logprops = dict(minwind=s['uw'].min(),
                    maxdrop=(l['uw']-s['uw']).max(),
                    time=self.t,
                    dt=self.dt)
        
    nf = p['nfractions']
        
    # LOOPING HAPPENS IN SWEEP FUNCTION NOW (RIGHT?)
    # for i in range(nf):
        

        # FIX: WHY ONLY FIRST FRACTION? 
        # WHERE ARE THE BOUNDARIES SET?
        # Just changed 0 to i...
        # Constant --> Neumann?

    # if 1:
        #print('sweep')

    # initiate emmpty solution matrix, this will effectively kill time dependence and create steady state.
    Ct = np.zeros(Ct.shape)
    
    # Flux boundary conditions
    if p['boundary_offshore'] == 'flux':
        Ct[:,0,:] =  p['offshore_flux'] * s['Cu0'][:,0,:] # s['Cu'][:,0,i]?
    if p['boundary_onshore'] == 'flux':
        Ct[:,-1,:] = p['onshore_flux'] *  s['Cu0'][:,-1,:] # s['Cu0']
    if p['boundary_lateral'] == 'flux':
        Ct[0,:,:] = p['lateral_flux'] *  s['Cu0'][0,:,:] # s['Cu0']
        Ct[-1,:,:] = p['lateral_flux'] *  s['Cu0'][-1,:,:] # s['Cu0']

    # # Circular and re-circular boundary conditions
    # if p['boundary_offshore'] == 'circular':
    #     Ct[:,0,0] =  -1                
    #     Ct[:,-1,0] =  -1                  
    # if p['boundary_offshore'] == 're_circular':
    #     Ct[:,0,0] =  -2                
    #     Ct[:,-1,0] =  -2         
    # if p['boundary_lateral'] == 'circular':
    #     Ct[0,:,0] =  -1                
    #     Ct[-1,:,0] =  -1
    # if p['boundary_lateral'] == 're_circular':
    #     Ct[0,:,0] =  -2                
    #     Ct[-1,:,0] =  -2

    # Constant boundary conditions
    if p['boundary_offshore'] == 'constant':
        Ct[:,0,:] = Ct[:,1,:]
    if p['boundary_onshore'] == 'constant':
        Ct[:,-1,:] = Ct[:,-2,:]
    if p['boundary_lateral'] == 'constant':
        Ct[0,:,:] = Ct[1,:,:]
        Ct[-1,:,:] = Ct[-2,:,:]

    onshore_bc = p['boundary_onshore']
    offshore_bc = p['boundary_offshore']
    lateral_bc = p['boundary_lateral']

    Cu, Ct, pickup = sweep(Ct, s['CuBed'].copy(), s['CuAir'].copy(), 
                        s['zeta'].copy(), s['mass'].copy(), 
                        self.dt, p['T'], 
                        s['ds'], s['dn'], s['us'], s['un'], w,
                        onshore_bc, offshore_bc, lateral_bc, s['uws'][0,0], s['uwn'][0,0])

    qs = Ct * s['us'] 
    qn = Ct * s['un'] 
    q = np.hypot(qs, qn)


    return dict(Cu=Cu,
                Ct=Ct,
                qs=qs,
                qn=qn,
                pickup=pickup,
                w=w,
                w_init=w_init,
                w_air=w_air,
                w_bed=w_bed,
                q=q)
    
    
def solve_steadystatepieter(self) -> dict:
    
    beta = 1. 
    
    l = self.l
    s = self.s
    p = self.p

    Ct = s['Ct'].copy()
    qs = s['qs'].copy()
    qn = s['qn'].copy()
    pickup = s['pickup'].copy()
    
    Ts = p['T']
    
    # compute transport weights for all sediment fractions
    w_init, w_air, w_bed = aeolis.transport.compute_weights(s, p)

    if self.t == 0.:
        # use initial guess for first time step
        w = p['grain_dist'].reshape((1,1,-1))
        w = w.repeat(p['ny']+1, axis=0)
        w = w.repeat(p['nx']+1, axis=1)
        return dict(w=w)
    else:
        w = w_init.copy()

    # set model state properties that are added to warnings and errors
    logprops = dict(minwind=s['uw'].min(),
                    maxdrop=(l['uw']-s['uw']).max(),
                    time=self.t,
                    dt=self.dt)
    
    nf = p['nfractions']
    
    ufs = np.zeros((p['ny']+1,p['nx']+2))
    ufn = np.zeros((p['ny']+2,p['nx']+1))    
    
    for i in range(nf): #loop over fractions
    
        #define velocity fluxes
        
        ufs[:,1:-1] = 0.5*s['us'][:,:-1,i] + 0.5*s['us'][:,1:,i]
        ufn[1:-1,:] = 0.5*s['un'][:-1,:,i] + 0.5*s['un'][1:,:,i]
        
        #boundary values
        ufs[:,0]  = s['us'][:,0,i]
        ufs[:,-1] = s['us'][:,-1,i]
        
        if p['boundary_lateral'] == 'circular':
            ufn[0,:] = 0.5*s['un'][0,:,i] + 0.5*s['un'][-1,:,i]
            ufn[-1,:] = ufn[0,:]
        else:
            ufn[0,:]  = s['un'][0,:,i]
            ufn[-1,:] = s['un'][-1,:,i]
    
        beta = abs(beta)
        if beta >= 1.:
            # define upwind direction
            ixfs = np.asarray(ufs >= 0., dtype=float)
            ixfn = np.asarray(ufn >= 0., dtype=float)
        else:
            # or centralizing weights
            ixfs = beta + np.zeros(ufs)
            ixfn = beta + np.zeros(ufn)

        # initialize matrix diagonals
        A0 = np.zeros(s['zb'].shape)
        Apx = np.zeros(s['zb'].shape)
        Ap1 = np.zeros(s['zb'].shape)
        Amx = np.zeros(s['zb'].shape)
        Am1 = np.zeros(s['zb'].shape)

        # populate matrix diagonals
        #A0         += s['dsdn'] / self.dt                                        #time derivative
        A0         += s['dsdn'] / Ts                                        #source term
        A0[:,1:]   -= s['dn'][:,1:]  * ufs[:,1:-1] * (1. - ixfs[:,1:-1])    #lower x-face
        Am1[:,1:]  -= s['dn'][:,1:]  * ufs[:,1:-1] *       ixfs[:,1:-1]     #lower x-face
        A0[:,:-1]  += s['dn'][:,:-1] * ufs[:,1:-1] *       ixfs[:,1:-1]     #upper x-face
        Ap1[:,:-1] += s['dn'][:,:-1] * ufs[:,1:-1] * (1. - ixfs[:,1:-1])    #upper x-face
        A0[1:,:]   -= s['ds'][1:,:]  * ufn[1:-1,:] * (1. - ixfn[1:-1,:])    #lower y-face
        Amx[1:,:]  -= s['ds'][1:,:]  * ufn[1:-1,:] *       ixfn[1:-1,:]     #lower y-face
        A0[:-1,:]  += s['ds'][:-1,:] * ufn[1:-1,:] *       ixfn[1:-1,:]     #upper y-face
        Apx[:-1,:] += s['ds'][:-1,:] * ufn[1:-1,:] * (1. - ixfn[1:-1,:])    #upper y-face
    
        # add boundaries
        # offshore boundary (i=0)

        if p['boundary_offshore'] == 'flux':
            #nothing to be done
            pass
        elif p['boundary_offshore'] == 'constant':
            #constant sediment concentration (Ct) in the air
            A0[:,0] = 1.
            Apx[:,0] = 0.
            Amx[:,0] = 0.
            Ap1[:,0] = 0.
            Am1[:,0] = 0.
        elif p['boundary_offshore'] == 'gradient':
            #remove the flux at the inner face of the cell
            A0[:,0]  -= s['dn'][:,0] * ufs[:,1] *       ixfs[:,1]           #upper x-face
            Ap1[:,0] -= s['dn'][:,0] * ufs[:,1] * (1. - ixfs[:,1])          #upper x-face
        elif p['boundary_offshore'] == 'circular':
            raise NotImplementedError('Cross-shore cricular boundary condition not yet implemented')
        else:
            raise ValueError('Unknown offshore boundary condition [%s]' % self.p['boundary_offshore'])

        #onshore boundary (i=nx)

        if p['boundary_onshore'] == 'flux':
            #nothing to be done
            pass
        elif p['boundary_onshore'] == 'constant':
            #constant sediment concentration (hC) in the air
            A0[:,-1] = 1.
            Apx[:,-1] = 0.
            Amx[:,-1] = 0.
            Ap1[:,-1] = 0.
            Am1[:,-1] = 0.
        elif p['boundary_onshore'] == 'gradient':
            #remove the flux at the inner face of the cell
            A0[:,-1]   += s['dn'][:,-1]  * ufs[:,-2]   * (1. - ixfs[:,-2])      #lower x-face
            Am1[:,-1]  += s['dn'][:,-1]  * ufs[:,-2]   *       ixfs[:,-2]       #lower x-face
        elif p['boundary_onshore'] == 'circular':
            raise NotImplementedError('Cross-shore cricular boundary condition not yet implemented')
        else:
            raise ValueError('Unknown offshore boundary condition [%s]' % self.p['boundary_onshore'])
    
        #lateral boundaries (j=0; j=ny)    

        if p['boundary_lateral'] == 'flux':
            #nothing to be done
            pass
        elif p['boundary_lateral'] == 'constant':
            #constant sediment concentration (hC) in the air
            A0[0,:] = 1.
            Apx[0,:] = 0.
            Amx[0,:] = 0.
            Ap1[0,:] = 0.
            Am1[0,:] = 0.
            A0[-1,:] = 1.
            Apx[-1,:] = 0.
            Amx[-1,:] = 0.
            Ap1[-1,:] = 0.
            Am1[-1,:] = 0.
        elif p['boundary_lateral'] == 'gradient':
            #remove the flux at the inner face of the cell
            A0[0,:]   -= s['ds'][0,:] * ufn[1,:]   *       ixfn[1,:]        #upper y-face
            Apx[0,:]  -= s['ds'][0,:] * ufn[1,:]   * (1. - ixfn[1,:])       #upper y-face
            A0[-1,:]  += s['ds'][-1,:] * ufn[-2,:] * (1. - ixfn[-2,:])      #lower y-face
            Amx[-1,:] += s['ds'][-1,:] * ufn[-2,:] *       ixfn[-2,:]       #lower y-face
        elif p['boundary_lateral'] == 'circular':   
            A0[0,:]   -= s['ds'][0,:]  * ufn[0,:]  * (1. - ixfn[0,:])       #lower y-face
            Amx[0,:]  -= s['ds'][0,:]  * ufn[0,:]  *       ixfn[0,:]        #lower y-face
            A0[-1,:]  += s['ds'][-1,:] * ufn[-1,:] *       ixfn[-1,:]       #upper y-face
            Apx[-1,:] += s['ds'][-1,:] * ufn[-1,:] * (1. - ixfn[-1,:])      #upper y-face
        else:
            raise ValueError('Unknown lateral boundary condition [%s]' % self.p['boundary_lateral'])
        
        # construct sparse matrix
        if p['ny'] > 0:
            j = p['nx']+1
            A = scipy.sparse.diags((Apx.flatten()[:j],
                                    Amx.flatten()[j:],
                                    Am1.flatten()[1:],
                                    A0.flatten(),
                                    Ap1.flatten()[:-1],
                                    Apx.flatten()[j:],
                                    Amx.flatten()[:j]),
                                    (-j*p['ny'],-j,-1,0,1,j,j*p['ny']), format='csr')
        else:
            j = p['nx']+1
            ny = 0
            A = scipy.sparse.diags((Am1.flatten()[1:],
                                    A0.flatten(),
                                    Ap1.flatten()[:-1]),
                                    (-1, 0, 1), format='csr')

        # solve transport for each fraction separately using latest
        # available weights


        # renormalize weights for all fractions equal or larger
        # than the current one such that the sum of all weights is
        # unity
        w = aeolis.transport.renormalize_weights(w, i)

        # iteratively find a solution of the linear system that
        # does not violate the availability of sediment in the bed
        for n in range(p['max_iter']):
            self._count('matrixsolve')
            
            # define upwind face value
            # sediment concentration
            Ctxfs_i = np.zeros(ufs.shape)
            Ctxfn_i = np.zeros(ufn.shape)
            
            Ctxfs_i[:,1:-1] = ixfs[:,1:-1] * Ct[:,:-1,i] \
                                + (1. - ixfs[:,1:-1]) * Ct[:,1:,i] 
            Ctxfn_i[1:-1,:] = ixfn[1:-1,:] * Ct[:-1,:,i] \
                                + (1. - ixfn[1:-1,:]) * Ct[1:,:,i] 

            if p['boundary_lateral'] == 'circular':
                Ctxfn_i[0,:] = ixfn[0,:] * Ct[-1,:,i] \
                                + (1. - ixfn[0,:]) *  Ct[0,:,i] 
            
            # calculate pickup
            D_i = s['dsdn'] / Ts * Ct[:,:,i]                                          
            A_i = s['dsdn'] / Ts * s['mass'][:,:,0,i] + D_i # Availability
            U_i = s['dsdn'] / Ts *  w[:,:,i] *  s['Cu'][:,:,i] 
                                        
            #deficit_i = E_i - A_i
            E_i= np.minimum(U_i, A_i)
            #pickup_i = E_i - D_i

            # create the right hand side of the linear system
            # sediment concentration
            yCt_i = np.zeros(s['zb'].shape)
                            
            yCt_i         += E_i - D_i                                      #source term
            yCt_i[:,1:]   += s['dn'][:,1:]  * ufs[:,1:-1] * Ctxfs_i[:,1:-1] #lower x-face
            yCt_i[:,:-1]  -= s['dn'][:,:-1] * ufs[:,1:-1] * Ctxfs_i[:,1:-1] #upper x-face
            yCt_i[1:,:]   += s['ds'][1:,:]  * ufn[1:-1,:] * Ctxfn_i[1:-1,:] #lower y-face
            yCt_i[:-1,:]  -= s['ds'][:-1,:] * ufn[1:-1,:] * Ctxfn_i[1:-1,:] #upper y-face
            
            # boundary conditions
            # offshore boundary (i=0)

            if p['boundary_offshore'] == 'flux':
                yCt_i[:,0]  += s['dn'][:,0] * ufs[:,0] * s['Cu0'][:,0,i] * p['offshore_flux'] 
            elif p['boundary_offshore'] == 'constant':
                #constant sediment concentration (Ct) in the air 
                yCt_i[:,0]  = p['constant_offshore_flux']

            elif p['boundary_offshore'] == 'gradient':
                #remove the flux at the inner face of the cell
                yCt_i[:,0]  += s['dn'][:,1] * ufs[:,1] * Ctxfs_i[:,1] 

            elif p['boundary_offshore'] == 'circular':
                raise NotImplementedError('Cross-shore cricular boundary condition not yet implemented')
            else:
                raise ValueError('Unknown offshore boundary condition [%s]' % self.p['boundary_offshore'])
                
            # onshore boundary (i=nx)

            if p['boundary_onshore'] == 'flux':
                yCt_i[:,-1]  += s['dn'][:,-1]  * ufs[:,-1] * s['Cu0'][:,-1,i] * p['onshore_flux']

            elif p['boundary_onshore'] == 'constant':
                #constant sediment concentration (Ct) in the air 
                yCt_i[:,-1]  = p['constant_onshore_flux']

            elif p['boundary_onshore'] == 'gradient':
                #remove the flux at the inner face of the cell
                yCt_i[:,-1]  -= s['dn'][:,-2] * ufs[:,-2] * Ctxfs_i[:,-2] 

            elif p['boundary_onshore'] == 'circular':
                raise NotImplementedError('Cross-shore cricular boundary condition not yet implemented')
            else:
                raise ValueError('Unknown onshore boundary condition [%s]' % self.p['boundary_onshore'])
                
            #lateral boundaries (j=0; j=ny)    

            if p['boundary_lateral'] == 'flux':
                
                yCt_i[0,:]   += s['ds'][0,:] * ufn[0,:]  * s['Cu0'][0,:,i] * p['lateral_flux'] #lower y-face
                yCt_i[-1,:]  -= s['ds'][-1,:] * ufn[-1,:] * s['Cu0'][-1,:,i] * p['lateral_flux'] #upper y-face                    
            elif p['boundary_lateral'] == 'constant':
                #constant sediment concentration (hC) in the air
                yCt_i[0,:]  = 0.
                yCt_i[-1,:] = 0.
            elif p['boundary_lateral'] == 'gradient':
                #remove the flux at the inner face of the cell
                yCt_i[-1,:] -= s['ds'][-2,:] * ufn[-2,:] * Ctxfn_i[-2,:] #lower y-face
                yCt_i[0,:]  += s['ds'][1,:]  * ufn[1,:]  * Ctxfn_i[1,:]  #upper y-face
            elif p['boundary_lateral'] == 'circular':
                yCt_i[0,:]  += s['ds'][0,:]  * ufn[0,:]  * Ctxfn_i[0,:]  #lower y-face
                yCt_i[-1,:] -= s['ds'][-1,:] * ufn[-1,:] * Ctxfn_i[-1,:] #upper y-face
            else:
                raise ValueError('Unknown lateral boundary condition [%s]' % self.p['boundary_lateral'])
            
            # print("ugs = %.*g" % (3,s['ugs'][10,10]))
            # print("ugn = %.*g" % (3,s['ugn'][10,10]))
            # print("%.*g" % (3,np.amax(np.absolute(y_i))))
            
            # solve system with current weights
            Ct_i = Ct[:,:,i].flatten()
            Ct_i += scipy.sparse.linalg.spsolve(A, yCt_i.flatten())
            Ct_i = prevent_tiny_negatives(Ct_i, p['max_error'])
            
            # check for negative values
            if Ct_i.min() < 0.:
                ix = Ct_i < 0.
                
#                    logger.warn(format_log('Removing negative concentrations',
#                                           nrcells=np.sum(ix),
#                                           fraction=i,
#                                           iteration=n,
#                                           minvalue=Ct_i.min(),
#                                           **logprops))

                Ct_i[~ix] *= 1. + Ct_i[ix].sum() / Ct_i[~ix].sum()
                Ct_i[ix] = 0.

            # determine pickup and deficit for current fraction
            Cu_i = s['Cu'][:,:,i].flatten()
            mass_i = s['mass'][:,:,0,i].flatten()
            w_i = w[:,:,i].flatten()
            Ts_i = Ts
            
            pickup_i = (w_i * Cu_i - Ct_i) / Ts_i * self.dt # Dit klopt niet! enkel geldig bij backward euler
            deficit_i = pickup_i - mass_i
            ix = (deficit_i > p['max_error']) \
                    & (w_i * Cu_i > 0.)

            pickup[:,:,i] = pickup_i.reshape(yCt_i.shape)
            Ct[:,:,i] = Ct_i.reshape(yCt_i.shape)
                    
            # quit the iteration if there is no deficit, otherwise
            # back-compute the maximum weight allowed to get zero
            # deficit for the current fraction and progress to
            # the next iteration step
            if not np.any(ix):
                logger.debug(format_log('Iteration converged',
                                        steps=n,
                                        fraction=i,
                                        **logprops))
                pickup_i = np.minimum(pickup_i, mass_i)
                break
            else:
                w_i[ix] = (mass_i[ix] * Ts_i / self.dt \
                            + Ct_i[ix]) / Cu_i[ix]
                w[:,:,i] = w_i.reshape(yCt_i.shape)

        # throw warning if the maximum number of iterations was
        # reached
        if np.any(ix):
            logger.warn(format_log('Iteration not converged',
                                    nrcells=np.sum(ix),
                                    fraction=i,
                                    **logprops))
        
        # check for unexpected negative values
        if Ct_i.min() < 0:
            logger.warn(format_log('Negative concentrations',
                                    nrcells=np.sum(Ct_i<0.),
                                    fraction=i,
                                    minvalue=Ct_i.min(),
                                    **logprops))
        if w_i.min() < 0:
            logger.warn(format_log('Negative weights',
                                    nrcells=np.sum(w_i<0),
                                    fraction=i,
                                    minvalue=w_i.min(),
                                    **logprops))
    # end loop over frations

    # check if there are any cells where the sum of all weights is
    # smaller than unity. these cells are supply-limited for all
    # fractions. Log these events.
    ix = 1. - np.sum(w, axis=2) > p['max_error']
    if np.any(ix):
        self._count('supplylim')
#            logger.warn(format_log('Ran out of sediment',
#                                   nrcells=np.sum(ix),
#                                   minweight=np.sum(w, axis=-1).min(),
#                                   **logprops))
    qs = Ct * s['us'] 
    qn = Ct * s['un']
                
    return dict(Ct=Ct,
                qs=qs,
                qn=qn,
                pickup=pickup,
                w=w,
                w_init=w_init,
                w_air=w_air,
                w_bed=w_bed)


def solve_pieter(self, alpha:float=.5, beta:float=1.) -> dict:
    '''Implements the explicit Euler forward, implicit Euler backward and semi-implicit Crank-Nicolson numerical schemes

    Determines weights of sediment fractions, sediment pickup and
    instantaneous sediment concentration. Returns a partial
    spatial grid dictionary that can be used to update the global
    spatial grid dictionary.

    Parameters
    ----------
    alpha : 
        Implicitness coefficient (0.0 for Euler forward, 1.0 for Euler backward or 0.5 for Crank-Nicolson, default=0.5)
    beta : float, optional
        Centralization coefficient (1.0 for upwind or 0.5 for centralized, default=1.0)

    Returns
    -------
        Partial spatial grid dictionary

    Examples
    --------
    >>> model.s.update(model.solve(alpha=1., beta=1.) # euler backward

    >>> model.s.update(model.solve(alpha=.5, beta=1.) # crank-nicolson

    See Also
    --------
    model.AeoLiS.euler_forward
    model.AeoLiS.euler_backward
    model.AeoLiS.crank_nicolson
    transport.compute_weights
    transport.renormalize_weights
    '''

    l = self.l
    s = self.s
    p = self.p

    Ct = s['Ct'].copy()
    qs = s['qs'].copy()
    qn = s['qn'].copy()
    pickup = s['pickup'].copy()
    
    Ts = p['T']
    
    # compute transport weights for all sediment fractions
    w_init, w_air, w_bed = aeolis.transport.compute_weights(s, p)

    if self.t == 0.:
        # use initial guess for first time step
        w = p['grain_dist'].reshape((1,1,-1))
        w = w.repeat(p['ny']+1, axis=0)
        w = w.repeat(p['nx']+1, axis=1)
        return dict(w=w)
    else:
        w = w_init.copy()

    # set model state properties that are added to warnings and errors
    logprops = dict(minwind=s['uw'].min(),
                    maxdrop=(l['uw']-s['uw']).max(),
                    time=self.t,
                    dt=self.dt)
    
    nf = p['nfractions']

    ufs = np.zeros((p['ny']+1,p['nx']+2))
    ufn = np.zeros((p['ny']+2,p['nx']+1))               
    
    for i in range(nf): #loop over fractions
    
        #define velocity fluxes
        ufs[:,1:-1] = 0.5*s['us'][:,:-1,i] + 0.5*s['us'][:,1:,i]
        ufn[1:-1,:] = 0.5*s['un'][:-1,:,i] + 0.5*s['un'][1:,:,i]
        
        #boundary values
        ufs[:,0]  = s['us'][:,0,i]
        ufs[:,-1] = s['us'][:,-1,i]
        
        if p['boundary_lateral'] == 'circular':
            ufn[0,:] = 0.5*s['un'][0,:,i] + 0.5*s['un'][-1,:,i]
            ufn[-1,:] = ufn[0,:]
        else:
            ufn[0,:]  = s['un'][0,:,i]
            ufn[-1,:] = s['un'][-1,:,i]    
    
        beta = abs(beta)
        if beta >= 1.:
            # define upwind direction
            ixfs = np.asarray(ufs >= 0., dtype=float)
            ixfn = np.asarray(ufn >= 0., dtype=float)
        else:
            # or centralizing weights
            ixfs = beta + np.zeros(ufs)
            ixfn = beta + np.zeros(ufn)

        # initialize matrix diagonals
        A0 = np.zeros(s['zb'].shape)
        Apx = np.zeros(s['zb'].shape)
        Ap1 = np.zeros(s['zb'].shape)
        Amx = np.zeros(s['zb'].shape)
        Am1 = np.zeros(s['zb'].shape)

        # populate matrix diagonals
        A0         += s['dsdn'] / self.dt                                        #time derivative
        A0         += s['dsdn'] / Ts                                     * alpha #source term
        A0[:,1:]   -= s['dn'][:,1:]  * ufs[:,1:-1] * (1. - ixfs[:,1:-1]) * alpha #lower x-face
        Am1[:,1:]  -= s['dn'][:,1:]  * ufs[:,1:-1] *       ixfs[:,1:-1]  * alpha #lower x-face
        A0[:,:-1]  += s['dn'][:,:-1] * ufs[:,1:-1] *       ixfs[:,1:-1]  * alpha #upper x-face
        Ap1[:,:-1] += s['dn'][:,:-1] * ufs[:,1:-1] * (1. - ixfs[:,1:-1]) * alpha #upper x-face
        A0[1:,:]   -= s['ds'][1:,:]  * ufn[1:-1,:] * (1. - ixfn[1:-1,:]) * alpha #lower y-face
        Amx[1:,:]  -= s['ds'][1:,:]  * ufn[1:-1,:] *       ixfn[1:-1,:]  * alpha #lower y-face
        A0[:-1,:]  += s['ds'][:-1,:] * ufn[1:-1,:] *       ixfn[1:-1,:]  * alpha #upper y-face
        Apx[:-1,:] += s['ds'][:-1,:] * ufn[1:-1,:] * (1. - ixfn[1:-1,:]) * alpha #upper y-face
    
        # add boundaries
        # offshore boundary (i=0)

        if p['boundary_offshore'] == 'flux':
            #nothing to be done
            pass
        elif p['boundary_offshore'] == 'constant':
            #constant sediment concentration (Ct) in the air
            A0[:,0] = 1.
            Apx[:,0] = 0.
            Amx[:,0] = 0.
            Ap1[:,0] = 0.
            Am1[:,0] = 0.
        elif p['boundary_offshore'] == 'gradient':
            #remove the flux at the inner face of the cell
            A0[:,0]  -= s['dn'][:,0] * ufs[:,1] *       ixfs[:,1]  * alpha #upper x-face
            Ap1[:,0] -= s['dn'][:,0] * ufs[:,1] * (1. - ixfs[:,1]) * alpha #upper x-face
        elif p['boundary_offshore'] == 'circular':
            raise NotImplementedError('Cross-shore cricular boundary condition not yet implemented')
        else:
            raise ValueError('Unknown offshore boundary condition [%s]' % self.p['boundary_offshore'])

        #onshore boundary (i=nx)

        if p['boundary_onshore'] == 'flux':
            #nothing to be done
            pass
        elif p['boundary_onshore'] == 'constant':
            #constant sediment concentration (hC) in the air
            A0[:,-1] = 1.
            Apx[:,-1] = 0.
            Amx[:,-1] = 0.
            Ap1[:,-1] = 0.
            Am1[:,-1] = 0.
        elif p['boundary_onshore'] == 'gradient':
            #remove the flux at the inner face of the cell
            A0[:,-1]   += s['dn'][:,-1]  * ufs[:,-2]   * (1. - ixfs[:,-2]) * alpha #lower x-face
            Am1[:,-1]  += s['dn'][:,-1]  * ufs[:,-2]   *       ixfs[:,-2]  * alpha #lower x-face
        elif p['boundary_onshore'] == 'circular':
            raise NotImplementedError('Cross-shore cricular boundary condition not yet implemented')
        else:
            raise ValueError('Unknown offshore boundary condition [%s]' % self.p['boundary_onshore'])
    
        #lateral boundaries (j=0; j=ny)    

        if p['boundary_lateral'] == 'flux':
            #nothing to be done
            pass
        elif p['boundary_lateral'] == 'constant':
            #constant sediment concentration (hC) in the air
            A0[0,:] = 1.
            Apx[0,:] = 0.
            Amx[0,:] = 0.
            Ap1[0,:] = 0.
            Am1[0,:] = 0.
            A0[-1,:] = 1.
            Apx[-1,:] = 0.
            Amx[-1,:] = 0.
            Ap1[-1,:] = 0.
            Am1[-1,:] = 0.
        elif p['boundary_lateral'] == 'gradient':
            #remove the flux at the inner face of the cell
            A0[0,:]   -= s['ds'][0,:] * ufn[1,:]   *       ixfn[1,:]   * alpha #upper y-face
            Apx[0,:]  -= s['ds'][0,:] * ufn[1,:]   * (1. - ixfn[1,:])  * alpha #upper y-face
            A0[-1,:]  += s['ds'][-1,:] * ufn[-2,:] * (1. - ixfn[-2,:]) * alpha #lower y-face
            Amx[-1,:] += s['ds'][-1,:] * ufn[-2,:] *       ixfn[-2,:]  * alpha #lower y-face
        elif p['boundary_lateral'] == 'circular':
            A0[0,:]   -= s['ds'][0,:]  * ufn[0,:]  * (1. - ixfn[0,:])  * alpha #lower y-face
            Amx[0,:]  -= s['ds'][0,:]  * ufn[0,:]  *       ixfn[0,:]   * alpha #lower y-face
            A0[-1,:]  += s['ds'][-1,:] * ufn[-1,:] *       ixfn[-1,:]  * alpha #upper y-face
            Apx[-1,:] += s['ds'][-1,:] * ufn[-1,:] * (1. - ixfn[-1,:]) * alpha #upper y-face
        else:
            raise ValueError('Unknown lateral boundary condition [%s]' % self.p['boundary_lateral'])
        
        # construct sparse matrix
        if p['ny'] > 0:
            j = p['nx']+1
            A = scipy.sparse.diags((Apx.flatten()[:j],
                                    Amx.flatten()[j:],
                                    Am1.flatten()[1:],
                                    A0.flatten(),
                                    Ap1.flatten()[:-1],
                                    Apx.flatten()[j:],
                                    Amx.flatten()[:j]),
                                    (-j*p['ny'],-j,-1,0,1,j,j*p['ny']), format='csr')
        else:
            A = scipy.sparse.diags((Am1.flatten()[1:],
                                    A0.flatten(),
                                    Ap1.flatten()[:-1]),
                                    (-1,0,1), format='csr')

        # solve transport for each fraction separately using latest
        # available weights
    
        # renormalize weights for all fractions equal or larger
        # than the current one such that the sum of all weights is
        # unity
        w = aeolis.transport.renormalize_weights(w, i)

        # iteratively find a solution of the linear system that
        # does not violate the availability of sediment in the bed
        for n in range(p['max_iter']):
            self._count('matrixsolve')
#                print("iteration nr = %d" % n)
            # define upwind face value
            # sediment concentration
            Ctxfs_i = np.zeros(ufs.shape)
            Ctxfn_i = np.zeros(ufn.shape)
            
            Ctxfs_i[:,1:-1] = ixfs[:,1:-1] * ( alpha * Ct[:,:-1,i] \
                                                + (1. - alpha ) * l['Ct'][:,:-1,i] ) \
                + (1. - ixfs[:,1:-1]) * ( alpha * Ct[:,1:,i] \
                                            + (1. - alpha ) * l['Ct'][:,1:,i] )
            Ctxfn_i[1:-1,:] = ixfn[1:-1,:] * (alpha * Ct[:-1,:,i] \
                                                + (1. - alpha ) * l['Ct'][:-1,:,i] ) \
                + (1. - ixfn[1:-1,:]) * ( alpha * Ct[1:,:,i] \
                                            + (1. - alpha ) * l['Ct'][1:,:,i] )
                
            if p['boundary_lateral'] == 'circular':
                Ctxfn_i[0,:] = ixfn[0,:] * (alpha * Ct[-1,:,i] \
                                            + (1. - alpha ) * l['Ct'][-1,:,i] ) \
                    + (1. - ixfn[0,:]) * ( alpha * Ct[0,:,i] \
                                            + (1. - alpha ) * l['Ct'][0,:,i] )
                Ctxfn_i[-1,:] = Ctxfn_i[0,:]                   
            
            # calculate pickup
            D_i = s['dsdn'] / Ts * ( alpha * Ct[:,:,i]  \
                                        + (1. - alpha ) * l['Ct'][:,:,i] )
            A_i = s['dsdn'] / Ts * s['mass'][:,:,0,i] + D_i # Availability
            U_i = s['dsdn'] / Ts * ( w[:,:,i] * alpha * s['Cu'][:,:,i] \
                                        + (1. - alpha ) * l['w'][:,:,i] * l['Cu'][:,:,i] )
            #deficit_i = E_i - A_i
            E_i= np.minimum(U_i, A_i)
            #pickup_i = E_i - D_i

            # create the right hand side of the linear system
            # sediment concentration
            yCt_i = np.zeros(s['zb'].shape)
            yCt_i         -= s['dsdn'] / self.dt * ( Ct[:,:,i] \
                                                    - l['Ct'][:,:,i] )      #time derivative
            yCt_i         += E_i - D_i                                      #source term
            yCt_i[:,1:]   += s['dn'][:,1:]  * ufs[:,1:-1] * Ctxfs_i[:,1:-1] #lower x-face
            yCt_i[:,:-1]  -= s['dn'][:,:-1] * ufs[:,1:-1] * Ctxfs_i[:,1:-1] #upper x-face
            yCt_i[1:,:]   += s['ds'][1:,:]  * ufn[1:-1,:] * Ctxfn_i[1:-1,:] #lower y-face
            yCt_i[:-1,:]  -= s['ds'][:-1,:] * ufn[1:-1,:] * Ctxfn_i[1:-1,:] #upper y-face
                
            # boundary conditions
            # offshore boundary (i=0)

            if p['boundary_offshore'] == 'flux':
                yCt_i[:,0]  += s['dn'][:,0] * ufs[:,0] * s['Cu0'][:,0,i] * p['offshore_flux'] 

            elif p['boundary_offshore'] == 'constant':
                #constant sediment concentration (Ct) in the air (for now = 0)
                yCt_i[:,0]  = 0.

            elif p['boundary_offshore'] == 'gradient':
                #remove the flux at the inner face of the cell
                yCt_i[:,0]  += s['dn'][:,1] * ufs[:,1] * Ctxfs_i[:,1] #upper x-face

            elif p['boundary_offshore'] == 'circular':
                raise NotImplementedError('Cross-shore cricular boundary condition not yet implemented')
            else:
                raise ValueError('Unknown offshore boundary condition [%s]' % self.p['boundary_offshore'])
                
            # onshore boundary (i=nx)

            if p['boundary_onshore'] == 'flux':
                yCt_i[:,-1]  += s['dn'][:,-1]  * ufs[:,-1] * s['Cu0'][:,-1,i] * p['onshore_flux'] 

            elif p['boundary_onshore'] == 'constant':
                #constant sediment concentration (Ct) in the air (for now = 0)
                yCt_i[:,-1]  = 0.

            elif p['boundary_onshore'] == 'gradient':
                #remove the flux at the inner face of the cell
                yCt_i[:,-1]  -= s['dn'][:,-2] * ufs[:,-2] * Ctxfs_i[:,-2] #lower x-face

            elif p['boundary_onshore'] == 'circular':
                raise NotImplementedError('Cross-shore cricular boundary condition not yet implemented')
            else:
                raise ValueError('Unknown onshore boundary condition [%s]' % self.p['boundary_onshore'])
                
            #lateral boundaries (j=0; j=ny)    

            if p['boundary_lateral'] == 'flux':
                
                yCt_i[0,:]  += s['ds'][0,:] * ufn[0,:]  * s['Cu0'][0,:,i] * p['lateral_flux'] #lower y-face
                yCt_i[-1,:] -= s['ds'][-1,:] * ufn[-1,:] * s['Cu0'][-1,:,i] * p['lateral_flux'] #upper y-face
                
            elif p['boundary_lateral'] == 'constant':
                #constant sediment concentration (hC) in the air
                yCt_i[0,:]  = 0.
                yCt_i[-1,:] = 0.
            elif p['boundary_lateral'] == 'gradient':
                #remove the flux at the inner face of the cell
                yCt_i[-1,:] -= s['ds'][-2,:] * ufn[-2,:] * Ctxfn_i[-2,:] #lower y-face
                yCt_i[0,:]  += s['ds'][1,:]  * ufn[1,:]  * Ctxfn_i[1,:]  #upper y-face
            elif p['boundary_lateral'] == 'circular':
                yCt_i[0,:]  += s['ds'][0,:]  * ufn[0,:]  * Ctxfn_i[0,:]  #lower y-face
                yCt_i[-1,:] -= s['ds'][-1,:] * ufn[-1,:] * Ctxfn_i[-1,:] #upper y-face
            else:
                raise ValueError('Unknown lateral boundary condition [%s]' % self.p['boundary_lateral'])
            
            # print("ugs = %.*g" % (3,s['ugs'][10,10]))
            # print("ugn = %.*g" % (3,s['ugn'][10,10]))
            # print("%.*g" % (3,np.amax(np.absolute(y_i))))
            
            # solve system with current weights
            Ct_i = Ct[:,:,i].flatten()
            Ct_i += scipy.sparse.linalg.spsolve(A, yCt_i.flatten())
            Ct_i = prevent_tiny_negatives(Ct_i, p['max_error'])
            
            # check for negative values
            if Ct_i.min() < 0.:
                ix = Ct_i < 0.
                
#                    logger.warn(format_log('Removing negative concentrations',
#                                           nrcells=np.sum(ix),
#                                           fraction=i,
#                                           iteration=n,
#                                           minvalue=Ct_i.min(),
#                                           **logprops))
                
                if 0: #Ct_i[~ix].sum()>0.:
                    # compensate the negative concentrations by distributing them over the positives.
                    # I guess the idea is to conserve mass but it is not sure if this is needed, 
                    # mass continuity in the system is guaranteed by exchange with bed.
                    Ct_i[~ix] *= 1. + Ct_i[ix].sum() / Ct_i[~ix].sum()
                Ct_i[ix] = 0.

            # determine pickup and deficit for current fraction
            Cu_i = s['Cu'][:,:,i].flatten()
            mass_i = s['mass'][:,:,0,i].flatten()
            w_i = w[:,:,i].flatten()
            Ts_i = Ts
            
            pickup_i = (w_i * Cu_i - Ct_i) / Ts_i * self.dt # Dit klopt niet! enkel geldig bij backward euler
            deficit_i = pickup_i - mass_i
            ix = (deficit_i > p['max_error']) \
                    & (w_i * Cu_i > 0.)

            pickup[:,:,i] = pickup_i.reshape(yCt_i.shape)
            Ct[:,:,i] = Ct_i.reshape(yCt_i.shape)
                    
            # quit the iteration if there is no deficit, otherwise
            # back-compute the maximum weight allowed to get zero
            # deficit for the current fraction and progress to
            # the next iteration step
            if not np.any(ix):
                logger.debug(format_log('Iteration converged',
                                        steps=n,
                                        fraction=i,
                                        **logprops))
                pickup_i = np.minimum(pickup_i, mass_i)
                break
            else:
                w_i[ix] = (mass_i[ix] * Ts_i / self.dt \
                            + Ct_i[ix]) / Cu_i[ix]
                w[:,:,i] = w_i.reshape(yCt_i.shape)

        # throw warning if the maximum number of iterations was
        # reached
        
        if np.any(ix):
            logger.warn(format_log('Iteration not converged',
                                    nrcells=np.sum(ix),
                                    fraction=i,
                                    **logprops))
        
        if 0: #let's disable these warnings
            # check for unexpected negative values
            if Ct_i.min() < 0:
                logger.warn(format_log('Negative concentrations',
                                    nrcells=np.sum(Ct_i<0.),
                                    fraction=i,
                                    minvalue=Ct_i.min(),
                                    **logprops))
            if w_i.min() < 0:
                logger.warn(format_log('Negative weights',
                                    nrcells=np.sum(w_i<0),
                                    fraction=i,
                                    minvalue=w_i.min(),
                                    **logprops))
    # end loop over frations

    # check if there are any cells where the sum of all weights is
    # smaller than unity. these cells are supply-limited for all
    # fractions. Log these events.
    ix = 1. - np.sum(w, axis=2) > p['max_error']
    if np.any(ix):
        self._count('supplylim')
#            logger.warn(format_log('Ran out of sediment',
#                                   nrcells=np.sum(ix),
#                                   minweight=np.sum(w, axis=-1).min(),
#                                   **logprops))

    qs = Ct * s['us'] 
    qn = Ct * s['un']
    qs = Ct * s['us'] 
    qn = Ct * s['un'] 
    q = np.hypot(qs, qn)
    
                
    return dict(Ct=Ct,
                qs=qs,
                qn=qn,
                pickup=pickup,
                w=w,
                w_init=w_init,
                w_air=w_air,
                w_bed=w_bed,
                q=q)   


# Note: @njit(cache=True) is intentionally not used here.
# This function acts as an orchestrator, delegating work to Numba-compiled helper functions.
# Decorating the orchestrator itself with njit provides no performance benefit,
# since most of the computation is already handled by optimized Numba functions.
def sweep(Ct, Cu_bed, Cu_air, zeta, mass, dt, Ts, ds, dn, us, un, w, 
          onshore_bc, offshore_bc, lateral_bc, uws, uwn):

    Cu = Cu_bed.copy()

    pickup = np.zeros(Cu.shape)
    i=0
    k=0

    nf = np.shape(Ct)[2]

    # Are the lateral boundary conditions circular?
    # Why are we doing this???
    if lateral_bc == 'circular':
        Ct[0,:,:] = 0                
        Ct[-1,:,:] = 0
    if onshore_bc == 'circular':
        Ct[:,-1,:] = 0                
    if offshore_bc == 'circular':            
        Ct[:,0,:] = 0
    
    ufs = np.zeros((np.shape(us)[0], np.shape(us)[1]+1, np.shape(us)[2]))
    ufn = np.zeros((np.shape(un)[0]+1, np.shape(un)[1], np.shape(un)[2]))
    
    # define velocity at cell faces
    ufs[:,1:-1, :] = 0.5*us[:,:-1, :] + 0.5*us[:,1:, :]
    ufn[1:-1,:, :] = 0.5*un[:-1,:, :] + 0.5*un[1:,:, :]

    # print(ufs[5,:,0])

    # set empty boundary values, extending the velocities at the boundaries
    ufs[:,0, :]  = ufs[:,1, :]
    ufs[:,-1, :] = ufs[:,-2, :]
   
    ufn[0,:, :]  = ufn[1,:, :]
    ufn[-1,:, :] = ufn[-2,:, :]
    
    # Lets take the average of the top and bottom and left/right boundary cells
    # apply the average to the boundary cells
    # this ensures that the inflow at one side is equal to the outflow at the other side

    ufs[:,0,:]  = (ufs[:,0,:]+ufs[:,-1,:])/2
    ufs[:,-1,:] = ufs[:,0,:]     
    ufs[0,:,:]  = (ufs[0,:,:]+ufs[-1,:,:])/2
    ufs[-1,:,:] = ufs[0,:,:]     
    
    ufn[:,0,:]  = (ufn[:,0,:]+ufn[:,-1,:])/2
    ufn[:,-1,:] = ufn[:,0,:]     
    ufn[0,:,:]  = (ufn[0,:,:]+ufn[-1,:,:])/2
    ufn[-1,:,:] = ufn[0,:,:] 

    # also correct for the potential gradients at the boundary cells in the equilibrium concentrations
    Cu[:,0,:]  = Cu[:,1,:]
    Cu[:,-1,:] = Cu[:,-2,:]
    Cu[0,:,:]  = Cu[1,:,:]
    Cu[-1,:,:] = Cu[-2,:,:]

    Ct_last = Ct.copy()

    while k==0 or np.any(np.abs(Ct[:,:,i]-Ct_last[:,:,i])>1e-8):
        Ct_last = Ct.copy()

        # # lateral boundaries circular
        # if lateral_bc == 'circular':
        #     Ct[0,:,:],Ct[-1,:,:] = Ct[-1,:,:].copy(),Ct[0,:,:].copy()
        # if onshore_bc == 'circular':
        #     Ct[:,-1,:] = Ct[:,0,:].copy()
        # if offshore_bc == 'circular':
        #     Ct[:,0,:] = Ct[:,-1,:].copy()

        # only for incoming
        

        # circular boundaries (CHECK THIS!)
        if lateral_bc == 'circular' and uwn >= 0:
            Ct[0,:,:] = Ct[-1,:,:].copy()
        if lateral_bc == 'circular' and uwn < 0:
            Ct[-1,:,:] = Ct[0,:,:].copy()
        if onshore_bc == 'circular' and uws < 0:
            Ct[:,-1,:] = Ct[:,0,:].copy()
        if offshore_bc == 'circular' and uws >= 0:
            Ct[:,0,:] = Ct[:,-1,:].copy()

        # constant boundaries (maybe use Cu here to help?)
        if lateral_bc == 'constant' and uwn >= 0:
            Ct[0,:,:] = Ct[1,:,:].copy()
        if lateral_bc == 'constant' and uwn < 0:
            Ct[-1,:,:] = Ct[-2,:,:].copy()
        if onshore_bc == 'constant' and uws < 0:
            Ct[:,-1,:] = Ct[:,-2,:].copy()
        if offshore_bc == 'constant' and uws >= 0:
            Ct[:,0,:] = Ct[:,1,:].copy()

        # flux boundaries (zero influx)
        if lateral_bc == 'flux' and uwn >= 0:
            Ct[0,:,:] = 0.
        if lateral_bc == 'flux' and uwn < 0:
            Ct[-1,:,:] = 0.
        if onshore_bc == 'flux' and uws < 0:
            Ct[:,-1,:] = 0.
        if offshore_bc == 'flux' and uws >= 0:
            Ct[:,0,:] = 0.

        # initialize visited matrix and quadrant matrix
        visited = np.zeros(Ct.shape[:2], dtype=np.bool_)
        quad = np.zeros(Ct.shape[:2], dtype=np.uint8)

        # solve quadrants
        _solve_quadrant1(Ct, Cu_air, Cu_bed, zeta, mass, pickup,
                          dt, Ts, ds, dn, ufs, ufn, w, visited, quad, nf)

        _solve_quadrant2(Ct, Cu_air, Cu_bed, zeta, mass, pickup,
                          dt, Ts, ds, dn, ufs, ufn, w, visited, quad, nf)

        _solve_quadrant3(Ct, Cu_air, Cu_bed, zeta, mass, pickup,
                          dt, Ts, ds, dn, ufs, ufn, w, visited, quad, nf)

        _solve_quadrant4(Ct, Cu_air, Cu_bed, zeta, mass, pickup,
                          dt, Ts, ds, dn, ufs, ufn, w, visited, quad, nf)

        _solve_generic_stencil(Ct, Cu_air, Cu_bed, zeta, mass, pickup,
                                dt, Ts, ds, dn, ufs, ufn, w, visited, quad, nf)

        
        # check the boundaries of the pickup matrix for unvisited cells
        # print(np.shape(visited[0,:]==False))
        # WHAT IS THIS????
        # pickup[0,:,0] = pickup[1,:,0].copy() 
        # pickup[-1,:,0] = pickup[-2,:,0].copy() 

        # Try trick: Where Ct > 3 * Cu, set Ct = to Cu
        Ct = prevent_excessive_concentrations(Ct, Cu_air)

        omega = 0.99
        Ct[:] = Ct_last + omega*(Ct - Ct_last)

        k+=1

        if k > 1000:
            logger.warning(f'Limit of k, max. difference Ct: {np.max(np.abs(Ct[:,:,i]-Ct_last[:,:,i]))} \n      uws: {uws}, uwn: {uwn}')
            
            # fig,ax = plt.subplots(5,1, figsize=(12,6))
            # im0 = ax[0].imshow(Ct[:,:,0], origin='lower', cmap='viridis', vmin=0)#, vmax=0.05)
            # ax[0].set_title('Sediment Concentration (Ct)')
            # fig.colorbar(im0, ax=ax[0], label='Ct')
            # im1 = ax[1].imshow(zeta, origin='lower', cmap='viridis', vmin=0, vmax=1)
            # ax[1].set_title('Zeta')
            # fig.colorbar(im1, ax=ax[1], label='zeta')
            # im2 = ax[2].imshow(np.abs(Ct[:,:,0]-Ct_last[:,:,0]), origin='lower', cmap='viridis')
            # ax[2].set_title('Difference in Ct')
            # fig.colorbar(im2, ax=ax[2], label='|Ct - Ct_last|')
            # im2 = ax[3].imshow(us[:,:,0], origin='lower', cmap='viridis')
            # ax[3].quiver(us[:,:,0], un[:,:,0], scale=100, color='white')
            # ax[3].set_title('us')   
            # fig.colorbar(im2, ax=ax[3], label='us')
            # im3 = ax[4].imshow(un[:,:,0], origin='lower', cmap='viridis')
            # ax[4].set_title('un')
            # fig.colorbar(im3, ax=ax[4], label='un')
            # plt.tight_layout()
            # plt.show()

            break

    # print(f"Number of sweeps: {k}")



    # Store Cu for output
    
    # Prevent division by zero
    Cu_air_safe = Cu_air.copy()
    Cu_air_safe[Cu_air_safe == 0] = 1e-10
    for f in range(nf):
        w_zeta_air = (1 - zeta) * Ct[:, :, f] / Cu_air_safe[:, :, f]
        w_zeta_bed = 1 - w_zeta_air
        Cu[:, :, f] = w_zeta_air * Cu_air[:, :, f] + w_zeta_bed * Cu_bed[:, :, f]
    
    return Cu, Ct, pickup


@njit(cache=True)
def _solve_quadrant1(Ct, Cu_air, Cu_bed, zeta, mass, pickup,
                     dt, Ts, ds, dn, ufs, ufn, w, visited, quad, nf):

    n0 = 1 # 1
    s0 = 1 # 1

    for n in range(n0, Ct.shape[0]):
        for s in range(s0, Ct.shape[1]):

            if ((not visited[n, s]) and
                (ufn[n,s,0] >= 0) and (ufs[n,s,0] >= 0) and
                (ufn[n+1,s,0] >= 0) and (ufs[n,s+1,0] >= 0)):

                A = ds[n,s] * dn[n,s]

                for f in range(nf):

                    # compute a,b
                    if Cu_air[n,s,f] > 0 and Ct[n,s,f] > 0:
                        a = (1 - zeta[n,s]) * (Cu_air[n,s,f] - Cu_bed[n,s,f]) / Cu_air[n,s,f]
                        b = Cu_bed[n,s,f]
                    else:
                        a = 0.0
                        b = Cu_bed[n,s,f]

                    # inflow term
                    N = (
                        Ct[n-1,s,f] * ufn[n,s,f] * ds[n,s] +
                        Ct[n, s-1,f] * ufs[n,s,f] * dn[n,s]
                    )

                    # denominator term
                    D = (
                        ufn[n+1,s,f] * ds[n,s] +
                        ufs[n, s+1,f] * dn[n,s] +
                        A / Ts
                    )

                    Ct_new = (N + w[n,s,f] * b * A/Ts) / (D - w[n,s,f] * a * A/Ts)
                    Ct[n,s,f] = Ct_new

                    Cu_local = b + a * Ct_new
                    p = (w[n,s,f] * Cu_local - Ct_new) * dt/Ts

                    if p > mass[n,s,0,f]:
                        p = mass[n,s,0,f]
                        N_lim = N + p * A/dt
                        D_lim = ufn[n+1,s,f] * ds[n,s] + ufs[n,s+1,f] * dn[n,s]
                        Ct[n,s,f] = N_lim / D_lim

                    pickup[n,s,f] = p

                visited[n,s] = True
                quad[n,s] = 1


@njit(cache=True)
def _solve_quadrant2(Ct, Cu_air, Cu_bed, zeta, mass, pickup,
                     dt, Ts, ds, dn, ufs, ufn, w, visited, quad, nf):

    n0 = 1 # 1
    s1 = Ct.shape[1]-2 # Ct.shape[1]-2

    for n in range(n0, Ct.shape[0]):
        for s in range(s1, -1, -1):

            if ((not visited[n,s]) and
                (ufn[n,s,0] >= 0) and (ufs[n,s,0] <= 0) and
                (ufn[n+1,s,0] >= 0) and (ufs[n,s+1,0] <= 0)):

                A = ds[n,s] * dn[n,s]

                for f in range(nf):

                    if Cu_air[n,s,f] > 0 and Ct[n,s,f] > 0:
                        a = (1 - zeta[n,s]) * (Cu_air[n,s,f] - Cu_bed[n,s,f]) / Cu_air[n,s,f]
                        b = Cu_bed[n,s,f]
                    else:
                        a = 0.0
                        b = Cu_bed[n,s,f]

                    N = (
                        Ct[n-1,s,f] * ufn[n,s,f] * ds[n,s] +
                        -Ct[n,s+1,f] * ufs[n,s+1,f] * dn[n,s]
                    )

                    D = (
                        ufn[n+1,s,f] * ds[n,s] +
                        -ufs[n,s,f] * dn[n,s] +
                        A/Ts
                    )

                    Ct_new = (N + w[n,s,f] * b * A/Ts) / (D - w[n,s,f] * a * A/Ts)
                    Ct[n,s,f] = Ct_new

                    Cu_local = b + a * Ct_new
                    p = (w[n,s,f] * Cu_local - Ct_new) * dt/Ts

                    if p > mass[n,s,0,f]:
                        p = mass[n,s,0,f]
                        N_lim = N + p*A/dt
                        D_lim = ufn[n+1,s,f]*ds[n,s] + -ufs[n,s,f]*dn[n,s]
                        Ct[n,s,f] = N_lim / D_lim

                    pickup[n,s,f] = p

                visited[n,s] = True
                quad[n,s] = 2


@njit(cache=True)
def _solve_quadrant3(Ct, Cu_air, Cu_bed, zeta, mass, pickup,
                     dt, Ts, ds, dn, ufs, ufn, w, visited, quad, nf):

    n1 = Ct.shape[0]-2 # Ct.shape[0]-2
    s1 = Ct.shape[1]-2 # Ct.shape[1]-2

    for n in range(n1, -1, -1):
        for s in range(s1, -1, -1):

            if ((not visited[n,s]) and
                (ufn[n,s,0] <= 0) and (ufs[n,s,0] <= 0) and
                (ufn[n+1,s,0] <= 0) and (ufs[n,s+1,0] <= 0)):

                A = ds[n,s] * dn[n,s]

                for f in range(nf):

                    if Cu_air[n,s,f] > 0 and Ct[n,s,f] > 0:
                        a = (1 - zeta[n,s]) * (Cu_air[n,s,f] - Cu_bed[n,s,f]) / Cu_air[n,s,f]
                        b = Cu_bed[n,s,f]
                    else:
                        a = 0.0
                        b = Cu_bed[n,s,f]

                    N = (
                        -Ct[n+1,s,f] * ufn[n+1,s,f] * dn[n,s] +
                        -Ct[n,s+1,f] * ufs[n,s+1,f] * dn[n,s]
                    )

                    D = (
                        -ufn[n,s,f] * dn[n,s] +
                        -ufs[n,s,f] * dn[n,s] +
                        A/Ts
                    )

                    Ct_new = (N + w[n,s,f]*b*A/Ts) / (D - w[n,s,f]*a*A/Ts)
                    Ct[n,s,f] = Ct_new

                    Cu_local = b + a * Ct_new
                    p = (w[n,s,f]*Cu_local - Ct_new)*dt/Ts

                    if p > mass[n,s,0,f]:
                        p = mass[n,s,0,f]
                        N_lim = N + p*A/dt
                        D_lim = -ufn[n,s,f]*dn[n,s] + -ufs[n,s,f]*dn[n,s]
                        Ct[n,s,f] = N_lim / D_lim

                    pickup[n,s,f] = p

                visited[n,s] = True
                quad[n,s] = 3


@njit(cache=True)
def _solve_quadrant4(Ct, Cu_air, Cu_bed, zeta, mass, pickup,
                     dt, Ts, ds, dn, ufs, ufn, w, visited, quad, nf):

    n1 = Ct.shape[0]-2 # Ct.shape[0]-2
    s0 = 1 # 1

    for n in range(n1, -1, -1):
        for s in range(s0, Ct.shape[1]):

            if ((not visited[n,s]) and
                (ufn[n,s,0] <= 0) and (ufs[n,s,0] >= 0) and
                (ufn[n+1,s,0] <= 0) and (ufs[n,s+1,0] >= 0)):

                A = ds[n,s] * dn[n,s]

                for f in range(nf):

                    if Cu_air[n,s,f] > 0 and Ct[n,s,f] > 0:
                        a = (1 - zeta[n,s])*(Cu_air[n,s,f] - Cu_bed[n,s,f]) / Cu_air[n,s,f]
                        b = Cu_bed[n,s,f]
                    else:
                        a = 0.0
                        b = Cu_bed[n,s,f]

                    N = (
                        Ct[n,s-1,f] * ufs[n,s,f] * dn[n,s] +
                        -Ct[n+1,s,f] * ufn[n+1,s,f] * dn[n,s]
                    )

                    D = (
                        ufs[n,s+1,f] * dn[n,s] +
                        -ufn[n,s,f] * dn[n,s] +
                        A/Ts
                    )

                    Ct_new = (N + w[n,s,f]*b*A/Ts) / (D - w[n,s,f]*a*A/Ts)
                    Ct[n,s,f] = Ct_new

                    Cu_local = b + a * Ct_new
                    p = (w[n,s,f]*Cu_local - Ct_new)*dt/Ts

                    if p > mass[n,s,0,f]:
                        p = mass[n,s,0,f]
                        N_lim = N + p*A/dt
                        D_lim = ufs[n,s+1,f]*dn[n,s] + -ufn[n,s,f]*dn[n,s]
                        Ct[n,s,f] = N_lim / D_lim

                    pickup[n,s,f] = p

                visited[n,s] = True
                quad[n,s] = 4


@njit(cache=True)
def _solve_generic_stencil(Ct, Cu_air, Cu_bed, zeta, mass, pickup,
                           dt, Ts, ds, dn, ufs, ufn, w, visited, quad, nf):

    Nx, Ny = Ct.shape[0], Ct.shape[1]

    for n in range(1, Nx-1):
        for s in range(1, Ny-1):

            if not visited[n,s]:

                A = ds[n,s] * dn[n,s]

                for f in range(nf):

                    # Cu = b + a * Ct
                    if Cu_air[n,s,f] > 0 and Ct[n,s,f] > 0:
                        a = (1 - zeta[n,s])*(Cu_air[n,s,f] - Cu_bed[n,s,f]) / Cu_air[n,s,f]
                        b = Cu_bed[n,s,f]
                    else:
                        a = 0.0
                        b = Cu_bed[n,s,f]

                    # start with source term
                    N = w[n,s,f] * b * A/Ts
                    D = A/Ts

                    # inflow contributions
                    if ufn[n,s,0] > 0:
                        N += Ct[n-1,s,f] * ufn[n,s,f] * ds[n,s]
                    else:
                        D += -ufn[n,s,f] * dn[n,s]

                    if ufs[n,s,0] > 0:
                        N += Ct[n,s-1,f] * ufs[n,s,f] * dn[n,s]
                    else:
                        D += -ufs[n,s,f] * dn[n,s]

                    # outflow contributions
                    if ufn[n+1,s,0] > 0:
                        D += ufn[n+1,s,f] * ds[n,s]
                    else:
                        N += -Ct[n+1,s,f] * ufn[n+1,s,f] * dn[n,s]

                    if ufs[n,s+1,0] > 0:
                        D += ufs[n,s+1,f] * dn[n,s]
                    else:
                        N += -Ct[n,s+1,f] * ufs[n,s+1,f] * dn[n,s]

                    # ---- DENOMINATOR PROTECTION ----
                    wa = w[n,s,f] * a
                    if wa > 0.999:
                        wa = 0.999

                    den = D - wa * A / Ts

                    # In extremely pathological cases D can be tiny; just in case:
                    if den == 0.0:
                        den = 1e-12

                    Ct_new = N / den
                    Ct[n,s,f] = Ct_new

                    Cu_local = b + a * Ct_new
                    p = (w[n,s,f]*Cu_local - Ct_new)*dt/Ts

                    if p > mass[n,s,0,f]:
                        p = mass[n,s,0,f]
                        # recompute limited:
                        N_lim = N + p*A/dt
                        # approximate denominator (no source term)
                        den_lim = D - A/Ts
                        if abs(den_lim) > 1e-12:
                            Ct[n,s,f] = N_lim / den_lim
                        # else: leave Ct[n,s,f] as is

                    pickup[n,s,f] = p

                visited[n,s] = True
                quad[n,s] = 5


@njit(cache=True)
def prevent_excessive_concentrations(Ct, Cu):
    Nx, Ny, Nf = Ct.shape
    for n in range(Nx):
        for s in range(Ny):
            for f in range(Nf):
                if Cu[n,s,f] > 0:
                    if Ct[n,s,f] > 10 * Cu[n,s,f]:
                        Ct[n,s,f] = 0.5 * Cu[n,s,f]
                        # Ct[n,s,f] -= (Ct[n,s,f] - Cu[n,s,f]) * 0.9
                        # Ct[n,s,f] *= 0.5
                        # # set to average of neighbors
                        # total = 0.0
                        # count = 0
                        # # check neighbors
                        # for dn in [-1, 0, 1]:
                        #     for ds in [-1, 0, 1]:
                        #         if (dn == 0 and ds == 0):
                        #             continue
                        #         nn = n + dn
                        #         ss = s + ds
                        #         if (0 <= nn < Nx) and (0 <= ss < Ny):
                        #             total += Ct[nn, ss, f]
                        #             count += 1
                        # if count > 0:
                        #     Ct[n,s,f] = total / count
    return Ct