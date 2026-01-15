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
from scipy import ndimage, misc
from numba import njit

# import AeoLiS modules
import aeolis.transport
from aeolis.utils import prevent_tiny_negatives, format_log, rotate

# initialize logger
logger = logging.getLogger(__name__)


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

    # Store previous states (still necessary?)
    l = self.l
    s = self.s
    p = self.p

    # Get variables
    Ct = s['Ct'].copy()
    pickup = s['pickup'].copy()
    Ts = p['T']
    nf = p['nfractions']

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
        
    # initiate emmpty solution matrix, this will effectively kill time dependence and create steady state.
    Ct = np.zeros(Ct.shape)
            
    # iterate to steady state
    Cu, Ct, pickup, pickup0, w, dCt, iters = sweep(Ct, s['CuBed'].copy(), s['CuAir'].copy(), 
                        s['zeta'].copy(), s['mass'].copy(), 
                        self.dt, p['T'], 
                        s['ds'], s['dn'], s['us'], s['un'], w,
                        p['boundary_offshore'], p['boundary_onshore'], p['boundary_lateral'],
                        p['offshore_flux'], p['onshore_flux'], p['lateral_flux'],
                        s['uws'][0,0], s['uwn'][0,0], p['max_iter'], p['max_error'], p['bi'])

    # compute pickup
    qs = Ct * s['us'] 
    qn = Ct * s['un'] 
    q = np.hypot(qs, qn)

    return dict(Cu=Cu,
                iters=iters,
                dCt=dCt,
                Ct=Ct,
                qs=qs,
                qn=qn,
                pickup=pickup,
                pickup0=pickup0,
                w=w,
                w_init=w_init,
                w_air=w_air,
                w_bed=w_bed,
                q=q)


def solve_EB(self, alpha:float=.5, beta:float=1.) -> dict:
    ''' OLD PIETER SOLVER! 
    Implements the explicit Euler forward, implicit Euler backward and semi-implicit Crank-Nicolson numerical schemes

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
          offshore_bc, onshore_bc, lateral_bc,
          offshore_flux, onshore_flux, lateral_flux,
          uws, uwn, max_iter, max_error, bi):

    # --- adding ghost cells -------------------------------------------------
    (Ct_g, Cu_air_g, Cu_bed_g, zeta_g, mass_g,
    ds_g, dn_g, us_g, un_g, w_g) = [_ghostify(a, offshore_bc, onshore_bc, lateral_bc)
        for a in (Ct, Cu_air, Cu_bed, zeta, mass, ds, dn, us, un, w)]

    # --- initialize Cu  and pickup on ghosted grid --------------------------
    Cu_g = Cu_bed_g.copy()
    pickup_g  = np.zeros_like(Cu_g)
    pickup0_g = np.zeros_like(Cu_g)

    nf = np.shape(Ct_g)[2]
    k=0

    # --- define face velocities ---------------------------------------------
    ufs_g = np.zeros((np.shape(us_g)[0], np.shape(us_g)[1]+1, np.shape(us_g)[2]))
    ufn_g = np.zeros((np.shape(un_g)[0]+1, np.shape(un_g)[1], np.shape(un_g)[2]))
    
    # --- define velocity at cell faces --------------------------------------
    ufs_g[:,1:-1, :] = 0.5*us_g[:,:-1, :] + 0.5*us_g[:,1:, :]
    ufn_g[1:-1,:, :] = 0.5*un_g[:-1,:, :] + 0.5*un_g[1:,:, :]

    # --- apply boundary conditions to face velocities -----------------------
    
    # s-direction (offshore/onshore)
    if offshore_bc == 'circular' and onshore_bc == 'circular':
        u_circ_s = 0.5*us_g[:, -1, :] + 0.5*us_g[:, 0, :]

        # apply to physical influx/outflux faces
        ufs_g[:, 1,  :] = u_circ_s
        ufs_g[:, -2, :] = u_circ_s
        # update ghost faces for consistency
        ufs_g[:, 0,  :] = u_circ_s
        ufs_g[:, -1, :] = u_circ_s

    else:
        ufs_g[:, 0,  :] = ufs_g[:, 1, :]
        ufs_g[:, -1, :] = ufs_g[:, -2, :]

    # n-direction (lateral)
    if lateral_bc == 'circular':
        u_circ_n = 0.5*un_g[-1, :, :] + 0.5*un_g[0, :, :]

        # apply to physical influx/outflux faces
        ufn_g[1, :,  :] = u_circ_n
        ufn_g[-2, :, :] = u_circ_n
        # update ghost faces for consistency
        ufn_g[0, :,  :] = u_circ_n
        ufn_g[-1, :, :] = u_circ_n

    else:
        ufn_g[0, :,  :] = ufn_g[1, :, :]
        ufn_g[-1, :, :] = ufn_g[-2, :, :]

    # --- set Ct for the first iteration (reduces number of iterations) ------
    for i in range(nf):
        Ct_g[:,:,i] = Cu_air_g[:,:,i] * zeta_g + Cu_bed_g[:,:,i] * (1. - zeta_g)

    # --- start iteration to steady state ------------------------------------
    iters = np.zeros_like(Cu_g, dtype=np.int32)
    Ct_last = Ct_g.copy()
    while k==0 or np.any(np.abs(Ct_g-Ct_last) > max_error):
        Ct_last = Ct_g.copy()

        # --- update Cu ------------------------------------------------------
        Cu_g = update_Cu(Ct_g, Cu_air_g, Cu_bed_g, zeta_g)

        # --- apply boundary conditions --------------------------------------
        Ct_g, Cu_g = apply_boundary(Ct_g, Cu_g, uws, uwn,
                                offshore_bc, onshore_bc, lateral_bc,
                                offshore_flux, onshore_flux, lateral_flux)

        # --- enforce physical limits (Prevent negative concentrations) ------
        for i_nf in range(nf):
            Ct_g[:,:,i_nf] = np.clip(Ct_g[:,:,i_nf], 0.0, np.max(Cu_air_g[:,:,i_nf])) # pragmatic upper limit to prevent blow-up
        Cu_g = np.maximum(Cu_g, 0.0)

        # --- update weights (w) ---------------------------------------------
        if nf > 1:
            w_g[:] = update_weights(Ct_g, Cu_g, mass_g, bi)

        # --- initialize visited matrix and quadrant matrix ------------------
        visited = np.zeros(Ct_g.shape[:2], dtype=np.bool_)
        quad = np.zeros(Ct_g.shape[:2], dtype=np.uint8)

        # --- perform sweeps in all four quadrants and generic stencil -------
        solvers = [_solve_quadrant1, _solve_quadrant2,
                   _solve_quadrant3, _solve_quadrant4,
                   _solve_generic_stencil]
        for solver in solvers:
            solver(Ct_g, Cu_air_g, Cu_bed_g, zeta_g, mass_g, pickup_g, pickup0_g, w_g,
                   dt, Ts, ds_g, dn_g, ufs_g, ufn_g, visited, quad, nf)
            
        k+=1

        # --- add iteration count to all cells that have not yet converged ---
        ix_notconverged = np.abs(Ct_g - Ct_last) > max_error
        iters[ix_notconverged] = k

        # --- safety break to prevent infinite loops -------------------------
        if k > max_iter:
            max_diff = np.max(np.abs(Ct_g - Ct_last))
            ix_max_diff = np.unravel_index(np.argmax(np.abs(Ct_g - Ct_last)), Ct_g.shape)
            rel_total_diff = 100 * np.sum(np.abs(Ct_g - Ct_last)) / np.sum(np.abs(Ct_last))
            rel_cell_diff = 100 * max_diff / np.abs(Ct_last[ix_max_diff])
            logger.warning(f'Limit of k, max. dCt {max_diff:.3e}, rel. dCt total: {rel_total_diff:.1f}%, cell: {rel_cell_diff:.1f}%')
            
            # when iteration not converged, set relevant results to zero
            Ct_g[:] = 0.
            Cu_g[:] = 0.
            pickup_g[:] = 0.
            pickup0_g[:] = 0.
            
            break
                
    # print(f'  Converged in {k} iterations.')

    # --- update Cu (final time for output) ----------------------------------
    Cu_g = update_Cu(Ct_g, Cu_air_g, Cu_bed_g, zeta_g)
    dCt = Ct_g - Ct_last

    # --- remove ghost cells -------------------------------------------------
    (Cu, Ct, pickup, pickup0, w, dCt, iters, ufs, ufn) = [
        _deghost(a) for a in (Cu_g, Ct_g, pickup_g, pickup0_g, w_g, dCt, iters, ufs_g, ufn_g)]

    return Cu, Ct, pickup, pickup0, w, dCt, iters


@njit
def update_Cu(Ct, Cu_air, Cu_bed, zeta):
    ''' Update equilibrium sediment concentration Cu,
    based on actual concentration Ct, bed-interaction factor zeta,
    airborne concentration Cu_air and bed concentration Cu_bed.
    '''
    ny, nx, nf = Ct.shape
    Cu = np.zeros((ny, nx, nf))

    # Loop over all cells and fractions
    for f in range(nf):
        for iy in range(ny):
            for ix in range(nx):

                ca = Cu_air[iy, ix, f]
                if ca == 0.0: # no air concentration, so no transport
                    Cu[iy, ix, f] = 0.0
                else: # compute weighted Cu
                    w_zeta_air = (1.0 - zeta[iy, ix]) * Ct[iy, ix, f] / ca
                    w_zeta_bed = 1.0 - w_zeta_air
                    Cu[iy, ix, f] = (w_zeta_air * ca + w_zeta_bed * Cu_bed[iy, ix, f])

    return Cu


@njit
def update_weights(Ct, Cu, mass, bi):
    ''' Update weights based on current Ct, Cu and bed mass.
    '''
    ny, nx, nf = Ct.shape
    w = np.zeros((ny, nx, nf))

    # loop over all cells
    for iy in range(ny):
        for ix in range(nx):

            # air contribution
            sum_air = 0.0
            for f in range(nf):
                ca = Cu[iy, ix, f]
                if ca != 0.0:
                    w[iy, ix, f] = Ct[iy, ix, f] / ca
                    sum_air += w[iy, ix, f]
                else:
                    w[iy, ix, f] = 0.0

            # bed contribution 
            sum_bed = 0.0
            for f in range(nf):
                sum_bed += mass[iy, ix, 0, f]
            scale_bed = 1.0 - min(1.0, (1.0 - bi) * sum_air)

            for f in range(nf):
                w[iy, ix, f] = (
                    (1.0 - bi) * w[iy, ix, f] +
                    (mass[iy, ix, 0, f] / sum_bed if sum_bed > 0.0 else 0.0)
                    * scale_bed
                )

            # normalize weights
            s = 0.0
            for f in range(nf):
                s += w[iy, ix, f]

            if s > 0.0:
                for f in range(nf):
                    w[iy, ix, f] /= s

    return w


def apply_boundary(Ct, Cu, uws, uwn,
                         offshore_bc, onshore_bc, lateral_bc,
                         offshore_flux, onshore_flux, lateral_flux):

    # --- pair enforcement for s-direction circular --------------------------
    if (offshore_bc == 'circular') != (onshore_bc == 'circular'):
        msg = "offshore_bc and onshore_bc must both be 'circular' (or both non-circular)."
        logger.error(msg)
        raise ValueError(msg)

    # ---- s-direction ghosts (west/east) ------------------------------------
    # circular: west ghost copies last physical col; east ghost copies first physical col
    if (offshore_bc == 'circular'):
        Ct[:, 0,  :] = Ct[:, -2, :]
        Ct[:, -1, :] = Ct[:,  1, :]
        Cu[:, 0,  :] = Cu[:, -2, :]
        Cu[:, -1, :] = Cu[:,  1, :]

    # non-circular: apply specified BCs
    else: 

        # inflow from west (offshore)
        if uws >= 0:  
            if offshore_bc == 'flux':
                Ct[:, 0, :] = offshore_flux * Cu[:, 1, :]
            elif offshore_bc == 'constant':
                Ct[:, 0, :] = Ct[:, 1, :]
            else:
                logger.error(f"Unknown offshore BC: {offshore_bc}")
        else:
            Ct[:, 0, :] = Ct[:, 1, :] # outflow: zero-gradient ghost

        # inflow from east (onshore)
        if uws < 0:   
            if onshore_bc == 'flux':
                Ct[:, -1, :] = onshore_flux * Cu[:, -2, :]
            elif onshore_bc == 'constant':
                Ct[:, -1, :] = Ct[:, -2, :]
            else:
                logger.error(f"Unknown onshore BC: {onshore_bc}")
        else:
            Ct[:, -1, :] = Ct[:, -2, :] # outflow: zero-gradient ghost

        # set Cu ghosts to zero-gradient
        Cu[:, 0, :] = Cu[:, 1, :]
        Cu[:, -1, :] = Cu[:, -2, :]

    # ----- n-direction ghosts (south/north) ---------------------------------
    if lateral_bc == 'circular':
        Ct[0,  :, :] = Ct[-2, :, :]
        Ct[-1, :, :] = Ct[ 1, :, :]
        Cu[0,  :, :] = Cu[-2, :, :]
        Cu[-1, :, :] = Cu[ 1, :, :]
    else:
        if uwn >= 0: # southern inflow
            if lateral_bc == 'flux':
                Ct[0, :, :] = lateral_flux * Cu[1, :, :]
            elif lateral_bc == 'constant':
                Ct[0, :, :] = Ct[1, :, :]
            else:
                logger.error(f"Unknown lateral BC: {lateral_bc}")
        else:
            Ct[0, :, :] = Ct[1, :, :]

        if uwn < 0: # northern inflow
            if lateral_bc == 'flux':
                Ct[-1, :, :] = lateral_flux * Cu[-2, :, :]
            elif lateral_bc == 'constant':
                Ct[-1, :, :] = Ct[-2, :, :]
            else:
                logger.error(f"Unknown lateral BC: {lateral_bc}")
        else:
            Ct[-1, :, :] = Ct[-2, :, :]

        # set Cu ghosts to zero-gradient
        Cu[0, :, :] = Cu[1, :, :]
        Cu[-1, :, :] = Cu[-2, :, :]

    return Ct, Cu


@njit(cache=True)
def _solve_quadrant1(Ct, Cu_air, Cu_bed, zeta, mass, pickup, pickup0, w,
                     dt, Ts, ds, dn, ufs, ufn, visited, quad, nf):
    ''' Sweep solver for quadrant 1 (u_n >=0, u_s >=0)'''

    # loop over all internal cells
    for n in range(1, Ct.shape[0] - 1):
        for s in range(1, Ct.shape[1] - 1):

            # check if cell is unvisited and flow is into quadrant 1
            if ((not visited[n, s]) and
                (ufn[n,s,0] >= 0) and (ufs[n,s,0] >= 0) and
                (ufn[n+1,s,0] >= 0) and (ufs[n,s+1,0] >= 0)):

                A = ds[n,s] * dn[n,s] # cell area

                # loop over all fractions
                for f in range(nf):
                    wf = w[n,s,f] # define weight

                    # compute a,b coefficients
                    if Cu_air[n,s,f] > 0 and Ct[n,s,f] > 0:
                        a = (1 - zeta[n,s]) * (Cu_air[n,s,f] - Cu_bed[n,s,f]) / Cu_air[n,s,f]
                        b = Cu_bed[n,s,f]
                    else:
                        a = 0.0
                        b = Cu_bed[n,s,f]

                    # nomninator and denominator definitions
                    N = (Ct[n-1,s,f] * ufn[n,s,f] * ds[n,s] +
                        Ct[n, s-1,f] * ufs[n,s,f] * dn[n,s])
                    D = (ufn[n+1,s,f] * ds[n,s] +
                        ufs[n, s+1,f] * dn[n,s] + A / Ts)
                    
                    # effective denominator adjustment
                    wa = wf * a
                    if wa > 0.999: wa = 0.999
                    
                    D_eff = D - wa * A / Ts
                    if D_eff < 1e-12: D_eff = 1e-12
                    
                    # solve unlimited state
                    N_unlim = N + wf * b * A/Ts
                    Ct_new = N_unlim / D_eff
                    Ct[n,s,f] = Ct_new
                    
                    # calculate pickup 
                    Cu_local = b + a * Ct_new
                    p = (wf * Cu_local - Ct_new) * dt/Ts

                    # copy of pickup before limiting
                    pickup0[n,s,f] = np.abs(p) 

                    # check against available mass
                    if p > mass[n,s,0,f]:
                        p_lim = mass[n,s,0,f]
                        
                        # solve limited state
                        N_lim = N + p_lim * A/dt
                        Ct_new = N_lim / D_eff
                        Ct[n,s,f] = Ct_new

                        # net pickup after deposition
                        deposition = (1.0 - wa) * Ct_new * dt / Ts
                        p = p_lim - deposition

                    # store pickup
                    pickup[n,s,f] = p

                # mark cell as visited and assign quadrant
                visited[n,s] = True
                quad[n,s] = 1


@njit(cache=True)
def _solve_quadrant2(Ct, Cu_air, Cu_bed, zeta, mass, pickup, pickup0, w,
                     dt, Ts, ds, dn, ufs, ufn, visited, quad, nf):
    ''' Sweep solver for quadrant 2 (u_n >=0, u_s <=0) '''

    for n in range(1, Ct.shape[0]-1):
        for s in range(Ct.shape[1]-2, 0, -1):

            if ((not visited[n,s]) and
                (ufn[n,s,0] >= 0) and (ufs[n,s,0] <= 0) and
                (ufn[n+1,s,0] >= 0) and (ufs[n,s+1,0] <= 0)):

                A = ds[n,s] * dn[n,s]

                for f in range(nf):
                    wf = w[n,s,f]

                    if Cu_air[n,s,f] > 0 and Ct[n,s,f] > 0:
                        a = (1 - zeta[n,s]) * (Cu_air[n,s,f] - Cu_bed[n,s,f]) / Cu_air[n,s,f]
                        b = Cu_bed[n,s,f]
                    else:
                        a = 0.0
                        b = Cu_bed[n,s,f]

                    N = (Ct[n-1,s,f] * ufn[n,s,f] * ds[n,s] +
                        -Ct[n,s+1,f] * ufs[n,s+1,f] * dn[n,s])

                    D = (ufn[n+1,s,f] * ds[n,s] +
                        -ufs[n,s,f] * dn[n,s] + A/Ts)

                    wa = wf * a
                    if wa > 0.999: wa = 0.999
                    
                    D_eff = D - wa * A / Ts
                    if D_eff < 1e-12: D_eff = 1e-12
                    
                    N_unlim = N + wf * b * A/Ts
                    Ct_new = N_unlim / D_eff
                    Ct[n,s,f] = Ct_new
                    
                    Cu_local = b + a * Ct_new
                    p = (wf * Cu_local - Ct_new) * dt/Ts
                    
                    pickup0[n,s,f] = np.abs(p)
                    
                    if p > mass[n,s,0,f]:
                        p_lim = mass[n,s,0,f]
                        
                        N_lim = N + p_lim * A/dt
                        Ct_new = N_lim / D_eff
                        Ct[n,s,f] = Ct_new

                        deposition = (1.0 - wa) * Ct_new * dt / Ts
                        p = p_lim - deposition

                    pickup[n,s,f] = p

                visited[n,s] = True
                quad[n,s] = 2


@njit(cache=True)
def _solve_quadrant3(Ct, Cu_air, Cu_bed, zeta, mass, pickup, pickup0, w,
                     dt, Ts, ds, dn, ufs, ufn, visited, quad, nf):
    ''' Sweep solver for quadrant 3 (u_n <=0, u_s <=0) '''

    for n in range(Ct.shape[0]-2, 0, -1):
        for s in range(Ct.shape[1]-2, 0, -1):

            if ((not visited[n,s]) and
                (ufn[n,s,0] <= 0) and (ufs[n,s,0] <= 0) and
                (ufn[n+1,s,0] <= 0) and (ufs[n,s+1,0] <= 0)):

                A = ds[n,s] * dn[n,s]

                for f in range(nf):
                    wf = w[n,s,f]

                    if Cu_air[n,s,f] > 0 and Ct[n,s,f] > 0:
                        a = (1 - zeta[n,s]) * (Cu_air[n,s,f] - Cu_bed[n,s,f]) / Cu_air[n,s,f]
                        b = Cu_bed[n,s,f]
                    else:
                        a = 0.0
                        b = Cu_bed[n,s,f]

                    # Note: Original code uses dn for ufn in Q3. Preserved for consistency.
                    N = (-Ct[n+1,s,f] * ufn[n+1,s,f] * dn[n,s] +
                        -Ct[n,s+1,f] * ufs[n,s+1,f] * dn[n,s])

                    D = (-ufn[n,s,f] * dn[n,s] +
                        -ufs[n,s,f] * dn[n,s] + A/Ts)

                    wa = wf * a
                    if wa > 0.999: wa = 0.999
                    
                    D_eff = D - wa * A / Ts
                    if D_eff < 1e-12: D_eff = 1e-12
                    
                    N_unlim = N + wf * b * A/Ts
                    Ct_new = N_unlim / D_eff
                    Ct[n,s,f] = Ct_new
                    
                    Cu_local = b + a * Ct_new
                    p = (wf * Cu_local - Ct_new) * dt/Ts
                    
                    pickup0[n,s,f] = np.abs(p)
                    
                    if p > mass[n,s,0,f]:
                        p_lim = mass[n,s,0,f]
                        
                        N_lim = N + p_lim * A/dt
                        Ct_new = N_lim / D_eff
                        Ct[n,s,f] = Ct_new

                        deposition = (1.0 - wa) * Ct_new * dt / Ts
                        p = p_lim - deposition

                    pickup[n,s,f] = p

                visited[n,s] = True
                quad[n,s] = 3


@njit(cache=True)
def _solve_quadrant4(Ct, Cu_air, Cu_bed, zeta, mass, pickup, pickup0, w,
                     dt, Ts, ds, dn, ufs, ufn, visited, quad, nf):
    ''' Sweep solver for quadrant 4 (u_n <=0, u_s >=0) '''

    for n in range(Ct.shape[0]-2, 0, -1):
        for s in range(1, Ct.shape[1]-1):

            if ((not visited[n,s]) and
                (ufn[n,s,0] <= 0) and (ufs[n,s,0] >= 0) and
                (ufn[n+1,s,0] <= 0) and (ufs[n,s+1,0] >= 0)):

                A = ds[n,s] * dn[n,s]

                for f in range(nf):
                    wf = w[n,s,f]

                    if Cu_air[n,s,f] > 0 and Ct[n,s,f] > 0:
                        a = (1 - zeta[n,s])*(Cu_air[n,s,f] - Cu_bed[n,s,f]) / Cu_air[n,s,f]
                        b = Cu_bed[n,s,f]
                    else:
                        a = 0.0
                        b = Cu_bed[n,s,f]

                    # Note: Original code uses dn for ufn in Q4. Preserved for consistency.
                    N = (Ct[n,s-1,f] * ufs[n,s,f] * dn[n,s] +
                        -Ct[n+1,s,f] * ufn[n+1,s,f] * dn[n,s])

                    D = (ufs[n,s+1,f] * dn[n,s] +
                        -ufn[n,s,f] * dn[n,s] + A/Ts)
                    
                    wa = wf * a
                    if wa > 0.999: wa = 0.999
                    
                    D_eff = D - wa * A / Ts
                    if D_eff < 1e-12: D_eff = 1e-12
                    
                    N_unlim = N + wf * b * A/Ts
                    Ct_new = N_unlim / D_eff
                    Ct[n,s,f] = Ct_new
                    
                    Cu_local = b + a * Ct_new
                    p = (wf * Cu_local - Ct_new) * dt/Ts
                    
                    pickup0[n,s,f] = np.abs(p)
                    
                    if p > mass[n,s,0,f]:
                        p_lim = mass[n,s,0,f]
            
                        N_lim = N + p_lim * A/dt
                        Ct_new = N_lim / D_eff
                        Ct[n,s,f] = Ct_new

                        deposition = (1.0 - wa) * Ct_new * dt / Ts
                        p = p_lim - deposition

                    pickup[n,s,f] = p

                visited[n,s] = True
                quad[n,s] = 4


@njit(cache=True)
def _solve_generic_stencil(Ct, Cu_air, Cu_bed, zeta, mass, pickup, pickup0, w,
                           dt, Ts, ds, dn, ufs, ufn, visited, quad, nf):

    Nx, Ny = Ct.shape[0], Ct.shape[1]

    for n in range(1, Nx-1):
        for s in range(1, Ny-1):

            if not visited[n,s]:

                A = ds[n,s] * dn[n,s]

                for f in range(nf):
                    wf = w[n,s,f]

                    if Cu_air[n,s,f] > 0 and Ct[n,s,f] > 0:
                        a = (1 - zeta[n,s])*(Cu_air[n,s,f] - Cu_bed[n,s,f]) / Cu_air[n,s,f]
                        b = Cu_bed[n,s,f]
                    else:
                        a = 0.0
                        b = Cu_bed[n,s,f]

                    # start with source term
                    N = (
                        (Ct[n-1,s,f] * ufn[n,s,f] * ds[n,s] if ufn[n,s,0] > 0 else 0) +
                        (Ct[n,s-1,f] * ufs[n,s,f] * dn[n,s] if ufs[n,s,0] > 0 else 0) +
                        (-Ct[n+1,s,f] * ufn[n+1,s,f] * dn[n,s] if ufn[n+1,s,0] <= 0 else 0) +
                        (-Ct[n,s+1,f] * ufs[n,s+1,f] * dn[n,s] if ufs[n,s+1,0] <= 0 else 0)
                    )

                    D = A/Ts
                    if ufn[n,s,0] <= 0: D += -ufn[n,s,f] * dn[n,s]
                    if ufs[n,s,0] <= 0: D += -ufs[n,s,f] * dn[n,s]
                    if ufn[n+1,s,0] > 0: D += ufn[n+1,s,f] * ds[n,s]
                    if ufs[n,s+1,0] > 0: D += ufs[n,s+1,f] * dn[n,s]

                    wa = wf * a
                    if wa > 0.999: wa = 0.999
                    
                    D_eff = D - wa * A / Ts
                    if D_eff < 1e-12: D_eff = 1e-12
                    
                    N_unlim = N + wf * b * A/Ts
                    Ct_new = N_unlim / D_eff
                    Ct[n,s,f] = Ct_new
                    
                    Cu_local = b + a * Ct_new
                    p = (wf * Cu_local - Ct_new) * dt/Ts
                    
                    pickup0[n,s,f] = np.abs(p)
                    
                    if p > mass[n,s,0,f]:
                        p_lim = mass[n,s,0,f]
                        
                        N_lim = N + p_lim * A/dt
                        Ct_new = N_lim / D_eff
                        Ct[n,s,f] = Ct_new

                        deposition = (1.0 - wa) * Ct_new * dt / Ts
                        p = p_lim - deposition

                    pickup[n,s,f] = p

                visited[n,s] = True
                quad[n,s] = 5


def _ghostify(arr, offshore_bc=None, onshore_bc=None, lateral_bc=None):
    """
    Add 1-cell ghost halo in y and x and fill ghosts according to BCs.
    Assumes arr is physical domain (ny, nx, ...).
    """

    # allocate halo
    if arr.ndim == 2:
        ny, nx = arr.shape
        out = np.empty((ny+2, nx+2), dtype=arr.dtype)
        out[1:-1, 1:-1] = arr

    elif arr.ndim == 3:
        ny, nx, nf = arr.shape
        out = np.empty((ny+2, nx+2, nf), dtype=arr.dtype)
        out[1:-1, 1:-1, :] = arr

    elif arr.ndim == 4:
        ny, nx, a, b = arr.shape
        out = np.empty((ny+2, nx+2, a, b), dtype=arr.dtype)
        out[1:-1, 1:-1, :, :] = arr

    else:
        raise ValueError("Unsupported ndim for ghostify")

    # --- fill west/east ghosts (s-direction) --------------------------------
    if offshore_bc == 'circular' and onshore_bc == 'circular':
        out[:, 0,  ...] = out[:, -2, ...]
        out[:, -1, ...] = out[:,  1, ...]
    else:
        # zero-gradient default
        out[:, 0,  ...] = out[:, 1,  ...]
        out[:, -1, ...] = out[:, -2, ...]

    # --- fill south/north ghosts (n-direction) ------------------------------
    if lateral_bc == 'circular':
        out[0,  :, ...] = out[-2, :, ...]
        out[-1, :, ...] = out[ 1, :, ...]
    else:
        # zero-gradient default
        out[0,  :, ...] = out[1,  :, ...]
        out[-1, :, ...] = out[-2, :, ...]

    return out


def _deghost(arr_g):
    """Remove 1-cell halo."""
    return arr_g[1:-1, 1:-1, ...]
