# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 12:05:32 2021

@author: new
"""
import torch
from torch.autograd import Function, Variable
from torch.nn import Module
import numpy as np
from collections import namedtuple
import time
from enum import Enum
import sys

WARNING_INDEX = 0

import operator
import torch.nn as nn
import torch.nn.functional as F
import itertools


QuadCost = namedtuple('QuadCost', 'C c')
LinDx = namedtuple('LinDx', 'F f')

#QuadCost.__new__.__defaults__ = (None,) * len(QuadCost._fields)
#LinDx.__new__.__defaults__ = (None,) * len(LinDx._fields)


LqrBackOut = namedtuple('lqrBackOut', 'n_total_qp_iter')
LqrForOut = namedtuple(
    'lqrForOut',
    'objs full_du_norm alpha_du_norm mean_alphas costs')
    

class CtrlPassthroughDynamics(nn.Module):
    def __init__(self, dynamics):
        super().__init__()
        self.dynamics = dynamics

    def forward(self, tilde_x, u):
        tilde_x_dim, u_dim = tilde_x.ndimension(), u.ndimension()
        if tilde_x_dim == 1:
            tilde_x = tilde_x.unsqueeze(0)
        if u_dim == 1:
            u = u.unsqueeze(0)

        n_ctrl = u.size(1)
        x = tilde_x[:,n_ctrl:]
        xtp1 = self.dynamics(x, u)
        tilde_xtp1 = torch.cat((u, xtp1), dim=1)

        if tilde_x_dim == 1:
            tilde_xtp1 = tilde_xtp1.squeeze()

        return tilde_xtp1

    def grad_input(self, x, u):
        assert False, "Unimplemented"




def grad(net, inputs, eps=1e-4):
    assert(inputs.ndimension() == 2)
    nBatch, nDim = inputs.size()
    xp, xn = [], []
    e = 0.5*eps*torch.eye(nDim).type_as(inputs.data)
    for b in range(nBatch):
        for i in range(nDim):
            xp.append((inputs.data[b].clone()+e[i]).unsqueeze(0))
            xn.append((inputs.data[b].clone()-e[i]).unsqueeze(0))
    xs = Variable(torch.cat(xp+xn))
    fs = net(xs)
    fDim = fs.size(1) if fs.ndimension() > 1 else 1
    fs_p, fs_n = torch.split(fs, nBatch*nDim)
    g = ((fs_p-fs_n)/eps).view(nBatch, nDim, fDim).squeeze(2)
    return g



def hess(net, inputs, eps=1e-4):
    assert(inputs.ndimension() == 2)
    nBatch, nDim = inputs.size()
    xpp, xpn, xnp, xnn = [], [], [], []
    e = eps*torch.eye(nDim).type_as(inputs.data)
    for b,i,j in itertools.product(range(nBatch), range(nDim), range(nDim)):
        xpp.append((inputs.data[b].clone()+e[i]+e[j]).unsqueeze(0))
        xpn.append((inputs.data[b].clone()+e[i]-e[j]).unsqueeze(0))
        xnp.append((inputs.data[b].clone()-e[i]+e[j]).unsqueeze(0))
        xnn.append((inputs.data[b].clone()-e[i]-e[j]).unsqueeze(0))
    xs = Variable(torch.cat(xpp+xpn+xnp+xnn))
    fs = net(xs)
    fDim = fs.size(1) if fs.ndimension() > 1 else 1
    fpp, fpn, fnp, fnn = torch.split(fs, nBatch*nDim*nDim)
    h = ((fpp-fpn-fnp+fnn)/(4*eps*eps)).view(nBatch, nDim, nDim, fDim).squeeze(3)
    return h




def jacobian(f, x, eps):
    if x.ndimension() == 2:
        assert x.size(0) == 1
        x = x.squeeze()

    e = Variable(torch.eye(len(x)).type_as(get_data_maybe(x)))
    J = []
    for i in range(len(x)):
        J.append((f(x + eps*e[i]) - f(x - eps*e[i]))/(2.*eps))
    J = torch.stack(J).transpose(0,1)
    return J


def expandParam(X, n_batch, nDim):
    if X.ndimension() in (0, nDim):
        return X, False
    elif X.ndimension() == nDim - 1:
        return X.unsqueeze(0).expand(*([n_batch] + list(X.size()))), True
    else:
        raise RuntimeError("Unexpected number of dimensions.")


def bdiag(d):
    assert d.ndimension() == 2
    nBatch, sz = d.size()
    dtype = d.type() if not isinstance(d, Variable) else d.data.type()
    D = torch.zeros(nBatch, sz, sz).type(dtype)
    I = torch.eye(sz).repeat(nBatch, 1, 1).type(dtype).byte()
    D[I] = d.view(-1)
    return D


def bger(x, y):
    return x.unsqueeze(2).bmm(y.unsqueeze(1))


def bmv(X, y):
    return X.bmm(y.unsqueeze(2)).squeeze(2)


def bquad(x, Q):
    return x.unsqueeze(1).bmm(Q).bmm(x.unsqueeze(2)).squeeze(1).squeeze(1)


def bdot(x, y):
    return torch.bmm(x.unsqueeze(1), y.unsqueeze(2)).squeeze(1).squeeze(1)


def eclamp(x, lower, upper):
    # In-place!!
    if type(lower) == type(x):
        assert x.size() == lower.size()

    if type(upper) == type(x):
        assert x.size() == upper.size()

    I = x < lower
    x[I] = lower[I] if not isinstance(lower, float) else lower

    I = x > upper
    x[I] = upper[I] if not isinstance(upper, float) else upper

    return x


def get_data_maybe(x):
    return x if not isinstance(x, Variable) else x.data


_seen_tables = []
def table_log(tag, d):
    # TODO: There's probably a better way to handle formatting here,
    # or a better way altogether to replace this quick hack.
    global _seen_tables

    def print_row(r):
        print('| ' + ' | '.join(r) + ' |')

    if tag not in _seen_tables:
        print_row(map(operator.itemgetter(0), d))
        _seen_tables.append(tag)

    s = []
    for di in d:
        assert len(di) in [2,3]
        if len(di) == 3:
            e, fmt = di[1:]
            s.append(fmt.format(e))
        else:
            e = di[1]
            s.append(str(e))
    print_row(s)


def get_traj(T, u, x_init, dynamics):

    if isinstance(dynamics, LinDx):
        F = get_data_maybe(dynamics.F)
        f = get_data_maybe(dynamics.f)
        if f is not None:
            assert f.shape == F.shape[:3]

    x = [get_data_maybe(x_init)]
    for t in range(T):
        xt = x[t]
        ut = get_data_maybe(u[t])
        if t < T-1:
            # new_x = f(Variable(xt), Variable(ut)).data
            if isinstance(dynamics, LinDx):
                xut = torch.cat((xt, ut), 1)
                new_x = bmv(F[t], xut)
                if f is not None:
                    new_x += f[t]
            else:
                new_x = dynamics.forward( Variable(xt), Variable(ut)).data # lwh, add dynamics
            x.append(new_x)
    x = torch.stack(x, dim=0)
    return x


def get_cost(T, u, cost, dynamics=None, x_init=None, x=None):

    assert x_init is not None or x is not None

    if isinstance(cost, QuadCost):
        C = get_data_maybe(cost.C)
        c = get_data_maybe(cost.c)

    if x is None:
        x = get_traj(T, u, x_init, dynamics)

    objs = []
    for t in range(T):
        xt = x[t]
        ut = u[t]
        xut = torch.cat((xt, ut), 1)
        if isinstance(cost, QuadCost):
            obj = 0.5*bquad(xut, C[t]) + bdot(xut, c[t])
        else:
            obj = cost(xut)
        objs.append(obj)
    objs = torch.stack(objs, dim=0)
    total_obj = torch.sum(objs, dim=0)
    return total_obj


def detach_maybe(x):
    if x is None:
        return None
    return x if not x.requires_grad else x.detach()


def data_maybe(x):
    if x is None:
        return None
    return x.data


# @profile
def pnqp(H, q, lower, upper, x_init=None, n_iter=20):
    GAMMA = 0.1
    n_batch, n, _ = H.size()
    pnqp_I = 1e-11*torch.eye(n).type_as(H).expand_as(H)


    def obj(x):
        return 0.5*bquad(x, H) + bdot(q, x)

    if x_init is None:
        if n == 1:
            x_init = -(1./H.squeeze(2))*q
        else:
            H_lu = H.lu()
            x_init = -q.unsqueeze(2).lu_solve(*H_lu).squeeze(2) # Clamped in the x assignment.
    else:
        x_init = x_init.clone() # Don't over-write the original x_init.

    x = eclamp(x_init, lower, upper)

    # Active examples in the batch.
    J = torch.ones(n_batch).type_as(x).byte()

    for i in range(n_iter):
        g = bmv(H, x) + q

        # TODO: Could clean up the types here.
        Ic = (((x == lower) & (g > 0)) | ((x == upper) & (g < 0))).float()
        If = 1-Ic

        if If.is_cuda:
            Hff_I = bger(If.float(), If.float()).type_as(If)
            not_Hff_I = 1-Hff_I
            Hfc_I = bger(If.float(), Ic.float()).type_as(If)
        else:
            Hff_I = bger(If, If)
            not_Hff_I = 1-Hff_I
            Hfc_I = bger(If, Ic)

        g_ = g.clone()
        g_[Ic.bool()] = 0.
        H_ = H.clone()
        H_[not_Hff_I.bool()] = 0.0
        H_ += pnqp_I

        if n == 1:
            dx = -(1./H_.squeeze(2))*g_
        else:
            H_lu_ = H_.lu()
            dx = -g_.unsqueeze(2).lu_solve(*H_lu_).squeeze(2)

        J = torch.norm(dx, 2, 1) >= 1e-4
        m = J.sum().item() # Number of active examples in the batch.
        if m == 0:
            return x, H_ if n == 1 else H_lu_, If, i

        alpha = torch.ones(n_batch).type_as(x)
        decay = 0.1
        max_armijo = GAMMA
        count = 0
        while max_armijo <= GAMMA and count < 10:
            # Crude way of making sure too much time isn't being spent
            # doing the line search.
            # assert count < 10

            maybe_x = eclamp(x+torch.diag(alpha).mm(dx), lower, upper)
            armijos = (GAMMA+1e-6)*torch.ones(n_batch).type_as(x)
            armijos[J] = (obj(x)-obj(maybe_x))[J]/bdot(g, x-maybe_x)[J]
            I = armijos <= GAMMA
            alpha[I] *= decay
            max_armijo = torch.max(armijos)
            count += 1

        x = maybe_x

    # TODO: Maybe change this to a warning.
    print("[WARNING] pnqp warning: Did not converge")
    return x, H_ if n == 1 else H_lu_, If, i





    

class LQRStep(Function):
    """A single step of the box-constrained iLQR solver.

    Required Args:
        n_state, n_ctrl, T
        x_init: The initial state [n_batch, n_state]

    Optional Args:
        u_lower, u_upper: The lower- and upper-bounds on the controls.
            These can either be floats or shaped as [T, n_batch, n_ctrl]
            TODO: Better support automatic expansion of these.
        TODO
    """

    def __init__(
            self,
            n_state,
            n_ctrl,
            T,
            u_lower=None,
            u_upper=None,
            u_zero_I=None,
            delta_u=None,
            linesearch_decay=0.2,
            max_linesearch_iter=10,
            true_cost=None,
            true_dynamics=None,
            delta_space=True,
            current_x=None,
            current_u=None,
            verbose=0,
            back_eps=1e-3,
            no_op_forward=False,
    ):
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.T = T

        self.u_lower = get_data_maybe(u_lower)
        self.u_upper = get_data_maybe(u_upper)

        # TODO: Better checks for this
        if isinstance(self.u_lower, int):
            self.u_lower = float(self.u_lower)
        if isinstance(self.u_upper, int):
            self.u_upper = float(self.u_upper)
        if isinstance(self.u_lower, np.float32):
            self.u_lower = u_lower.item()
        if isinstance(self.u_upper, np.float32):
            self.u_upper = u_upper.item()

        self.u_zero_I = u_zero_I
        self.delta_u = delta_u
        self.linesearch_decay = linesearch_decay
        self.max_linesearch_iter = max_linesearch_iter
        self.true_cost = true_cost
        self.true_dynamics = true_dynamics
        self.delta_space = delta_space
        self.current_x = get_data_maybe(current_x)
        self.current_u = get_data_maybe(current_u)
        self.verbose = verbose

        self.back_eps = back_eps

        self.no_op_forward = no_op_forward

    # @profile
    def forward(self, x_init, C, c, F, f=None):
        if self.no_op_forward:
            self.save_for_backward(
                x_init, C, c, F, f, self.current_x, self.current_u)
            return self.current_x, self.current_u

        if self.delta_space:
            # Taylor-expand the objective to do the backward pass in
            # the delta space.
            assert self.current_x is not None
            assert self.current_u is not None
            c_back = []
            for t in range(self.T):
                xt = self.current_x[t]
                ut = self.current_u[t]
                xut = torch.cat((xt, ut), 1)
                c_back.append(bmv(C[t], xut) + c[t])
            c_back = torch.stack(c_back)
            f_back = None
        else:
            assert False

        Ks, ks, self.back_out = self.lqr_backward(C, c_back, F, f_back)
        new_x, new_u, self.for_out = self.lqr_forward(
            x_init, C, c, F, f, Ks, ks)
        self.save_for_backward(x_init, C, c, F, f, new_x, new_u)

        return new_x, new_u

    def backward(self, dl_dx, dl_du):
        start = time.time()
        x_init, C, c, F, f, new_x, new_u = self.saved_tensors

        r = []
        for t in range(self.T):
            rt = torch.cat((dl_dx[t], dl_du[t]), 1)
            r.append(rt)
        r = torch.stack(r)

        if self.u_lower is None:
            I = None
        else:
            I = (torch.abs(new_u - self.u_lower) <= 1e-8) | \
                (torch.abs(new_u - self.u_upper) <= 1e-8)
        dx_init = Variable(torch.zeros_like(x_init))
        _mpc = MPC(
            self.n_state, self.n_ctrl, self.T,
            u_zero_I=I,
            u_init=None,
            lqr_iter=1,
            verbose=-1,
            n_batch=C.size(1),
            delta_u=None,
            # exit_unconverged=True, # It's really bad if this doesn't converge.
            exit_unconverged=False, # It's really bad if this doesn't converge.
            eps=self.back_eps,
        )
        dx, du, _ = _mpc(dx_init, mpc.QuadCost(C, -r), mpc.LinDx(F, None))

        dx, du = dx.data, du.data
        dxu = torch.cat((dx, du), 2)
        xu = torch.cat((new_x, new_u), 2)

        dC = torch.zeros_like(C)
        for t in range(self.T):
            xut = torch.cat((new_x[t], new_u[t]), 1)
            dxut = dxu[t]
            dCt = -0.5*(bger(dxut, xut) + bger(xut, dxut))
            dC[t] = dCt

        dc = -dxu

        lams = []
        prev_lam = None
        for t in range(self.T-1, -1, -1):
            Ct_xx = C[t,:,:self.n_state,:self.n_state]
            Ct_xu = C[t,:,:self.n_state,self.n_state:]
            ct_x = c[t,:,:self.n_state]
            xt = new_x[t]
            ut = new_u[t]
            lamt = bmv(Ct_xx, xt) + bmv(Ct_xu, ut) + ct_x
            if prev_lam is not None:
                Fxt = F[t,:,:,:self.n_state].transpose(1, 2)
                lamt += bmv(Fxt, prev_lam)
            lams.append(lamt)
            prev_lam = lamt
        lams = list(reversed(lams))

        dlams = []
        prev_dlam = None
        for t in range(self.T-1, -1, -1):
            dCt_xx = C[t,:,:self.n_state,:self.n_state]
            dCt_xu = C[t,:,:self.n_state,self.n_state:]
            drt_x = -r[t,:,:self.n_state]
            dxt = dx[t]
            dut = du[t]
            dlamt = bmv(dCt_xx, dxt) + bmv(dCt_xu, dut) + drt_x
            if prev_dlam is not None:
                Fxt = F[t,:,:,:self.n_state].transpose(1, 2)
                dlamt += bmv(Fxt, prev_dlam)
            dlams.append(dlamt)
            prev_dlam = dlamt
        dlams = torch.stack(list(reversed(dlams)))

        dF = torch.zeros_like(F)
        for t in range(self.T-1):
            xut = xu[t]
            lamt = lams[t+1]

            dxut = dxu[t]
            dlamt = dlams[t+1]

            dF[t] = -(bger(dlamt, xut) + bger(lamt, dxut))

        if f.nelement() > 0:
            _dlams = dlams[1:]
            assert _dlams.shape == f.shape
            df = -_dlams
        else:
            df = torch.Tensor()

        dx_init = -dlams[0]

        self.backward_time = time.time()-start
        return dx_init, dC, dc, dF, df

    # @profile
    def lqr_backward(self, C, c, F, f):
        n_batch = C.size(1)

        u = self.current_u
        Ks = []
        ks = []
        prev_kt = None
        n_total_qp_iter = 0
        Vtp1 = vtp1 = None
        for t in range(self.T-1, -1, -1):
            if t == self.T-1:
                Qt = C[t]
                qt = c[t]
            else:
                Ft = F[t]
                Ft_T = Ft.transpose(1,2)
                Qt = C[t] + Ft_T.bmm(Vtp1).bmm(Ft)
                if f is None or f.nelement() == 0:
                    qt = c[t] + Ft_T.bmm(vtp1.unsqueeze(2)).squeeze(2)
                else:
                    ft = f[t]
                    qt = c[t] + Ft_T.bmm(Vtp1).bmm(ft.unsqueeze(2)).squeeze(2) + \
                        Ft_T.bmm(vtp1.unsqueeze(2)).squeeze(2)

            n_state = self.n_state
            Qt_xx = Qt[:, :n_state, :n_state]
            Qt_xu = Qt[:, :n_state, n_state:]
            Qt_ux = Qt[:, n_state:, :n_state]
            Qt_uu = Qt[:, n_state:, n_state:]
            qt_x = qt[:, :n_state]
            qt_u = qt[:, n_state:]

            if self.u_lower is None:
                if self.n_ctrl == 1 and self.u_zero_I is None:
                    Kt = -(1./Qt_uu)*Qt_ux
                    kt = -(1./Qt_uu.squeeze(2))*qt_u
                else:
                    if self.u_zero_I is None:
                        Qt_uu_inv = [
                            torch.pinverse(Qt_uu[i]) for i in range(Qt_uu.shape[0])
                        ]
                        Qt_uu_inv = torch.stack(Qt_uu_inv)
                        Kt = -Qt_uu_inv.bmm(Qt_ux)
                        kt = bmv(-Qt_uu_inv, qt_u)

                        # Qt_uu_LU = Qt_uu.lu()
                        # Kt = -Qt_ux.lu_solve(*Qt_uu_LU)
                        # kt = -qt_u.lu_solve(*Qt_uu_LU)
                    else:
                        # Solve with zero constraints on the active controls.
                        I = self.u_zero_I[t].float()
                        notI = 1-I

                        qt_u_ = qt_u.clone()
                        qt_u_[I.bool()] = 0

                        Qt_uu_ = Qt_uu.clone()

                        if I.is_cuda:
                            notI_ = notI.float()
                            Qt_uu_I = (1-bger(notI_, notI_)).type_as(I)
                        else:
                            Qt_uu_I = 1-bger(notI, notI)

                        Qt_uu_[Qt_uu_I.bool()] = 0.
                        Qt_uu_[bdiag(I).bool()] += 1e-8

                        Qt_ux_ = Qt_ux.clone()
                        Qt_ux_[I.unsqueeze(2).repeat(1,1,Qt_ux.size(2)).bool()] = 0.

                        if self.n_ctrl == 1:
                            Kt = -(1./Qt_uu_)*Qt_ux_
                            kt = -(1./Qt_uu.squeeze(2))*qt_u_
                        else:
                            Qt_uu_LU_ = Qt_uu_.lu()
                            Kt = -Qt_ux_.lu_solve(*Qt_uu_LU_)
                            kt = -qt_u_.unsqueeze(2).lu_solve(*Qt_uu_LU_).squeeze(2)
            else:
                assert self.delta_space
                lb = self.get_bound('lower', t) - u[t]
                ub = self.get_bound('upper', t) - u[t]
                if self.delta_u is not None:
                    lb[lb < -self.delta_u] = -self.delta_u
                    ub[ub > self.delta_u] = self.delta_u
                kt, Qt_uu_free_LU, If, n_qp_iter = pnqp(
                    Qt_uu, qt_u, lb, ub,
                    x_init=prev_kt, n_iter=20)
                if self.verbose > 1:
                    print('  + n_qp_iter: ', n_qp_iter+1)
                n_total_qp_iter += 1+n_qp_iter
                prev_kt = kt
                Qt_ux_ = Qt_ux.clone()
                Qt_ux_[(1-If).unsqueeze(2).repeat(1,1,Qt_ux.size(2)).bool()] = 0
                if self.n_ctrl == 1:
                    # Bad naming, Qt_uu_free_LU isn't the LU in this case.
                    Kt = -((1./Qt_uu_free_LU)*Qt_ux_)
                else:
                    Kt = -Qt_ux_.lu_solve(*Qt_uu_free_LU)

            Kt_T = Kt.transpose(1,2)

            Ks.append(Kt)
            ks.append(kt)

            Vtp1 = Qt_xx + Qt_xu.bmm(Kt) + Kt_T.bmm(Qt_ux) + Kt_T.bmm(Qt_uu).bmm(Kt)
            vtp1 = qt_x + Qt_xu.bmm(kt.unsqueeze(2)).squeeze(2) + \
                Kt_T.bmm(qt_u.unsqueeze(2)).squeeze(2) + \
                Kt_T.bmm(Qt_uu).bmm(kt.unsqueeze(2)).squeeze(2)

        return Ks, ks, LqrBackOut(n_total_qp_iter=n_total_qp_iter)


    # @profile
    def lqr_forward(self, x_init, C, c, F, f, Ks, ks):
        x = self.current_x
        u = self.current_u
        n_batch = C.size(1)

        old_cost = get_cost(self.T, u, self.true_cost, self.true_dynamics, x=x)

        current_cost = None
        alphas = torch.ones(n_batch).type_as(C)
        full_du_norm = None

        i = 0
        while (current_cost is None or \
               (old_cost is not None and \
                  torch.any((current_cost > old_cost)).cpu().item() == 1)) and \
              i < self.max_linesearch_iter:
            new_u = []
            new_x = [x_init]
            dx = [torch.zeros_like(x_init)]
            objs = []
            for t in range(self.T):
                t_rev = self.T-1-t
                Kt = Ks[t_rev]
                kt = ks[t_rev]
                new_xt = new_x[t]
                xt = x[t]
                ut = u[t]
                dxt = dx[t]
                new_ut = bmv(Kt, dxt) + ut + torch.diag(alphas).mm(kt)

                # Currently unimplemented:
                assert not ((self.delta_u is not None) and (self.u_lower is None))

                if self.u_zero_I is not None:
                    new_ut[self.u_zero_I[t]] = 0.

                if self.u_lower is not None:
                    lb = self.get_bound('lower', t)
                    ub = self.get_bound('upper', t)

                    if self.delta_u is not None:
                        lb_limit, ub_limit = lb, ub
                        lb = u[t] - self.delta_u
                        ub = u[t] + self.delta_u
                        I = lb < lb_limit
                        lb[I] = lb_limit if isinstance(lb_limit, float) else lb_limit[I]
                        I = ub > ub_limit
                        ub[I] = ub_limit if isinstance(lb_limit, float) else ub_limit[I]
                    new_ut = eclamp(new_ut, lb, ub)
                new_u.append(new_ut)

                new_xut = torch.cat((new_xt, new_ut), dim=1)
                if t < self.T-1:
                    if isinstance(self.true_dynamics, LinDx):
                        F, f = self.true_dynamics.F, self.true_dynamics.f
                        new_xtp1 = bmv(F[t], new_xut)
                        if f is not None and f.nelement() > 0:
                            new_xtp1 += f[t]
                    else:
                        new_xtp1 = self.true_dynamics(
                            Variable(new_xt), Variable(new_ut)).data

                    new_x.append(new_xtp1)
                    dx.append(new_xtp1 - x[t+1])

                if isinstance(self.true_cost, QuadCost):
                    C, c = self.true_cost.C, self.true_cost.c
                    obj = 0.5*bquad(new_xut, C[t]) + bdot(new_xut, c[t])
                else:
                    obj = self.true_cost(new_xut)

                objs.append(obj)

            objs = torch.stack(objs)
            current_cost = torch.sum(objs, dim=0)

            new_u = torch.stack(new_u)
            new_x = torch.stack(new_x)
            if full_du_norm is None:
                full_du_norm = (u-new_u).transpose(1,2).contiguous().view(
                    n_batch, -1).norm(2, 1)

            alphas[current_cost > old_cost] *= self.linesearch_decay
            i += 1

        # If the iteration limit is hit, some alphas
        # are one step too small.
        alphas[current_cost > old_cost] /= self.linesearch_decay
        alpha_du_norm = (u-new_u).transpose(1,2).contiguous().view(
            n_batch, -1).norm(2, 1)

        return new_x, new_u, LqrForOut(
            objs, full_du_norm,
            alpha_du_norm,
            torch.mean(alphas),
            current_cost
        )


    def get_bound(self, side, t):
        v = getattr(self, 'u_'+side)
        if isinstance(v, float):
            return v
        else:
            return v[t]










class GradMethods(Enum):
    AUTO_DIFF = 1
    FINITE_DIFF = 2
    ANALYTIC = 3
    ANALYTIC_CHECK = 4


class SlewRateCost(Module):
    """Hacky way of adding the slew rate penalty to costs."""
    # TODO: It would be cleaner to update this to just use the slew
    # rate penalty instead of # slew_C
    def __init__(self, cost, slew_C, n_state, n_ctrl):
        super().__init__()
        self.cost = cost
        self.slew_C = slew_C
        self.n_state = n_state
        self.n_ctrl = n_ctrl

    def forward(self, tau):
        true_tau = tau[:, self.n_ctrl:]
        true_cost = self.cost(true_tau)
        # The slew constraints are time-invariant.
        slew_cost = 0.5 * bquad(tau, self.slew_C[0])
        return true_cost + slew_cost

    def grad_input(self, x, u):
        raise NotImplementedError("Implement grad_input")


class MPC(Module):
    """A differentiable box-constrained iLQR solver.

    This provides a differentiable solver for the following box-constrained
    control problem with a quadratic cost (defined by C and c) and
    non-linear dynamics (defined by f):

        min_{tau={x,u}} sum_t 0.5 tau_t^T C_t tau_t + c_t^T tau_t
                        s.t. x_{t+1} = f(x_t, u_t)
                            x_0 = x_init
                            u_lower <= u <= u_upper

    This implements the Control-Limited Differential Dynamic Programming
    paper with a first-order approximation to the non-linear dynamics:
    https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf

    Some of the notation here is from Sergey Levine's notes:
    http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_8_model_based_planning.pdf

    Required Args:
        n_state, n_ctrl, T

    Optional Args:
        u_lower, u_upper: The lower- and upper-bounds on the controls.
            These can either be floats or shaped as [T, n_batch, n_ctrl]
        u_init: The initial control sequence, useful for warm-starting:
            [T, n_batch, n_ctrl]
        lqr_iter: The number of LQR iterations to perform.
        grad_method: The method to compute the Jacobian of the dynamics.
            GradMethods.ANALYTIC: Use a manually-defined Jacobian.
                + Fast and accurate, use this if possible
            GradMethods.AUTO_DIFF: Use PyTorch's autograd.
                + Slow
            GradMethods.FINITE_DIFF: Use naive finite differences
                + Inaccurate
        delta_u (float): The amount each component of the controls
            is allowed to change in each LQR iteration.
        verbose (int):
            -1: No output or warnings
             0: Warnings
            1+: Detailed iteration info
        eps: Termination threshold, on the norm of the full control
             step (without line search)
        back_eps: `eps` value to use in the backwards pass.
        n_batch: May be necessary for now if it can't be inferred.
                 TODO: Infer, potentially remove this.
        linesearch_decay (float): Multiplicative decay factor for the
            line search.
        max_linesearch_iter (int): Can be used to disable the line search
            if 1 is used for some problems the line search can
            be harmful.
        exit_unconverged: Assert False if a fixed point is not reached.
        detach_unconverged: Detach examples from the graph that do
            not hit a fixed point so they are not differentiated through.
        backprop: Allow the solver to be differentiated through.
        slew_rate_penalty (float): Penalty term applied to
            ||u_t - u_{t+1}||_2^2 in the objective.
        prev_ctrl: The previous nominal control sequence to initialize
            the solver with.
        not_improved_lim: The number of iterations to allow that don't
            improve the objective before returning early.
        best_cost_eps: Absolute threshold for the best cost
            to be updated.
    """

    def __init__(
            self, n_state, n_ctrl, T,
            u_lower=None, u_upper=None,
            u_zero_I=None,
            u_init=None,
            lqr_iter=10,
            grad_method=GradMethods.ANALYTIC,
            delta_u=None,
            verbose=0,
            eps=1e-7,
            back_eps=1e-7,
            n_batch=None,
            linesearch_decay=0.1, #0.2
            max_linesearch_iter=30, #10
            exit_unconverged=True,
            detach_unconverged=True,
            backprop=True,
            slew_rate_penalty=None,
            prev_ctrl=None,
            not_improved_lim=10,
            best_cost_eps=1e-4
    ):
        super().__init__()

        assert (u_lower is None) == (u_upper is None)
        assert max_linesearch_iter > 0

        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.T = T
        self.u_lower = u_lower
        self.u_upper = u_upper

        if not isinstance(u_lower, float):
            self.u_lower = detach_maybe(self.u_lower)

        if not isinstance(u_upper, float):
            self.u_upper = detach_maybe(self.u_upper)

        self.u_zero_I = detach_maybe(u_zero_I)
        self.u_init = detach_maybe(u_init)
        self.lqr_iter = lqr_iter
        self.grad_method = grad_method
        self.delta_u = delta_u
        self.verbose = verbose
        self.eps = eps
        self.back_eps = back_eps
        self.n_batch = n_batch
        self.linesearch_decay = linesearch_decay
        self.max_linesearch_iter = max_linesearch_iter
        self.exit_unconverged = exit_unconverged
        self.detach_unconverged = detach_unconverged
        self.backprop = backprop
        self.not_improved_lim = not_improved_lim
        self.best_cost_eps = best_cost_eps

        self.slew_rate_penalty = slew_rate_penalty
        self.prev_ctrl = prev_ctrl


    # @profile
    def forward(self, x_init, cost, dx):
        # QuadCost.C: [T, n_batch, n_tau, n_tau]
        # QuadCost.c: [T, n_batch, n_tau]
        
       
        assert isinstance(cost, QuadCost) or \
            isinstance(cost, Module) or isinstance(cost, Function)
            

        assert isinstance(dx, LinDx) or \
            isinstance(dx, Module) or isinstance(dx, Function)

        # TODO: Clean up inferences, expansions, and assumptions made here.
        if self.n_batch is not None:
            n_batch = self.n_batch
        elif isinstance(cost, QuadCost) and cost.C.ndimension() == 4:
            n_batch = cost.C.size(1)
        else:
            print('MPC Error: Could not infer batch size, pass in as n_batch')
            sys.exit(-1)


        # if c.ndimension() == 2:
        #     c = c.unsqueeze(1).expand(self.T, n_batch, -1)

        if isinstance(cost, QuadCost):
            C, c = cost
            if C.ndimension() == 2:
                # Add the time and batch dimensions.
                C = C.unsqueeze(0).unsqueeze(0).expand(
                    self.T, n_batch, self.n_state+self.n_ctrl, -1)
            elif C.ndimension() == 3:
                # Add the batch dimension.
                C = C.unsqueeze(1).expand(
                    self.T, n_batch, self.n_state+self.n_ctrl, -1)

            if c.ndimension() == 1:
                # Add the time and batch dimensions.
                c = c.unsqueeze(0).unsqueeze(0).expand(self.T, n_batch, -1)
            elif c.ndimension() == 2:
                # Add the batch dimension.
                c = c.unsqueeze(1).expand(self.T, n_batch, -1)

            if C.ndimension() != 4 or c.ndimension() != 3:
                print('MPC Error: Unexpected QuadCost shape.')
                sys.exit(-1)
            cost = QuadCost(C, c)

        assert x_init.ndimension() == 2 and x_init.size(0) == n_batch

        if self.u_init is None:
            u = torch.zeros(self.T, n_batch, self.n_ctrl).type_as(x_init.data)
        else:
            u = self.u_init
            if u.ndimension() == 2:
                u = u.unsqueeze(1).expand(self.T, n_batch, -1).clone()
        u = u.type_as(x_init.data)

        if self.verbose > 0:
            print('Initial mean(cost): {:.4e}'.format(
                torch.mean(get_cost(
                    self.T, u, cost, dx, x_init=x_init
                )).item()
            ))

        best = None

        n_not_improved = 0
        for i in range(self.lqr_iter):
            u = Variable(detach_maybe(u), requires_grad=True)
            # Linearize the dynamics around the current trajectory.
            x = get_traj(self.T, u, x_init=x_init, dynamics=dx)
            if isinstance(dx, LinDx):
                F, f = dx.F, dx.f
            else:
                F, f = self.linearize_dynamics(
                    x, detach_maybe(u), dx, diff=False)

            if isinstance(cost, QuadCost):
                C, c = cost.C, cost.c
            else:
                C, c, _ = self.approximate_cost(
                    x, detach_maybe(u), cost, diff=False)
            

            x, u, _lqr = self.solve_lqr_subproblem(
                x_init, C, c, F, f, cost, dx, x, u)
            
            
            back_out, for_out = _lqr.back_out, _lqr.for_out
            n_not_improved += 1

            assert x.ndimension() == 3
            assert u.ndimension() == 3

            if best is None:
                best = {
                    'x': list(torch.split(x, split_size_or_sections=1, dim=1)),
                    'u': list(torch.split(u, split_size_or_sections=1, dim=1)),
                    'costs': for_out.costs,
                    'full_du_norm': for_out.full_du_norm,
                }
            else:
                for j in range(n_batch):
                    if for_out.costs[j] <= best['costs'][j] + self.best_cost_eps:
                        n_not_improved = 0
                        best['x'][j] = x[:,j].unsqueeze(1)
                        best['u'][j] = u[:,j].unsqueeze(1)
                        best['costs'][j] = for_out.costs[j]
                        best['full_du_norm'][j] = for_out.full_du_norm[j]

            if self.verbose > 0:
                table_log('lqr', (
                    ('iter', i),
                    ('mean(cost)', torch.mean(best['costs']).item(), '{:.4e}'),
                    ('||full_du||_max', max(for_out.full_du_norm).item(), '{:.2e}'),
                    # ('||alpha_du||_max', max(for_out.alpha_du_norm), '{:.2e}'),
                    # TODO: alphas, total_qp_iters here is for the current
                    # iterate, not the best
                    ('mean(alphas)', for_out.mean_alphas.item(), '{:.2e}'),
                    ('total_qp_iters', back_out.n_total_qp_iter),
                ))

            if max(for_out.full_du_norm) < self.eps or \
               n_not_improved > self.not_improved_lim:
                break


        x = torch.cat(best['x'], dim=1)
        u = torch.cat(best['u'], dim=1)
        full_du_norm = best['full_du_norm']

        if isinstance(dx, LinDx):
            F, f = dx.F, dx.f
        else:
            F, f = self.linearize_dynamics(x, u, dx, diff=True)

        if isinstance(cost, QuadCost):
            C, c = cost.C, cost.c
        else:
            C, c, _ = self.approximate_cost(x, u, cost, diff=True)

        x, u, _ = self.solve_lqr_subproblem(
            x_init, C, c, F, f, cost, dx, x, u, no_op_forward=True)

        if self.detach_unconverged:
            if max(best['full_du_norm']) > self.eps:
                if self.exit_unconverged:
                    assert False

                if self.verbose >= 0:
                    print("LQR Warning: All examples did not converge to a fixed point.")
                    print("Detaching and *not* backpropping through the bad examples.")
                    
                    WARNING_INDEX   = 1
                    print(WARNING_INDEX)

                I = for_out.full_du_norm < self.eps
                Ix = Variable(I.unsqueeze(0).unsqueeze(2).expand_as(x)).type_as(x.data)
                Iu = Variable(I.unsqueeze(0).unsqueeze(2).expand_as(u)).type_as(u.data)
                x = x*Ix + x.clone().detach()*(1.-Ix)
                u = u*Iu + u.clone().detach()*(1.-Iu)

        costs = best['costs']
        return (x, u, costs)

    def solve_lqr_subproblem(self, x_init, C, c, F, f, cost, dynamics, x, u,
                             no_op_forward=False):
        if self.slew_rate_penalty is None or isinstance(cost, Module):
            _lqr = LQRStep(
                n_state=self.n_state,
                n_ctrl=self.n_ctrl,
                T=self.T,
                u_lower=self.u_lower,
                u_upper=self.u_upper,
                u_zero_I=self.u_zero_I,
                true_cost=cost,
                true_dynamics=dynamics,
                delta_u=self.delta_u,
                linesearch_decay=self.linesearch_decay,
                max_linesearch_iter=self.max_linesearch_iter,
                delta_space=True,
                current_x=x,
                current_u=u,
                back_eps=self.back_eps,
                no_op_forward=no_op_forward,
            )
            e = Variable(torch.Tensor())
            x, u = _lqr.forward(x_init, C, c, F, f if f is not None else e)

            return x, u, _lqr
        else:
            nsc = self.n_state + self.n_ctrl
            _n_state = nsc
            _nsc = _n_state + self.n_ctrl
            n_batch = C.size(1)
            _C = torch.zeros(self.T, n_batch, _nsc, _nsc).type_as(C)
            half_gamI = self.slew_rate_penalty*torch.eye(
                self.n_ctrl).unsqueeze(0).unsqueeze(0).repeat(self.T, n_batch, 1, 1)
            _C[:,:,:self.n_ctrl,:self.n_ctrl] = half_gamI
            _C[:,:,-self.n_ctrl:,:self.n_ctrl] = -half_gamI
            _C[:,:,:self.n_ctrl,-self.n_ctrl:] = -half_gamI
            _C[:,:,-self.n_ctrl:,-self.n_ctrl:] = half_gamI
            slew_C = _C.clone()
            _C = _C + torch.nn.ZeroPad2d((self.n_ctrl, 0, self.n_ctrl, 0))(C)

            _c = torch.cat((
                torch.zeros(self.T, n_batch, self.n_ctrl).type_as(c),c), 2)

            _F0 = torch.cat((
                torch.zeros(self.n_ctrl, self.n_state+self.n_ctrl),
                torch.eye(self.n_ctrl),
            ), 1).type_as(F).unsqueeze(0).unsqueeze(0).repeat(
                self.T-1, n_batch, 1, 1
            )
            _F1 = torch.cat((
                torch.zeros(
                    self.T-1, n_batch, self.n_state, self.n_ctrl
                ).type_as(F),F), 3)
            _F = torch.cat((_F0, _F1), 2)

            if f is not None:
                _f = torch.cat((
                    torch.zeros(self.T-1, n_batch, self.n_ctrl).type_as(f),f), 2)
            else:
                _f = Variable(torch.Tensor())

            u_data = detach_maybe(u)
            if self.prev_ctrl is not None:
                prev_u = self.prev_ctrl
                if prev_u.ndimension() == 1:
                    prev_u = prev_u.unsqueeze(0)
                if prev_u.ndimension() == 2:
                    prev_u = prev_u.unsqueeze(0)
                prev_u = prev_u.data
            else:
                prev_u = torch.zeros(1, n_batch, self.n_ctrl).type_as(u)
            utm1s = torch.cat((prev_u, u_data[:-1])).clone()
            _x = torch.cat((utm1s, x), 2)

            _x_init = torch.cat((Variable(prev_u[0]), x_init), 1)

            if not isinstance(dynamics, LinDx):
                _dynamics = CtrlPassthroughDynamics(dynamics)
            else:
                _dynamics = None

            if isinstance(cost, QuadCost):
                _true_cost = QuadCost(_C, _c)
            else:
                _true_cost = SlewRateCost(
                    cost, slew_C, self.n_state, self.n_ctrl
                )

            _lqr = LQRStep(
                n_state=_n_state,
                n_ctrl=self.n_ctrl,
                T=self.T,
                u_lower=self.u_lower,
                u_upper=self.u_upper,
                u_zero_I=self.u_zero_I,
                true_cost=_true_cost,
                true_dynamics=_dynamics,
                delta_u=self.delta_u,
                linesearch_decay=self.linesearch_decay,
                max_linesearch_iter=self.max_linesearch_iter,
                delta_space=True,
                current_x=_x,
                current_u=u,
                back_eps=self.back_eps,
                no_op_forward=no_op_forward,
            )
            x, u = _lqr(_x_init, _C, _c, _F, _f)
            x = x[:,:,self.n_ctrl:]

            return x, u, _lqr

    def approximate_cost(self, x, u, Cf, diff=True):
        with torch.enable_grad():
            tau = torch.cat((x, u), dim=2).data
            tau = Variable(tau, requires_grad=True)
            if self.slew_rate_penalty is not None:
                print("""
MPC Error: Using a non-convex cost with a slew rate penalty is not yet implemented.
The current implementation does not correctly do a line search.
More details: https://github.com/locuslab/mpc.pytorch/issues/12
""")
                sys.exit(-1)
                differences = tau[1:, :, -self.n_ctrl:] - tau[:-1, :, -self.n_ctrl:]
                slew_penalty = (self.slew_rate_penalty * differences.pow(2)).sum(-1)
            costs = list()
            hessians = list()
            grads = list()
            for t in range(self.T):
                tau_t = tau[t]
                if self.slew_rate_penalty is not None:
                    cost = Cf(tau_t) + (slew_penalty[t-1] if t > 0 else 0)
                else:
                    cost = Cf(tau_t)

                grad = torch.autograd.grad(cost.sum(), tau_t,
                                           retain_graph=True)[0]
                hessian = list()
                for v_i in range(tau.shape[2]):
                    hessian.append(
                        torch.autograd.grad(grad[:, v_i].sum(), tau_t,
                                            retain_graph=True)[0]
                    )
                hessian = torch.stack(hessian, dim=-1)
                costs.append(cost)
                grads.append(grad - bmv(hessian, tau_t))
                hessians.append(hessian)
            costs = torch.stack(costs, dim=0)
            grads = torch.stack(grads, dim=0)
            hessians = torch.stack(hessians, dim=0)
            if not diff:
                return hessians.data, grads.data, costs.data
            return hessians, grads, costs

    # @profile
    def linearize_dynamics(self, x, u, dynamics, diff):
        # TODO: Cleanup variable usage.

        n_batch = x[0].size(0)

        if self.grad_method == GradMethods.ANALYTIC:
            _u = Variable(u[:-1].view(-1, self.n_ctrl), requires_grad=True)
            _x = Variable(x[:-1].contiguous().view(-1, self.n_state),
                          requires_grad=True)

            # This inefficiently calls dynamics again, but is worth it because
            # we can efficiently compute grad_input for every time step at once.
            _new_x = dynamics(_x, _u)

            # This check is a little expensive and should only be done if
            # modifying this code.
            # assert torch.abs(_new_x.data - torch.cat(x[1:])).max() <= 1e-6

            if not diff:
                _new_x = _new_x.data
                _x = _x.data
                _u = _u.data

            R, S = dynamics.grad_input(_x, _u)

            f = _new_x - bmv(R, _x) - bmv(S, _u)
            f = f.view(self.T-1, n_batch, self.n_state)

            R = R.contiguous().view(self.T-1, n_batch, self.n_state, self.n_state)
            S = S.contiguous().view(self.T-1, n_batch, self.n_state, self.n_ctrl)
            F = torch.cat((R, S), 3)

            if not diff:
                F, f = list(map(Variable, [F, f]))
            return F, f
            
        else:
            # TODO: This is inefficient and confusing.
            x_init = x[0]
            x = [x_init]
            F, f = [], []
            for t in range(self.T):
                if t < self.T-1:
                    xt = Variable(x[t], requires_grad=True)
                    ut = Variable(u[t], requires_grad=True)
                    xut = torch.cat((xt, ut), 1)
                    new_x = dynamics.forward( xt, ut) # by lwh, add 

                    # Linear dynamics approximation.
                    if self.grad_method in [GradMethods.AUTO_DIFF,
                                             GradMethods.ANALYTIC_CHECK]:
                        Rt, St = [], []
                        for j in range(self.n_state):
                            Rj, Sj = torch.autograd.grad(
                                new_x[:,j].sum(), [xt, ut],
                                retain_graph=True)
                            if not diff:
                                Rj, Sj = Rj.data, Sj.data
                            Rt.append(Rj)
                            St.append(Sj)
                        Rt = torch.stack(Rt, dim=1)
                        St = torch.stack(St, dim=1)


                    elif self.grad_method == GradMethods.FINITE_DIFF:
                        Rt, St = [], []
                        for i in range(n_batch):
                            Ri = jacobian( 
                                lambda s: dynamics.forward( s, ut[i]), xt[i], 1e-4 #add by lwh
                            )
                            Si = jacobian(
                                lambda a : dynamics.forward( xt[i], a), ut[i], 1e-4 # add by lwh
                            )
                            if not diff:
                                Ri, Si = Ri.data, Si.data
                            Rt.append(Ri)
                            St.append(Si)
                        
                        
                        Rt = torch.stack(Rt)
                        St = torch.stack(St)
                        #print(Rt.shape,St.shape)
                        newRt = torch.zeros((1,13,13),dtype=torch.float32)
                        newSt = torch.zeros((1,13,2),dtype=torch.float32)
                        for i in range(13):
                            for j in range(13):
                                newRt[0, i,j] = Rt[0,0, j,i]
                            for j in range(2):
                                newSt[0, i,j] = St[0,0,j,i]
                        Rt = newRt
                        St = newSt
                        
                    else:
                        assert False

                    Ft = torch.cat((Rt, St), 2)
                    

                        
                    F.append(Ft)

                    if not diff:
                        xt, ut, new_x = xt.data, ut.data, new_x.data
                    
                    #print(Rt.shape, St.shape)
                    ft = new_x - bmv(Rt, xt) - bmv(St, ut)
                    f.append(ft)

                if t < self.T-1:
                    x.append(detach_maybe(new_x))

            F = torch.stack(F, 0)
            f = torch.stack(f, 0)
            if not diff:
                F, f = list(map(Variable, [F, f]))
            return F, f