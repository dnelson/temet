"""Test and benchmark util.rotation."""

import numpy as np
import pytest

from temet.util import simParams as sim
from temet.util.rotation import ellipsoidfit


def _ellipsoidfit_synthetic(N=1000, h=0.1, R=10.0, q=1.0, s=1.0, phi=0.0, theta=0.0, noise_frac=0.0):
    """Generate random points in an ellipsoidal shell and test the iterative fitting algorithm."""
    rng = np.random.default_rng(424242)

    allphi = rng.uniform(0, 2 * np.pi, N)
    costheta = rng.uniform(-1.0, 1.0, N)
    alltheta = np.arccos(costheta)

    r = np.zeros(N)
    tot = 0

    while tot < N:
        u = rng.uniform()
        t = R * u ** (1.0 / 3.0)
        if t > (1 - h) * R:
            r[tot] = t
            tot += 1

    x = R * np.sin(alltheta) * np.cos(allphi)
    y = q * R * np.sin(alltheta) * np.sin(allphi)
    z = s * R * np.cos(alltheta)

    # rotate ellipsoid
    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)
    Rp = np.array([[np.cos(phi), -np.sin(phi), 0.0], [np.sin(phi), np.cos(phi), 0.0], [0.0, 0.0, 1.0]])
    Rt = np.array([[1.0, 0.0, 0.0], [0, np.cos(theta), -np.sin(theta)], [0.0, np.sin(theta), np.cos(theta)]])
    Rtot = Rt.dot(Rp)

    pos = np.vstack([x, y, z]).T
    pos = np.dot(Rtot, pos.T).T

    # add noise
    if noise_frac > 0:
        noise = rng.normal(loc=0.0, scale=noise_frac, size=pos.shape)
        pos += noise

    # fit
    rvir = 1.0
    mass = np.ones(N, dtype="float32")
    rin = 0.0
    rout = 1e10

    out = ellipsoidfit(pos, mass, rvir, rin, rout, weighted=False)

    # verify
    assert out[0] == pytest.approx(q, rel=1e-2)
    assert out[1] == pytest.approx(s, rel=2e-2)

    # print('q=',out[0],'expected=',q)
    # print('s=',out[1],'expected=',s)
    # print('R=',out[3])
    # print(Rtot)
    # rdot = [out[3][:,i].dot(Rtot[:,i]) for i in range(3)]
    # print(rdot)


def test_ellipsoidfit():
    """Execute some tests."""
    N = 5000

    _ellipsoidfit_synthetic(N, q=1.0, s=1.0)
    # _ellipsoidfit_synthetic(N,q=0.2,s=1.0,noise_frac=0.1)
    # _ellipsoidfit_synthetic(N,q=0.2,s=1.0,phi=45.0,theta=0.0,noise_frac=0.0)
    # _ellipsoidfit_synthetic(N,q=0.2,s=1.0,phi=45.0,theta=0.0,noise_frac=0.1)
    # _ellipsoidfit_synthetic(N,q=0.2,s=1.0,phi=30.0,theta=45.0,noise_frac=0.1)


@pytest.mark.requires_data
def test_ellipsoidfit_sim():
    """Find the shape (in stars, gas, or DM) of a set of galaxies/halos using the iterative ellipsoid method."""
    sP = sim(res=2160, run="tng", redshift=2.0)

    # field config
    ptType = "gas"  # 'gas'
    massField = "mass"  # 'sfr'

    # bin config
    solid = False  # otherwise shell
    binwidth = 0.3

    if 0:
        # generate a number of bins from the halo center to scalerad
        nbins = 10
        rmin = 1e-2  # linear, fractions of scalerad
        rmax = 1.0  # linear, fractions of scalerad

        logr = np.linspace(np.log10(rmin), np.log10(rmax), nbins)

        rin = 10 ** (logr - binwidth / 2.0)
        rout = 10 ** (logr + binwidth / 2.0)

    if 1:
        # or: use exact 'input' bin midpoints
        rbins = np.array([2.0])  # np.array([1.9,2.1]) # np.array([0.5,1.0,2.0,4.0])
        logr = np.log10(rbins)
        nbins = len(rbins)

        rin = rbins - binwidth / 2.0
        rout = rbins + binwidth / 2.0

    # select objects
    shIDs_z2_final25 = [139177]  # [29443,79350,60750,8069,57099,68178,110543,90627,55107,102285,113349,121252,
    # 125841,115247,115582,127580,132290,130665,129661,139177,145492,146306,154635,189521,246343] # fmt: skip

    for subhaloID in shIDs_z2_final25:
        # load a normalization/scale radius
        # scalerad = sP.groupCatSingle(haloID=groupID)['Group_R_Crit200']
        scalerad = sP.groupCatSingle(subhaloID=subhaloID)["SubhaloHalfmassRadType"][sP.ptNum("stars")]

        # load: halo properties and member-particles (scope is subhalo only)
        pos = sP.snapshotSubset(ptType, "pos_rel", subhaloID=subhaloID)
        pos /= scalerad
        mass = sP.snapshotSubset(ptType, massField, subhaloID=subhaloID)

        # for gas, filter to SFR>0 gas, and in general filter out all zero-weighted particles (for speed)
        if ptType == "gas":
            sfr = sP.snapshotSubset(ptType, "sfr", subhaloID=subhaloID)
            w = np.where(sfr > 0.0)
            mass = mass[w]
            pos = pos[w[0], :]

        w = np.where(mass > 0.0)
        mass = mass[w]
        pos = pos[w[0], :]

        # allocate
        q = np.zeros(nbins, dtype="float32")
        s = np.zeros(nbins, dtype="float32")
        n = np.zeros(nbins, dtype="int32")

        axes = np.zeros((nbins, 3, 3), dtype="float32")

        # fit (multiple radial shells)
        for i in range(nbins):
            if solid:
                ret = ellipsoidfit(pos, mass, scalerad, 0.0, 10 ** logr[i], weighted=True)
            else:
                ret = ellipsoidfit(pos, mass, scalerad, rin[i], rout[i])

            q[i], s[i], n[i], axes[i] = ret

            # print('[%6d %s] [r = %.2f] q = %.3f s = %.3f ratio = %.2f' % \
            # (subhaloID,ptType,10.0**logr[i],q[i],s[i],s[i]/q[i]))
            # print(subhaloID,i,q[i],s[i])

        assert q[0] == pytest.approx(0.41891116)
        assert s[0] == pytest.approx(0.05972898)
