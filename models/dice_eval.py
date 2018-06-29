# Imports
import numpy as np
import math
from collections import namedtuple


class Dice2007cjl(object):
    """This is a Python adoptation of the GAMS model available here:
    https://sites.google.com/site/openscienceletter/results/cjl----one-year/gams-code
    """

    def __init__(self):
        """Initialize object."""
        # SIMULATION SETTINGS #
        self.t_max = 600

        self.__set_exogenousvariables()
        self.reset()

    def __set_exogenousvariables(self):
        """Set all exogenous variables."""

        # parameters
        pop0 = 6514  # 2005 world population millions
        gpop0 = .035  # Growth rate of population per year
        popasym = 8600  # Asymptotic population
        a0 = .02722  # Initial level of total factor productivity
        ga0 = .0092  # Initial growth rate for technology per year
        dela = .001  # Decline rate of technol change per year
        # Emissions
        sig0 = .13418  # CO2-equivalent emissions-GNP ratio 2005
        gsigma = -.00730  # Initial growth of sigma per year
        dsig = .003  # Decline rate of decarbonization per year
        dsig2 = .000  # Quadratic term in decarbonization
        eland0 = 1.1000  # Carbon emissions from land 2005(GtC per year)
        # Preferences
        prstp = .015  # Initial rate of social time preference per year
        # Climate model
        fex0 = -.06  # Estimate of 2000 forcings of non-CO2 GHG
        fex1 = 0.30  # Estimate of 2100 forcings of non-CO2 GHG
        # Abatement cost
        expcost2 = 2.8  # Exponent of control cost function
        pback = 1.17  # Cost of backstop 2005 000$ per tC 2005
        backrat = 2  # Ratio initial to final backstop cost
        gback = .005  # Initial cost decline backstop pc per year

        # useful vectors for setting exogenous variables
        tvec = np.arange(self.t_max)
        onevec = np.ones(self.t_max)
        t_range = range(self.t_max-1)

        # EXOGENOUS VARIABLES #
        gpop = (np.exp(gpop0 * tvec)-1)/np.exp(gpop0 * tvec)
        self.L = pop0*(1-gpop) + gpop*popasym  # Population
        self.E_land = eland0*(1-0.01)**tvec  # Other emissions
        self.R = (1 + prstp)**(-tvec) # Discount rate
        self.F_ex = fex0 + .01*(fex1 - fex0)*tvec # External forcing
        self.F_ex[101:] = .30
        ga = ga0*np.exp(-dela*tvec)
        self.A = [a0]  # Total factor productivity
        [self.A.append(self.A[i]/(1-ga[i])) for i in t_range]
        gsig = gsigma*np.exp(-dsig*tvec - dsig2*tvec**2)
        sigma = [sig0] # ratio of (uncontrolled) ind. emissions to output
        [sigma.append(sigma[i]/(1-gsig[i+1])) for i in t_range]
        self.sigma = np.array(sigma)
        self.theta1 = (pback*self.sigma/(expcost2*backrat)
                       * (backrat-1+np.exp(-gback*tvec)))
        self.__default_sr = onevec*.22


    def reset(self):
        """Reset initial condition."""

        # INITIAL CONDITIONS #
        self.time = 0
        self.K = 137.  # 2005 value capital trill 2005 US dollars
        self.M_AT = 808.9  # Concentration in atmosphere 2005 (GtC)
        self.M_UP = 1255  # Concentration in upper strata 2005 (GtC)
        self.M_LO = 18365  # Concentration in lower strata 2005 (GtC)
        self.T_AT = .7307  # 2000 atmospheric temp change (C)from 1900
        self.T_LO = .0068  # 2000 lower strat. temp change (C) from 1900

        return self.observable_state()

    def step(self, mu=0, sr=None):
        """Move model forward."""
        # INPUT #
        # fixed default saving rate
        if sr is None:
            sr = .22

        # PARAMETERS #
        # Climate damage parameters calibrated for quadratic at 2.5 C for 2105
        aa1 = 0.00000  # Damage intercept
        aa2 = 0.0028388  # Damage quadratic term
        aa3 = 2.00  # Damage exponent
        # Temperature cycle
        c1 = .0220  # Climate-equation coefficient for upper level
        c3 = .300  # Transfer coeffic upper to lower stratum
        c4 = .0050  # Transfer coeffic for lower level
        fco22x = 3.8  # Estimated forcings of equilibrium co2 doubling
        # Carbon cycle
        b11 = 0.9810712  # Carbon cycle transition matrix
        b12 = 0.0189288  # Carbon cycle transition matrix
        b21 = 0.0097213  # Carbon cycle transition matrix
        b22 = 0.9852787  # Carbon cycle transition matrix
        b23 = 0.005  # Carbon cycle transition matrix
        b32 = 0.0003119  # Carbon cycle transition matrix
        b33 = 0.9996881  # Carbon cycle transition matrix
        # Other
        gama = .300  # Capital elasticity in production function
        theta2 = 2.8  # Exponent of control cost function
        dk = .100  # Depreciation rate on capital per year
        alpha = 2
        t2xco2 = 3  # Equilibrium temp impact of CO2 doubling oC
        lam = fco22x/t2xco2

        # For readability, define t
        t = self.time

        # MODEL EQUATION #

        # ECONOMY
        ygross = self.A[t]*self.L[t]**(1-gama)*self.K**gama  # gross output
        e = self.sigma[t]*(1-mu)*ygross + self.E_land[t]  # emissions
        Omega = 1/(1 + aa1*self.T_AT + aa2*self.T_AT**aa3)  # climate damages
        Lambda = self.theta1[t]*mu**theta2  # abatement cost
        y = ygross * Omega * (1-Lambda)  # net output
        i = sr*(y+.001)  # investment
        self.K = (1-dk)*self.K + i

        # CLIMATE
        # first calculate radiative forcing (based on old co2 concentration)
        f = fco22x*math.log(self.M_AT/596.4)/math.log(2) + self.F_ex[t]  # ra
        # co2 circulation
        self.M_AT, self.M_UP, self.M_LO = \
          tuple([b11*self.M_AT + b21*self.M_UP + e,
                 b12*self.M_AT + b22*self.M_UP + b32*self.M_LO,
                 b23*self.M_UP + b33*self.M_LO])
        # temperature circulation
        self.T_AT, self.T_LO = \
          tuple([self.T_AT + c1*(f-lam*self.T_AT-c3*(self.T_AT-self.T_LO)),
                 self.T_LO + c4*(self.T_AT - self.T_LO)])

        # UTILITY
        c = y - i
        u = self.L[t]*((c/self.L[t])**(1-alpha) - 1)/(1-alpha)
        reward = u * self.R[t]


        self.time+=1

        done = (self.time == self.t_max)
        return [self.observable_state(), np.array(reward), done]

    def observable_state(self):
        return (self.K, self.M_AT, self.M_UP, self.M_LO, self.T_AT, self.T_LO, self.time)
