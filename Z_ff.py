import numpy as np
import matplotlib.pyplot as plt

E = 0.31333
SW = 0.4747
CW = np.sqrt(1 - SW**2)
mW = 80.375
mZ = mW/CW
GF = E**2/(np.sqrt(2)*(2*SW*mW)**2)

def widthZff(mZ, mf, ca, cv):
    """
    Calculates the Breit-Wigner width for a Z fermion in the Fermilab Standard Model (FSM) scheme.
    Parameters:
    - mZ : float or array_like
        Mass of the Z boson in GeV. If an array is passed, it will return an array with the same shape containing the 
        Mass of the Z boson. If an array is given it must be one-dimensional and represent different masses.
    - mf : float or array_like
        Mass of the fermion. Must be scalar if mZ is scalar.
    - ca : float 
        weak coupling ca
    - cv : float
        strong coupling cv
    Returns:
    - Width of Z -> ff : float or array_like
    The returned value will have the same shape as mZ if mZ was an array.
    """
    #Checking input parameters
    assert type(ca)==float,"ca must be a float"
    assert type(cv)==float,"cv must be a float"
    #Converting inputs to arrays if needed
    mZ = np.array(mZ)
    mf = np.array(mf)
    #Calculating gamma using the formula from PDG
    
    xf = mf**2/mZ**2
    factor = GF/(6*np.sqrt(2)*np.pi)*np.sqrt(1-4*xf)
    Gamma = factor*(ca**2*(1 - 4*xf) + cv**2*(1 + 2*xf))
    return Gamma


weak_couplings = {
    'e':{'ca':-1/2, 'cv':-1/2 + 2*SW**2},
    'nu':{'ca':1/2, 'cv':1/2},
    'u':{'ca':1/2, 'cv':-1/2 + (4/3)*SW**2},
    'd':{'ca':-1/2, 'cv':-1/2 + (2/3)*SW**2}
}

mtau = 1.777
mnu = 0
mtop = 172.5
mb = 4.25

mZ_np = np.linspace(mZ, 200, 100)

plt.figure()
plt.plot(mZ_np, widthZff(mf=mnu, mZ=mZ_np, 
                         ca=weak_couplings['nu']['ca'], 
                         cv=weak_couplings['nu']['cv'])
                         , label=r'$\Gamma(Z \to \nu \nu)$')
plt.plot(mZ_np, widthZff(mf=mnu, mZ=mZ_np, 
                         ca=weak_couplings['e']['ca'], 
                         cv=weak_couplings['e']['cv']),
                         label=r'$\Gamma(Z \to e e)$')
plt.plot(mZ_np, widthZff(mf=mnu, mZ=mZ_np, 
                         ca=weak_couplings['u']['ca'], 
                         cv=weak_couplings['u']['cv']),
                         label=r'$\Gamma(Z \to uu)$')
plt.plot(mZ_np, widthZff(mf=mnu, mZ=mZ_np, 
                         ca=weak_couplings['d']['ca'], 
                         cv=weak_couplings['d']['cv']),
                         label=r'$\Gamma(Z \to d d)$')
plt.legend()
plt.show()
