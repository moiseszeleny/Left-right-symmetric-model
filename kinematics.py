#-*- coding: utf-8 -*-
from sympy import latex, S, Add, Matrix, sqrt, Mul, Symbol, Basic, Number, NumberSymbol
from sympy import Eq, Expr, sqrt, symbols, cos, sin, I, exp
from sympy import atan, acos, Interval, solveset

from sympy.physics.matrices import mgamma, msigma
from sympy.physics.quantum.dagger import Dagger



class V3D(Basic):
    _diff_wrt = True
    th = symbols(r'\theta', positive=True)
    phi = symbols(r'\phi', positive=True)

    def __init__(self, px=0, py=0, pz=0):
        self.px = px
        self.py = py
        self.pz = pz
        
    
    def __str__(self):
        return '({a},{b},{c})'.format(a=latex(self.px),\
                                    b=latex(self.py), c=latex(self.pz))
    def __repr__(self):
        return self.__str__()
    
    def __pos__(self):
        return self
    
    def __neg__(self):
        return V3D(S.NegativeOne*self.px,\
                  S.NegativeOne*self.py,S.NegativeOne*self.pz)
    
    def __add__(self,other):
        px, py, pz = self.px, self.py, self.pz
        kx, ky, kz = other.px, other.py, other.pz
        return V3D(Add(px, kx), Add(py, ky), Add(pz, kz))
    
    def __sub__(self,other):
        px, py, pz = self.px, self.py, self.pz
        kx, ky, kz = other.px, other.py, other.pz
        return V3D( Add(px,-kx), Add(py,-ky), Add(pz,-kz))
    
    def __abs__(self):
        px, py, pz = self.px, self.py, self.pz
        return sqrt(px**2 + py**2 + pz**2)
    
    def polarform(self, th=th, phi=phi):
        p_abs = abs(self)
        px = p_abs*sin(th)*cos(phi)
        py = p_abs*sin(th)*sin(phi)
        pz = p_abs*cos(th)
        return V3D(px, py, pz)
    
    def simplify(self):
        px = self.px.simplify()
        py = self.py.simplify()
        pz = self.pz.simplify()
        return V3D(px, py, pz)

class FV(Basic):
    _diff_wrt = True
    th = V3D.th
    phi = V3D.phi

    def __init__(self,p0=0,p1=0,p2=0,p3=0):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

    def __str__(self):
        return '({a},{b},{c},{d})'.format(a=latex(self.p0), b=latex(self.p1),\
                                           c=latex(self.p2), d=latex(self.p3))
    
    def __repr__(self):
        return self.__str__()
    
    def __pos__(self):
        return self
    
    def __neg__(self):
        return FV(S.NegativeOne*self.p0,S.NegativeOne*self.p1,\
                  S.NegativeOne*self.p2,S.NegativeOne*self.p3)
    
    
    def __add__(self,other):
        p0,p1,p2,p3 = self.p0,self.p1,self.p2,self.p3
        k0,k1,k2,k3 = other.p0,other.p1,other.p2,other.p3
        return FV(Add(p0,k0), Add(p1,k1), Add(p2,k2), Add(p3,k3))
    
    def __sub__(self,other):
        p0,p1,p2,p3 = self.p0,self.p1,self.p2,self.p3
        k0,k1,k2,k3 = other.p0,other.p1,other.p2,other.p3
        return FV(Add(p0,-k0), Add(p1,-k1), Add(p2,-k2), Add(p3,-k3))
    
    def __mul__(self,other):
        p0,p1,p2,p3 = self.p0,self.p1,self.p2,self.p3
        
        if isinstance(other,(int,float, Symbol, Number, NumberSymbol)):
            return FV(
                p0*other, p1*other,
                p2*other, p3*other)
        if isinstance(other,FV):
            k0,k1,k2,k3 = other.p0,other.p1,other.p2,other.p3
            return Add(Mul(p0,k0),-Mul(p1,k1),-Mul(p2,k2),-Mul(p3,k3))
    
    def __rmul__(self,other):
        p0,p1,p2,p3 = self.p0,self.p1,self.p2,self.p3
        
        if isinstance(other,(int,float, Symbol, Number, NumberSymbol)):
            return FV(
                other*p0, other*p1,
                other*p2, other*p3)
        if isinstance(other,FV):
            k0,k1,k2,k3 = other.p0,other.p1,other.p2,other.p3
            return Add(Mul(k0, p0),-Mul(k1, p1),-Mul(k2, p2),-Mul(k3, p3))
    
    def __div__(self,other):
        p0,p1,p2,p3 = self.p0,self.p1,self.p2,self.p3
        
        if isinstance(other,(int,float, Symbol, Number, NumberSymbol)):
            return FV(
                p0/other, p1/other,
                p2/other, p3/other)

    def __matmul__(self,other):
        if isinstance(other,FV):
            p0,p1,p2,p3 = self.p0,self.p1,self.p2,self.p3
            k0,k1,k2,k3 = other.p0,other.p1,other.p2,other.p3
            A = Matrix([[p0],[p1],[p2],[p3]])
            B = Matrix([[k0],[k1],[k2],[k3]])
            return A*B.transpose()
        if isinstance(other,[int,float]):
            return self*other
    
    def __abs__(self):
        p0,p1,p2,p3 = self.p0,self.p1,self.p2,self.p3
        return sqrt(p0**2 - p1**2 - p2**2 - p3**2)
    
    def __eq__(self,other):
        return self.p0 == other.p0 and self.p1 == other.p1 and self.p2 == other.p2 and self.p3 == other.p3
    
    def __neq__(self,other):
        return not self.__eq__(other)
    
    def slash(self):
        return self.p0*mgamma(0) - self.p1*mgamma(1) - self.p2*mgamma(2)- self.p3*mgamma(3)
    
    def subs(self,cambios):
        return FV(self.p0.subs(cambios),self.p1.subs(cambios),self.p2.subs(cambios),self.p3.subs(cambios))
    
    def matrixform(self):
        return Matrix(
            [
                [self.p0],
                [self.p1],
                [self.p2],
                [self.p3]
            ]
        )
    
    #def polarform(self, th=th, phi=phi):
    #    p_abs = abs(self)
    #    px = p_abs*sin(th)*cos(phi)
    #    py = p_abs*sin(th)*sin(phi)
    #    pz = p_abs*cos(th)
    #    return V3D(px, py, pz)

def E_rel(pmu,m):
    if isinstance(pmu,FV):
        return sqrt(pmu.p1**2 + pmu.p2**2 + pmu.p3**2  + m**2)
    else:
        print(f'{pmu} debe ser una instancia de FV')
    
    
    
def eqs_conservation(pi,pf):
    
    if isinstance(pi,FV) and isinstance(pf,FV):
        return Eq(pi.p0,pf.p0), (Eq(pi.p1,pf.p1),Eq(pi.p2,pf.p2),Eq(pi.p3,pf.p3))
    
    if isinstance(pi,FV) and isinstance(pf,list):
        return Eq(pi.p0,Add(*[pf[i].p0 for i in range(len(pf))])), \
        (
        Eq(pi.p1,Add(*[pf[i].p1 for i in range(len(pf))])),
        Eq(pi.p2,Add(*[pf[i].p2 for i in range(len(pf))])),
        Eq(pi.p3,Add(*[pf[i].p3 for i in range(len(pf))]))
        )
    
    if isinstance(pi,list) and isinstance(pf,FV):
        return Eq(Add(*[pi[i].p0 for i in range(len(pi))]),pf.p0), \
        (
        Eq(Add(*[pi[i].p1 for i in range(len(pi))]),pf.p1),
        Eq(Add(*[pi[i].p2 for i in range(len(pi))]),pf.p2),
        Eq(Add(*[pi[i].p3 for i in range(len(pi))]),pf.p3)
        )
    
    if isinstance(pi,list) and isinstance(pf,list):
        return Eq(Add(*[pi[i].p0 for i in range(len(pi))]),Add(*[pf[i].p0 for i in range(len(pf))])),\
        (
        Eq(Add(*[pi[i].p1 for i in range(len(pi))]),Add(*[pf[i].p1 for i in range(len(pf))])),
        Eq(Add(*[pi[i].p2 for i in range(len(pi))]),Add(*[pf[i].p2 for i in range(len(pf))])),
        Eq(Add(*[pi[i].p3 for i in range(len(pi))]),Add(*[pf[i].p3 for i in range(len(pf))]))
        )


class Spinor(Expr):
    def __init__(self, pmu, m, h):
        self.pmu = pmu
        self.m = m
        self.h = h
    
    
    def p_sigma(self):
        pmu = self.pmu
        m = self.m
        sp = (
            pmu.p1*msigma(1) + 
            pmu.p2*msigma(2)+ 
            pmu.p3*msigma(3)
            )/(pmu.p0 + m)
        return sp
    
    def p_sigma_bar(self):
        sp = self.p_sigma()
        return -sp
        
    def adj(self):
        return Dagger(self.matrixform())*mgamma(0)
    
    

class SpinorU(Spinor):
    def __init__(self, pmu, m, h):
        super().__init__(pmu, m, h)
    
    def Xi(self):
        pmu = self.pmu
        helicidad = self.h
        p1 = pmu.p1
        p2 = pmu.p2
        p3 = pmu.p3
        p = V3D(p1, p2, p3)
        r = abs(p).simplify()
        if p1 !=0:
            phi = atan(p2/p1).simplify()
        else:
            phi=0
        
        th = acos(p3/r).simplify()

        if helicidad == 1:
            out = Matrix(
                [
                    [cos(th/2)],
                    [sin(th/2)*exp(I*phi)]
                ]
            )
        elif helicidad == -1:
            out = Matrix(
                [
                    [-sin(th/2)*exp(-I*phi)],
                    [cos(th/2)]
                ]
            )
        else:
            raise ValueError('The argument helicidad must be 1 or -1.')
        return out

    def matrixform(self):
        pmu = self.pmu
        m = self.m
        h = self.h
        if isinstance(pmu,FV):
            E = pmu.p0
        else:
            raise ValueError(f'{pmu} debe ser un cuadrivector instancia de FV()')
        if h in [1, -1]:
            xi = self.Xi()
            u = Matrix( # giunti definition
            [
                [sqrt(E + m)*xi[0]],
                [sqrt(E + m)*xi[1]],
                [h*sqrt(E - m)*xi[0]],
                [h*sqrt(E - m)*xi[1]]
            ]
        )
        else:
            raise ValueError('h must be 1 or -1')
        return u

#### Definiendo spinores u(p), v(p) y su adjunto-
class SpinorV(Spinor):
    def __init__(self, pmu, m, h):
        super().__init__(pmu, m, h)

    def Xi(self):
        pmu = self.pmu
        helicidad = -self.h
        p1 = pmu.p1
        p2 = pmu.p2
        p3 = pmu.p3
        p = V3D(p1, p2, p3)
        r = abs(p).simplify()
        if p1 !=0:
            phi = atan(p2/p1).simplify()
        else:
            phi=0
        
        th = acos(p3/r).simplify()

        if helicidad == 1:
            out = Matrix(
                [
                    [cos(th/2)],
                    [sin(th/2)*exp(I*phi)]
                ]
            )
        elif helicidad == -1:
            out = Matrix(
                [
                    [-sin(th/2)*exp(-I*phi)],
                    [cos(th/2)]
                ]
            )
        else:
            raise ValueError('The argument helicidad must be 1 or -1.')
        return out
    
    def matrixform(self):
        pmu = self.pmu
        m = self.m
        h = self.h
        E = pmu.p0
        
        if h in (1, -1):
            xi = self.Xi()
            v = Matrix(
                [
                    [-sqrt(E - m)*xi[0]],
                    [-sqrt(E - m)*xi[1]],
                    [h*sqrt(E + m)*xi[0]],
                    [h*sqrt(E + m)*xi[1]]
                ]
            )
        else:
            raise ValueError('h must be 1 or -1')
        return v
    
def Lagrangiano_invariante(L,sim):
    '''
    Esta función toma un lagramgiano L y regresa el lagrangiano invatiante ante la simetría sim.
    sim: es un diccionario de la forma {x:Tx,...}, donde T denota la transformación que se aplicará al campo x.
    '''
    L = L.expand()
    terminos_iniciales = L.args
    terminos_transformado = L.subs(sim).args
    return Add(*[term for term in terminos_iniciales if term in terminos_transformado])


