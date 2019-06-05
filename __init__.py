import numpy as np
import scipy as sp
import pmagpy.ipmag as ipmag
import pmagpy.pmag as pmag
import scipy.optimize as sciopt #import quadrature function
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, Layout

def log_sinh(k):
    if k>700:
        s=k-np.log(2.0)
    else:
        s=np.log(np.sinh(k))
    return s

def Chernoff(a,MU1,K1,MU2,K2):
    K12=np.linalg.norm(a*MU1*K1+(1-a)*MU2*K2)
    JF=a*(log_sinh(K1)-np.log(K1))+(1-a)*(log_sinh(K2)-np.log(K2))-(log_sinh(K12)-np.log(K12))
    return np.exp(-JF)

#optimize to find Bayes error
def Bayes(MU1,K1,MU2,K2):
    
    alpha0=sciopt.fminbound(lambda alpha: Chernoff(alpha,MU1,K1,MU2,K2),0,1)
    
    return Chernoff(alpha0,MU1,K1,MU2,K2)/2


def calculator(IA,DA,KA,RA,IB,DB,KB,RB):

    muA = pmag.dir2cart([DA,IA])
    muB = pmag.dir2cart([DB,IB])
    
    C = Chernoff(0.5,muA,KA*RA,muB,KB*RB)
    output1 = widgets.HTML(value='<h4>Bhattacharyya Coefficient = {0:.3f}</h4>'.format(C))

    B = Bayes(muA,KA*RA,muB,KB*RB)
    output2 = widgets.HTML(value='<h4>Bayes error = {0:.3f}</h4>'.format(B))
    
    if (RA==1) & (RB==1):
        Rtitle = widgets.HTML(value='<h3>Comparison of Fisher distributions:</h3>')
    else:
        Rtitle = widgets.HTML(value='<h3>Comparison of paleomagnetic poles:</h3>')

    results=widgets.VBox((Rtitle,output1,output2))
    display(results)

def open_console(*args):
    
    style = {'description_width': 'initial'} #general style settings
    layout={'width': '220px'}
    
    spacer = widgets.HTML(value='<font color="white">This is some text!</font>')

    Atitle = widgets.HTML(value='<h4>Distribution A</h4>')
    IA=widgets.BoundedFloatText(value=0.0,min=-90.0,max=90.0,step=0.01,description='Inc$_A$ [-90$^\circ$: +90$^\circ$]:',style=style,layout=layout)
    DA=widgets.BoundedFloatText(value=0.0,min=0.0,max=360.0,step=0.01,description='Dec$_A$ [0$^\circ$: 360$^\circ$]:',style=style,layout=layout)
    KA=widgets.BoundedFloatText(value=0.01,min=0.0,max=100000,step=0.01,description='$\kappa_A$ [>0]:',style=style,layout=layout)
    RA=widgets.BoundedFloatText(value=1,min=1,max=100000,step=0.01,description='$R_A$ [$\geq$1]:',style=style,layout=layout)

    Btitle = widgets.HTML(value='<h4>Distribution B</h4>')
    IB=widgets.BoundedFloatText(value=0.0,min=-90.0,max=90.0,step=0.01,description='Inc$_B$ [-90$^\circ$: +90$^\circ$]:',style=style,layout=layout)
    DB=widgets.BoundedFloatText(value=0.0,min=0.0,max=360.0,step=0.01,description='Dec$_B$ [0$^\circ$: 360$^\circ$]:',style=style,layout=layout)
    KB=widgets.BoundedFloatText(value=0.01,min=0.0,max=100000,step=0.01,description='$\kappa_B$ [>0]:',style=style,layout=layout)
    RB=widgets.BoundedFloatText(value=1,min=1.0,max=100000,step=0.01,description='$R_B$ [$\geq$1]:',style=style,layout=layout)
    
    uA = widgets.VBox((Atitle,IA, DA, KA, RA),layout=Layout(overflow_y='initial',height='180px'))
    uB = widgets.VBox((Btitle,IB, DB, KB, RB),layout=Layout(overflow_y='initial',height='180px'))
    uAB = widgets.HBox((uA,spacer,uB),layout=Layout(overflow_y='initial',height='180px'))
    ui = widgets.VBox([uAB],layout=Layout(overflow_y='initial',height='180px')) 

    out = widgets.interactive_output(calculator, {'IA': IA, 'DA': DA, 'KA': KA, 'RA': RA, 'IB': IB, 'DB': DB, 'KB': KB, 'RB': RB})
    display(ui,out)

