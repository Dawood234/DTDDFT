import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import re

mpl.rcParams['figure.dpi']=100

basis_list='def2-SVP,def2-TZVP,def2-TZVPP,cc-pVDZ,cc-pVTZ'.split(',')
basis = basis_list[3]  # cc-pVDZ  
func = 'pbe96'
framework = 'tddft'  # Overall theoretical framework: 'tddft' or 'tda'

print(f"Loading data for: {framework}_{basis}_{func}")

def load_vector_data(data_file):
    """Load all vector data from a standardized data file"""
    if not os.path.exists(data_file):
        print(f"Warning: {data_file} not found")
        return {}
    
    data_dict = {}
    with open(data_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if '=' in line and 'np.array' in line and not line.strip().startswith('#'):
                try:
                    var_name = line.split('=')[0].strip()
                    # Extract array from the line
                    array_str = line.split('np.array(')[1].split(')')[0]
                    array_data = eval(array_str)
                    data_dict[var_name] = np.array(array_data)
                except Exception as e:
                    print(f"Warning: Could not parse line: {line.strip()}: {e}")
                    continue
    
    return data_dict

# Load data from standardized files
folder_name = f"{framework}_{basis}_{func}"

# Load vector data
data_file = f"../data_{folder_name}.txt"
vector_data = load_vector_data(data_file)

# Load matrix data - matrices are already extracted by scripts to apb_* and sqrtamb_* files
# For now, we'll note that matrix files exist and can be loaded by scripts if needed
apb_file = f"../apb_{folder_name}"
sqrtamb_file = f"../sqrtamb_{folder_name}"

# Check if matrix files exist
apb_exists = os.path.exists(apb_file)
sqrtamb_exists = os.path.exists(sqrtamb_file)

# Note: Matrix data is handled by the TDDFT calculation scripts directly
# The apb_* and sqrtamb_* files are already in the correct format
apb = None  # Will be loaded by calculation scripts as needed
sqrtamb = None  # Will be loaded by calculation scripts as needed

# BLA array 
bla = np.array([0.125466, 0.110464, 0.0954646, 0.0803673, 0.0652275, 0.0500952, 0.0349265, 0.0198214, 0.00452051,
                -0.0107809, -0.0138626, -0.0200338, -0.0230615, -0.0262975, -0.0295307, -0.0327402, -0.0359374, 
                -0.0390515, -0.0423162, -0.0457886, -0.0490522, -0.0523145, -0.0556133, -0.062, -0.075, -0.088, 
                -0.099, -0.111, -0.122, -0.134, -0.144])

# Extract vectors from loaded data with flexible naming
def get_vector_data(data_dict, possible_names):
    """Get vector data by trying multiple possible names"""
    for name in possible_names:
        if name in data_dict:
            return data_dict[name]
    print(f"Warning: None of {possible_names} found in data")
    return None

def load_matrix_data(filename):
    """Load matrix data from APB/sqrt(A-B) files"""
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found")
        return None
        
    with open(filename, 'r') as file:
        text = file.read()
    
    if not text or text.strip() == "":
        return None
        
    sections = re.split(r'\n-{3,}\n|\n-{3,}$', text, flags=re.MULTILINE) 
    matrices_text = sections[1:]  
    num_list = []
    
    for i in range(len(matrices_text)):
        if i % 2 != 0 and i != 0:
            secc = sections[i].split('\n')
            if len(secc) >= 5:
                try:
                    line1 = secc[3]
                    line2 = secc[4]
                    numline1 = line1.split()
                    numline2 = line2.split()
                    
                    if len(numline1) >= 4 and len(numline2) >= 4:
                        num = np.array([[float(numline1[2]), float(numline1[3])], 
                                      [float(numline2[2]), float(numline2[3])]])
                        num_list.append(num)
                except (IndexError, ValueError) as e:
                    print(f"Warning: Could not parse matrix section {i}: {e}")
                    continue
    
    return np.array(num_list) if num_list else None

# Extract all vector data
gs = get_vector_data(vector_data, [f'gs_{folder_name}', 'gs'])
mo14_vec = get_vector_data(vector_data, ['mo14'])
mo15_vec = get_vector_data(vector_data, ['mo15'])  
mo16_vec = get_vector_data(vector_data, ['mo16'])
mo17_vec = get_vector_data(vector_data, ['mo17'])
ag_tddft_vec = get_vector_data(vector_data, [f'ag_{framework}', 'ag'])
bu_tddft_vec = get_vector_data(vector_data, [f'bu_{framework}', 'bu'])
Hq1d_vec = get_vector_data(vector_data, ['Hq1d'])
Hq2d_vec = get_vector_data(vector_data, ['Hq2d'])

# Matrix data handled separately - files available at apb_* and sqrtamb_*
apb_vec = None  # Available in apb_file
sqrtamb_vec = None  # Available in sqrtamb_file

# Convert units
au_to_ev = 27.2114
print(f"Configuration: {basis}, {func}, {framework}")

if ag_tddft_vec is not None:
    ag_tddft_vec = ag_tddft_vec / au_to_ev
if bu_tddft_vec is not None:
    bu_tddft_vec = bu_tddft_vec / au_to_ev

# Create basis_fun dictionary for compatibility with existing code (matrices handled separately)
basis_fun = {
    f"{basis}_{func}": [gs, ag_tddft_vec, bu_tddft_vec, mo14_vec, mo15_vec, 
                        mo16_vec, mo17_vec, apb_file, sqrtamb_file, Hq1d_vec, Hq2d_vec]
}

# Unpack for compatibility with existing analysis code
if f"{basis}_{func}" in basis_fun:
    gs, ag_tddft_vec, bu_tddft_vec, mo14_vec, mo15_vec, mo16_vec, mo17_vec, apb_file_path, sqrtamb_file_path, Hq1d_vec, Hq2d_vec = basis_fun[f"{basis}_{func}"]

print(f"Data loaded successfully!")
print(f"Available vectors: gs={gs is not None}, ag={ag_tddft_vec is not None}, bu={bu_tddft_vec is not None}")
print(f"                   mo14-17={all(x is not None for x in [mo14_vec, mo15_vec, mo16_vec, mo17_vec])}")
print(f"                   Hq1d/2d={all(x is not None for x in [Hq1d_vec, Hq2d_vec])}")
print(f"                   matrix files: apb={apb_exists}, sqrtamb={sqrtamb_exists}")

class TDDFTCalculation:
    # Class variables to cache loaded matrix data
    _apb_data = None
    _sqrtamb_data = None
    
    def __init__(self, index):
        self.index = index
        self.om_tddft = ag_tddft_vec[index]
        self.om_d=2*bu_tddft_vec[index]
        self.mo14 = mo14_vec[index]
        self.mo15 = mo15_vec[index]
        self.mo16 = mo16_vec[index]
        self.mo17 = mo17_vec[index]
        self.Hq1d = -2 / np.sqrt(2) * Hq1d_vec[index]
        self.Hq2d = 2 / np.sqrt(2) * Hq2d_vec[index]
        self.Hdq1 = self.Hq1d
        self.Hdq2 = self.Hq2d

    def calculate_nu(self):
        nu1 = self.mo16 - self.mo14
        nu2 = self.mo17 - self.mo15
        nud = 2 * (self.mo16 - self.mo15)
        return nu1, nu2, nud

    @classmethod
    def _load_matrix_data_if_needed(cls):
        """Load matrix data from files if not already loaded"""
        if cls._apb_data is None:
            cls._apb_data = load_matrix_data(apb_file_path)
        if cls._sqrtamb_data is None:
            if framework.lower() == 'tda':
                cls._sqrtamb_data = cls._apb_data  # For TDA, sqrt(A-B) = A+B just as placeholder as it is not needed
            else:
                cls._sqrtamb_data = load_matrix_data(sqrtamb_file_path)

    def get_matrices(self):
        self._load_matrix_data_if_needed()
        if self._apb_data is not None and self.index < len(self._apb_data):
            apb = self._apb_data[self.index]
        else:
            apb = None
            
        if self._sqrtamb_data is not None and self.index < len(self._sqrtamb_data):
            sqrtamb = self._sqrtamb_data[self.index]
        else:
            sqrtamb = None
            
        return apb, sqrtamb

    def get_results(self):
        nu1, nu2, nud = self.calculate_nu()
        apb, sqrtamb = self.get_matrices()
        return self.om_tddft,self.om_d, nu1, nu2, nud, sqrtamb, apb, self.Hdq1, self.Hq1d, self.Hdq2, self.Hq2d
    

def dress( om,  nu1, nu2, om_d,Hq1d, Hq2d, Hdq1, Hdq2):
    res=np.array([[Hq1d*Hdq1/(4*np.sqrt(nu1*nu1))*(1+(nu1+om_d)*(nu1+om_d)/(om**2-(om_d**2))),(Hq1d*Hdq2)/(4*np.sqrt(nu1*nu2))*(1+(nu1+om_d)*(nu2+om_d)/(om**2-(om_d**2)))],
                  
                [(Hq2d*Hdq1)/(4*np.sqrt(nu2*nu1))*(1+(nu2+om_d)*(nu1+om_d)/(om**2-(om_d**2))),(Hq2d*Hdq2)/(4*np.sqrt(nu2*nu2))*(1+(nu2+om_d)*(nu2+om_d)/(om**2-(om_d**2)))] ])
    return res


def renormalized_f1f2(e1, e2, F1, F2, Hq1d, Hq2d, Hdq1, Hdq2, nu1, nu2, om_d):
    omega1_squared = e1**2
    omega2_squared = e2**2
    beta = np.array([[Hq1d * Hq1d, Hq1d * Hq2d],[Hq2d * Hdq1 , Hdq2 * Hdq2]])
    C = np.array([[( nu1 + om_d ) * ( nu1 + om_d ) , ( nu1 + om_d ) * ( nu2 + om_d ) ],[(nu2+om_d)*(nu1+om_d),(nu2+om_d)*(nu2+om_d)]])

    D = np.array([[om_d**2,om_d**2],[om_d**2,om_d**2]]) 
        
    numerator=beta*C
    denominator_omega1 = (omega1_squared - D)**2
    result_matrix_omega_1 = -numerator / denominator_omega1
    denominator_omega2 = (omega2_squared - D)**2
    result_matrix_omega_2 = -numerator / denominator_omega2
    delta1=np.diag((1,1)) - result_matrix_omega_1
    delta2=np.diag((1,1)) - result_matrix_omega_2
    renormalization_factor_1 = np.sqrt((F1.T)@(delta1@F1))
    renormalization_factor_2 = np.sqrt((F2.T)@(delta2@F2))
    F1=F1 / renormalization_factor_1
    F2=F2 / renormalization_factor_2
    return np.linalg.norm(F1)**2,np.linalg.norm(F2)**2

def DTTDFT(om_tddft,sqrtamb,apb,Hq1d, Hq2d, Hdq1, Hdq2, nu1, nu2, om_d):
    apb_dressed=apb+2*dress(om_tddft,nu1,nu2,om_d,Hq1d,Hq2d,Hdq1,Hdq2)
    C=(sqrtamb)@(apb_dressed)@sqrtamb
    E,F = np.linalg.eig(C)
    sorted_id = np.argsort(E)
    sorted_e=E[sorted_id]
    sorted_F = F[:, sorted_id]
    F1,F2=sorted_F[:,0],sorted_F[:,1]
    f2s=renormalized_f1f2(np.sqrt(sorted_e[0]),np.sqrt(sorted_e[1]),F1,F2,Hq1d, Hq2d, Hdq1, Hdq2, nu1, nu2, om_d)
    return f2s,np.sqrt(sorted_e)*au_to_ev

def mazur_dtddft(om_tddft,sqrtamb,apb,Hq1d, Hq2d, Hdq1, Hdq2, nu1, nu2, om_d):
    oms=np.array([om_tddft,0])
    def compute_matrix(om):
        apb_dressed=apb+2*dress(om_tddft,nu1,nu2,om_d,Hq1d,Hq2d,Hdq1,Hdq2)
        C=(sqrtamb)@(apb_dressed)@sqrtamb
        return C
    def diagonalize(C):
        E,F = np.linalg.eig(C)
        sorted_id = np.argsort(E)
        sorted_e=np.sqrt(E[sorted_id])
        sorted_F = F[:, sorted_id]
        return sorted_e,sorted_F
    
    converged = False
    previous_om = oms[0]
    tolerance = 1e-6  
    iterations = 0
    while not converged:
        C= compute_matrix(oms[0])
        oms,Fs = diagonalize(C)
        if abs(oms[0]- previous_om) < tolerance:
            converged = True
            # print('converged in {} iterations'.format(iterations))
        else:
            previous_om= oms[0]
        iterations += 1
    F1,F2=Fs[:,0],Fs[:,1]
    f2s=renormalized_f1f2(oms[0],oms[1],F1,F2,Hq1d, Hq2d, Hdq1, Hdq2, nu1, nu2, om_d)
    return f2s,oms*au_to_ev



#DTDA

def X_dress_bob(om, om_d,Hq1d, Hq2d, Hdq1, Hdq2):
    Hq1d, Hdq1, Hq2d, Hdq2 = Hq1d, Hdq1, Hq2d, Hdq2
    dress = np.array([
        [Hq1d*Hdq1/(om-om_d), (Hq1d*Hdq2)/(om-om_d)],
        [(Hq2d*Hdq1)/(om-om_d), (Hq2d*Hdq2)/(om-om_d)]
    ])
    return dress
    
def DTDA(om_tddft,apb,Hq1d, Hq2d, Hdq1, Hdq2,om_d,nu1, nu2):
    aa_dressed=apb+X_dress_bob( om_tddft, om_d,Hq1d, Hq2d, Hdq1, Hdq2)
    E,F = np.linalg.eig(aa_dressed)
    sorted_id = np.argsort(E)
    sorted_e=E[sorted_id]
    sorted_F = F[:, sorted_id]
    F1,F2=sorted_F[:,0],sorted_F[:,1]
    f2s=renormalized_f1f2(sorted_e[0],sorted_e[1],F1,F2,Hq1d, Hq2d, Hdq1, Hdq2, nu1, nu2, om_d)
    return f2s,sorted_e*au_to_ev

    
def mazur_DTDA(om_tddft,apb,Hq1d, Hq2d, Hdq1, Hdq2,nu1,nu2,om_d):
    oms=np.array([om_tddft,0]) 
    
    def compute_matrix(om):
        aa_dressed=apb+X_dress_bob(om_tddft, om_d,Hq1d, Hq2d, Hdq1, Hdq2)
        return aa_dressed
    
    def diagonalize(C):
        E,F = np.linalg.eig(C)
        sorted_id = np.argsort(E)
        sorted_e=E[sorted_id]
        sorted_F = F[:, sorted_id]
        return sorted_e,sorted_F


    converged = False
    previous_om = oms[0]
    tolerance = 1e-6  
    iterations = 0

    while not converged:
        A = compute_matrix(oms[0])
        oms,Fs = diagonalize(A)
        iterations += 1
        if abs(oms[0]- previous_om) < tolerance:
            converged = True
            # print("Converged in {} iterations".format(iterations))
        else:
            previous_om= oms[0]
    
    F1,F2=Fs[:,0],Fs[:,1]

    f2s=renormalized_f1f2(oms[0],oms[1],F1,F2,Hq1d, Hq2d, Hdq1, Hdq2, nu1, nu2, om_d)
    return f2s,oms*au_to_ev


#run plot and compare with ref

file_path='../ex_data.txt'
data = pd.read_csv(file_path, sep=r'\s+', comment='#')
if len(data.columns) >= 3:
    data.columns = ['x', 'bu', 'ag']
bla_ex= data['x'].to_numpy()
bu_delta= data['bu'].to_numpy()
ag_delta= data['ag'].to_numpy()

def plot_dtddft(algorithm,ag,bu,gs):
    f12,f22=[],[]
    e1,e2=[],[]
    nu1_vec,nu2_vec, nud_vec=[],[],[]
    om_d_vec=[]
    for i in range(len(bla)):
        data=TDDFTCalculation(i)
        om_tddft,om_d,nu1,nu2,nud,sqrtamb,apb,Hdq1,Hq1d,Hdq2,Hq2d=data.get_results()
        fs,es=algorithm(om_tddft,sqrtamb,apb,Hq1d, Hq2d, Hdq1, Hdq2, nu1, nu2, om_d)
        f12.append(fs[0])
        f22.append(fs[1])
        e1.append(es[0])
        e2.append(es[1])
        nu1_vec.append(nu1)
        nu2_vec.append(nu2)
        nud_vec.append(nud)
        om_d_vec.append(om_d)
    f12,f22,e1,e2=np.array(f12),np.array(f22),np.array(e1),np.array(e2)
    plt.figure(figsize=(10, 6))
    ag_shifted=ag+gs-gs[0]
    bu_shifted=bu+gs-gs[0]
    e11=e1+(gs-gs[0])*au_to_ev
    e22=e2+(gs-gs[0])*au_to_ev

    # plt.plot(np.concatenate((bla[:2],bla[6:])),np.concatenate((e11[:2],e11[6:])),label='DTDDFT#1',color='b',marker='+',markevery=3,markersize=8)
    plt.plot(bla, e11,  label='DTDDFT#1',color='b',marker='+',markevery=3,markersize=8)
    
    plt.plot(bla_ex,bu_delta,marker='o',markerfacecolor='None',label='Bu(Ref.)',color='k',markevery=4)
    plt.plot(bla_ex,ag_delta,label='Ag(Ref.)',marker='o',color='k',markevery=4)
    plt.plot(bla,ag_shifted*au_to_ev,label='Ag(ATDDFT)',color='b',marker='^',markevery=4,markerfacecolor='b',linestyle='dashed')
    plt.plot(bla,bu_shifted*au_to_ev,label='Bu(ATDDFT)',color='b',marker='^',markevery=4,markerfacecolor='None',linestyle='dashed')
    plt.gca().invert_xaxis()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.minorticks_on()
    plt.grid(True)
    plt.legend(fontsize=13,loc=(0.03,0.01))
    plt.xlabel(r'BLA(${\AA}$)',fontsize=15)
    plt.ylabel('E (eV)',fontsize=15)
    plt.title(f'{algorithm.__name__ if hasattr(algorithm, "__name__") else str(algorithm)}: {basis}, {func}')
    plt.show()

    return e11, e22, f12, f22, ag_shifted, bu_shifted,nu1_vec,nu2_vec,nud_vec
if framework=='tddft':
    e11, e22, f12, f22, ag_shifted, bu_shifted,nu1_vec,nu2_vec,nud_vec=plot_dtddft(mazur_dtddft,ag_tddft_vec,bu_tddft_vec,gs)
    # plt.plot(bla,f12)
    # plt.gca().invert_xaxis()
    # plt.ylabel(r'$\vert G_1\vert ^2$',fontsize=15)
    # plt.xlabel(r'BLA(${\AA}$)',fontsize=15)
    # # plt.xlim(0.130,-.10)
    # plt.show()