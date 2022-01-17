#Katie Brown: Physics 3926 Project 3
#Simulating and investigating white dwarf masses and radii

#Import required packages
from scipy.integrate import solve_ivp
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as c

# Part 1 & 2

def odes(r, x):
    """
    Defines a system of coupled ODEs in rho (density) and m (mass), both with r (radius)
    as the dependent variable
    :param r: initial value for r (float)
    :param x: initial values for rho and m (2-element array)
    :return: ODEs for rho and m (2-element array)
    """
    # Assign rho and m to a vector element
    rho = x[0]
    m = x[1]
    # Define each ODE
    drho_dr = -3 * m * rho * np.sqrt(1 + rho ** (2 / 3)) / (r ** 2 * rho ** (2 / 3))
    dm_dr = r ** 2 * rho
    return [drho_dr, dm_dr]

def solve_system(odes, x0, method):
    """
    Uses solve_ivp to solve the system of ODEs given initial conditions,
    converts values to physical quantities
    :param odes: a function defining the ODEs
    :param x0: initial values for rho and m (2-element array)
    :param method: method of numerical integration (string)
    :return: a tuple containing the final values for mass and radius
    """
    # Define constants:
    mu_e = 2
    rho_0 = 9.74e5 * mu_e
    r_0 = (7.72e8 / mu_e)
    m_0 = (5.67e33 / mu_e ** 2)

    r_i = 1e-5  # Initial radius (in cm)
    r_i_dl = r_i / r_0  # Initial radius converted to dimensionless parameter
    r_f_dl = 10  # Final radius (dimensionless)

    rho_f = 1e-8  # Density of approximately zero (at which to end integration)
    rho_f_dl = rho_f / rho_0  # Final density converted to dimensionless

    rho_end = lambda r, x:x[0] - rho_f_dl
    rho_end.terminal = True
    sol = solve_ivp(odes, (r_i_dl, r_f_dl), x0, method,events=rho_end)  # Solve IVP and get arrays for r, rho & m

    sol_r = sol.t           #Extract radius solutions
    sol_rho = sol.y[0]      #Extract density solutions
    sol_m = sol.y[1]        #Extract mass solutions

    # Convert solutions to physical quantities
    rho_quantity = sol_rho * rho_0 * u.g / u.cm ** 3
    r_quantity = sol_r * r_0 * u.cm
    m_quantity = sol_m * m_0 * u.g

    # Take final mass and radii
    m_final = m_quantity[-1]
    r_final = r_quantity[-1]

    return (m_final, r_final)

# Initialize lists of final masses and radii for different center densities
m_finals = []
r_finals = []

# Solve system for 10 center frequencies:
rho_c_array = [0.1,0.5,1,5,10,100,1000,1e4,1e5,2.5e6]
for rho_c in rho_c_array:
    solution = solve_system(odes, [rho_c, 0], 'RK45')       # Solve system for each center density
    m_finals.append(solution[0]/u.g)                        # Need to remove units from m and r
    r_finals.append(solution[1]/u.cm)
plt.scatter(m_finals, r_finals,label='Simulated White Dwarfs') # Plot m vs r
plt.axvline(x=m_finals[-1],linestyle='--',
            label=f'Estimated Chandrasekhar limit at M= {m_finals[-1]:.2e} g') # Plot line at estimates Ch. limit
plt.xlabel("Mass (g)")
plt.ylabel("Radius (cm)")
plt.legend(loc='upper right')
plt.title("Radius vs Mass for Simulated White Dwarfs")
plt.show()

# Convert estimated Chandrasekhar Limit in terms of M_sun and mu_e:
ch_lim = m_finals[-1]*4/(c.M_sun).to(u.g)
print('Estimated Chandrasekhar Limit:',ch_lim,'/mu_e^2')

# PART 3:
# The default method is RK45, compare with solutions solved using RK23
for rho_c in [0.1,100,2.5e6]:
    sol_RK45 = solve_system(odes, [rho_c, 0], 'RK45')       # Solve system with same I.C.s with both methods
    sol_RK23 = solve_system(odes, [rho_c, 0], 'RK23')

    # Calculate and display the absolute and relative differences in r and m of the two methods:
    print("Radius Calculated with RK45:",sol_RK45[1])
    print("Radius Calculated with RK23:",sol_RK23[1])
    print("Difference in radius solutions for center frequency of", rho_c, ":", abs(sol_RK45[1]-sol_RK23[1]))
    print("This is a relative difference in radius of:", abs(sol_RK45[1]-sol_RK23[1])/sol_RK45[1])
    print("Mass Calculated with RK45:",sol_RK45[0])
    print("Mass Calculated with RK23:",sol_RK23[0])
    print("Difference in mass solutions for center frequency of", rho_c, ":", abs(sol_RK45[0]-sol_RK23[0]))
    print("This is a relative difference in mass of:", abs(sol_RK45[0]-sol_RK23[0])/sol_RK45[0])

# PART 4:
# Open data file wd_mass_radius.csv
with open("wd_mass_radius.csv") as data_file:
    lines = data_file.readlines()
    lines.pop(0)            # Remove title from data
    M_Msun = []
    M_Unc = []
    R_Rsun = []
    R_Unc = []
    for line in lines:
        line_split = line.split(',')    # Split each line with , as the delimiter
        M_Msun.append(float(line_split[0]))
        M_Unc.append(float(line_split[1]))
        R_Rsun.append(float(line_split[2]))
        R_Unc.append(float(line_split[3]))

M_data = (M_Msun * c.M_sun).to(u.g)     # Covert mass data to g from multiple of solar mass
M_Unc_Conv = (M_Unc * c.M_sun).to(u.g)
R_data = (R_Rsun * c.R_sun).to(u.cm)    # Covert radius data to cm from multiple of solar radius
R_Unc_Conv = (R_Unc * c.R_sun).to(u.cm)

# Need to remove units from data to plot
M_data = M_data/u.g
M_Unc_Conv = M_Unc_Conv/u.g
R_data = R_data/u.cm
R_Unc_Conv = R_Unc_Conv/u.cm

# Plot white dwarf data with simulated points from part 2
plt.scatter(m_finals, r_finals,label='Simulated White Dwarfs')
plt.scatter(M_data,R_data,label='White Dwarf Data')
plt.errorbar(M_data,R_data,R_Unc_Conv,M_Unc_Conv, ls='none',c='r')
plt.xlabel("Mass (g)")
plt.ylabel("Radius (cm)")
plt.legend(loc='upper right')
plt.title("Simulated and Recorded White Dwarfs")
plt.show()


