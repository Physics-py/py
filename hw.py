###############################################--Sem2_MP2_L01--###############################################
'''
# The current I and the voltage V bears the relation IR = V .
# Estimate the value of R and error ∆R, from an experiment that produced the following
# data for voltage vs current.

import numpy as np
# Given data
voltage = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])  # Voltage (V)
current = np.array([0.0, 0.8, 1.5, 2.1, 2.6, 3.0, 3.3, 3.6, 3.9, 4.0])   # Current (mA)

# Convert current from milliamperes (mA) to amperes (A)
current_A = current / 1000.0  # Convert milliamperes to amperes

# Calculate the means
mean_v = np.mean(voltage)
mean_i = np.mean(current_A)

# Calculate the deviations from the mean
delta_v = voltage - mean_v
delta_i = current_A - mean_i

# Calculate the slope (resistance) using the least squares method
slope = np.sum(delta_v * delta_i) / np.sum(delta_i**2)

# Calculate the intercept (not needed for resistance calculation)
intercept = mean_v - slope * mean_i

# Calculate the error in the slope (resistance) using the least squares method
delta_R = np.sqrt(np.sum((delta_v - slope * delta_i)**2) / (len(voltage) - 2)) / np.sqrt(np.sum(delta_i**2))

# Resistance (R) is the slope of the line (V/I)
R = slope

# Print the results
print("Estimated Resistance (R):", R, "ohms")
print("Error in Resistance (ΔR):", delta_R, "ohms")

#Python code to perform least square fitting for equations of the forms y = bx+a,y = ax^b and y = ae^(bx).
import numpy as np 
def lsqf():
        # Asking user for the total number of elements, ensuring it's greater than 2
    while True:
        try:
            print("---------------------------------------")
            n = int(input("Enter the total number of elements : "))
            if n>2:
                break # Break out of the loop if input is valid
            else:
                print("Please! Enter a number greater than 2")
        except ValueError:
            print("Error Invalid input.")
    try:
        # Prompting user to choose the type of equation
        print("---------------------------------------------------------")
        print("Type 1 : y = bx+a\nType 2 : y = ax^b\nType 3 : y = ae^(bx) ")
        print("---------------------------------------------------------")
        while True:
            eq_type = int(input("Choose the type of equation(1,2 or 3) : "))
            if eq_type in [1,2,3]:
                break # Break out of the loop if input is valid
            else:
                print("Please! Enter a valid input. ") 

        # Getting input elements for x and y
        print("---------------------------------------------------------")
        print("Enter the elements of x .")
        print("---------------------------------------------------------")
        array1 = [float(input(f"Enter the element {i+1} of x ")) for i in range(n)]
        print("---------------------------------------------------------")
        print("Enter the elements of y .")
        print("---------------------------------------------------------")
        array2 = [float(input(f"Enter the element {i+1} of y ")) for i in range(n)]
        print("---------------------------------------------------------")

        # Preparing log arrays for further computation
        array3 = np.log(array1)
        array4 = np.log(array2)

        # Computing sigma values
        sigma_x = np.sum(array1)
        sigma_y = np.sum(array2)
        sigma_lx = np.sum(array3)
        sigma_ly = np.sum(array4)
        sigma_xy = np.sum(np.multiply(array1,array2))
        sigma_xsq = np.sum(np.square(array1))
        sigma_lxsq = np.sum(np.square(array3))
        sigma_lxly = np.sum(np.multiply(array3,array4))
        sigma_xly = np.sum(np.multiply(array1,array4))

        # Performing calculations based on the chosen equation type
        if eq_type ==1:
            a = (sigma_xsq*sigma_y-sigma_x*sigma_xy)/(n*sigma_xsq-sigma_x**2)
            b = (n*sigma_xy-sigma_x*sigma_y)/(n*sigma_xsq-sigma_x**2)
            return a,b
        elif eq_type ==2:
            a = np.exp((sigma_lxsq*sigma_ly-sigma_lx*sigma_lxly)/(n*sigma_lxsq-sigma_lx**2))   
            b = (n*sigma_lxly-sigma_lx*sigma_ly)/(n*sigma_lxsq-sigma_lx**2)         
            return a,b
        elif eq_type==3:
            a = np.exp((sigma_xsq*sigma_ly-sigma_x*sigma_xly)/(n*sigma_xsq-sigma_x**2))
            b = (n*sigma_xly-sigma_x*sigma_ly)/(n*sigma_xsq-sigma_x**2)  
            return a,b
        else:
            print("Error! Invalid input.")
            return None,None
    except ValueError:
        print("Error! Please enter a valid integer. Try Again!!!")

# Calling the function and printing the results if they are valid
result = lsqf()
if result is not None:
    a,b = result
    print(a,b)
'''
###############################################--Sem2_MP2_L03--###############################################

'''
#Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre

#defining factorial function
def fact(a):
    if a == 0:
        return 1
    else:
        return a * fact(a - 1) 

#calculating legendre polynomials
def legendre_my(n_values):
    x = np.linspace(-1, 1, 101)
    legendre_polynomials = []
    legendre_derivatives = []
    for n in n_values:
        if n % 2 == 0:
            m = n // 2
        else:
            m = (n - 1) // 2
        legendre_polynomial = np.zeros_like(x)
        legendre_derivative = np.zeros_like(x)
        for i in range(m + 1):

            legendre_polynomial += ((((-1) ** i) * fact(2 * n - 2 * i))/((2**n)*fact(n - i) *fact(n - 2 * i) * fact(i))) * (x **(n - 2 * i))
            if n-2*i ==0:
                legendre_derivative+=0
            else:
                legendre_derivative += ((((-1) ** i) * fact(2 * n - 2 * i))/((2**n)*fact(n - i)*fact(n-2*i)*fact(i))) * (n - 2 * i) * (x ** (n - 2 * i - 1))
            
        legendre_polynomials.append(legendre_polynomial)
        legendre_derivatives.append(legendre_derivative)
    return legendre_polynomials, legendre_derivatives

n_values = [0,1,2,3,4,5]
legendre_polynomials, legendre_derivatives = legendre_my(n_values)
# print(legendre_polynomials)
print(legendre_derivatives)

x = np.linspace(-1, 1, 101)

#storing the values of P0,P1,P2,... and P'0,P'1,P'2
p_values = []
for i in range(len(n_values)):
    p_values.append(legendre_polynomials[i])

pd_values = []
for i in range(len(n_values)):
    pd_values.append(legendre_derivatives[i])


data_p_values = np.column_stack((x, p_values[0], p_values[1], p_values[2]))
np.savetxt('leg00.dat', data_p_values, header='x P0(x) P1(x) P2(x)')

data_pd_values = np.column_stack((x, pd_values[0], pd_values[1], pd_values[2]))
np.savetxt('leg01.dat', data_pd_values, header='x Pd0(x) Pd1(x) Pd2(x)')


# #Using inbuilt fuction to calculate legendre polynomials using scipy

def compute_legendre_values(a,x):
    legendre_inbuilt = legendre(a)
    return legendre_inbuilt(x)
legendre_computed = [compute_legendre_values(a,x) for a in n_values]

#comparing inbuilt and custom legendre fuction values

comparison = np.allclose(legendre_computed, legendre_polynomials)
print("Values from both methods match:", comparison)

# Read the contents of the file
with open('leg00.dat', 'r') as file:
    file_contents_1 = file.read()
with open('leg01.dat', 'r') as file:
    file_contents_2 = file.read()

# Print the contents to the console
# print(file_contents_1)
# print(file_contents_2)

#Prove the recursion relation for Legendre polynomial : 

#retrieving all the required values used in all three recursion relations
pd1x = np.loadtxt('leg01.dat',skiprows =1, usecols = 2)
pd2x = np.loadtxt('leg01.dat',skiprows=1, usecols=3)
p0x = np.loadtxt('leg00.dat',skiprows=1,usecols = 1)
p1x = np.loadtxt('leg00.dat',skiprows = 1, usecols = 2)
p2x = np.loadtxt('leg00.dat',skiprows = 1, usecols = 3)

# (1) nPn(x) = xP′n(x) − P′n−1(x) for n=2

n = 2
n_hundred = np.tile(n,len(x))
lhs1 = n*p_values[2] 
rhs1 = x*pd2x - pd1x
compare_recursion1 = np.allclose(lhs1,rhs1)
print("The recursion relation for Legendre polynomial: nPn(x) = xP′n(x) − P′n−1(x) is ",compare_recursion1)

#storing the values in leg02.dat file
data_recursion1 = np.column_stack((x, n_hundred, n_hundred-1, pd2x, pd2x, pd1x))
np.savetxt('leg02.dat', data_recursion1, header='x  n  (n - 1)   P2(x)   Pd2(x)   Pd1(x)', fmt='%12.6f')


#(2) (2n +1)xPn(x) = (n+1)Pn+1 +nPn−1(x) for n =2
lhs2 = (2*n +1)*x*p2x
rhs2 = (n+1)*p_values[3] + n*p1x
compare_recursion2 = np.allclose(lhs2,rhs2)
print("The recursion relation for Legendre polynomial: (2n +1)xPn(x) = (n+1)Pn+1 +nPn−1(x) is ",compare_recursion2)

#storing the values in leg03.dat file  x, n, (n − 1), (n + 1), Pn(x), Pn−1(x) and Pn+1(x)
data_recursion2 = np.column_stack((x, n_hundred, n_hundred-1, n_hundred+1, p2x, p_values[3]))
np.savetxt('leg03.dat', data_recursion2, header='x   n   n-1   n+1   p2x   p3_values', fmt='%12.6f')

#(3) nPn(x) = (2n−1)xPn−1(x)−(n−1)Pn−2(x) for n=3
n=3
n_hundred = np.tile(n,len(x))
lhs3 = n*p_values[3]
rhs3 = (2*n-1)*x*p2x - (n-1)*p1x
compare_recursion3 = np.allclose(lhs3,rhs3)
print("The recursion relation for Legendre polynomial: nPn(x) = (2n−1)xPn−1(x)−(n−1)Pn−2(x) is ",compare_recursion3)

#storing the values in leg03.dat file   x, n, (n − 1), (n + 1), Pn(x), Pn−1(x) and Pn+1(x)
data_recursion3 = np.column_stack((x, n_hundred, n_hundred-1, n_hundred+1, p_values[3], p_values[4]))
np.savetxt('leg04.dat', data_recursion3, header='x   n   n-1   n+1   p2x   p3_values', fmt='%12.6f')

# -------orthogonality-------
def integrate(m, n, num_points, initial, final):
    x = np.linspace(initial, final, num_points)
    step_size = (final - initial) / (num_points - 1)
    legendre_polynomial_m = legendre_my([m])[0][0]
    legendre_polynomial_n = legendre_my([n])[0][0]
    
    integrand = legendre_polynomial_m * legendre_polynomial_n
    integral = np.sum(integrand) * step_size
    return integral

# Calculate orthogonality matrix
num_polynomials = len(x)
orthogonality_matrix = np.zeros((num_polynomials, num_polynomials))

for i in range(num_polynomials):
    for j in range(num_polynomials):
        orthogonality_matrix[i, j] = integrate(i, j, len(x), -1, 1)

# Display orthogonality matrix
print("Orthogonality Matrix:")
print(orthogonality_matrix)

#plotting the legendre polynomial for n values.
plt.figure(figsize=(10, 6))
for n, legendre_poly in zip(n_values, legendre_polynomials):
    plt.plot(x, legendre_poly, label=f'n={n}')
plt.title("Legendre Polynomials")
plt.xlabel("x")
plt.ylabel("P_n(x)")
plt.legend()
plt.grid(True)
plt.show()
'''
###############################################--Sem2_MP2_L04--###############################################
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import lagrange

xi = [0.00, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
yi = [1.0, 0.99, 0.96, 0.91, 0.85, 0.76, 0.67, 0.57, 0.46, 0.34, 0.22, 0.11, 0.00, -0.10, -0.18, -0.26]
# xi = [2.81, 3.24, 3.80 ,4.30, 4.37, 5.29, 6.03]
# yi = [0.5,1.2, 2.1, 2.9, 3.6, 4.5, 5.7]

n = len(xi)

def lagrange_basis(x_values, x, j):
    result = 1
    for i in range(n):
        if i != j:
            result *= (x - x_values[i]) / (x_values[j] - x_values[i])
    return result 

def lagrange_interpolation(x_values, y_values, x):
    result = 0
    for j in range(n):
        result += y_values[j] * lagrange_basis(x_values, x, j)
    return result

def lagrange_inverse_interpolation(x_values, y_values, y):
    return lagrange_interpolation(y_values, x_values, y)  

# Find out the value of the Bessel function at β = 2.3.
# Also find out the value of β for which the Bessel function J0(β) = 0.5

x = 2.3
print("the value of the Bessel function at β = 2.3 is interpolated value at x=2.3:", lagrange_interpolation(xi, yi, x))

y = 0.5
print("β for which the Bessel function J0(β) = 0.5 is interpolated value at y=0.5:", lagrange_inverse_interpolation(xi, yi, y))

# Find out the value of incident laser intensity if the detected photodetector voltage is 2.4
# y = 2.4
# print("the value of incident laser intensity if the detected photodetector voltage is 2.4 is interpolated value at y=2.4:", lagrange_inverse_interpolation(xi, yi, y) )

# Perform inbuilt Lagrange interpolation
lagrange_poly = lagrange(xi, yi)

x_value = 1.6
interpolated_value = lagrange_poly(x_value)
print("Interpolated value at x={}: {}".format(x_value, interpolated_value))

# Generate points for plotting the curve
x_range = np.linspace(min(xi), max(xi), 1000)
y_range_custom = [lagrange_interpolation(xi, yi, x) for x in x_range]
y_range_inbuilt = lagrange_poly(x_range)

# Plotting the data points and the interpolation curve
plt.scatter(xi, yi, color='blue', label='Data Points')
plt.plot(x_range, y_range_custom, color='red', label='Custom Lagrange Interpolation')
plt.plot(x_range, y_range_inbuilt, color='green', label='Inbuilt Lagrange Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of Lagrange Interpolation')
plt.legend()
plt.grid(True)
plt.show()
'''
# print('''----------------------------------------------------
# What did you learn ?
# ----------------------------------------------------
# Interpolation is a method to estimate values between known data points. Lagrange interpolation uses polynomials that pass through each data point exactly, allowing for accurate approximation of values within the data range. Through exploring interpolation techniques like Lagrange interpolation, I've learned how to effectively estimate unknown values based on available data, which is crucial in various fields such as mathematics, engineering, and data analysis.''')

# # print("---------------------------------------")
# # print("These codes are written by Ram(2023PHY1034)")
# # print("---------------------------------------")





