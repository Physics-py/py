# #Python code to perform least square fitting for equations of the forms y = bx+a,y = ax^b and y = ae^(bx).
# import numpy as np 
# def lsqf():
#         # Asking user for the total number of elements, ensuring it's greater than 2
#     while True:
#         try:
#             print("---------------------------------------")
#             n = int(input("Enter the total number of elements : "))
#             if n>2:
#                 break # Break out of the loop if input is valid
#             else:
#                 print("Please! Enter a number greater than 2")
#         except ValueError:
#             print("Error Invalid input.")
#     try:
#         # Prompting user to choose the type of equation
#         print("---------------------------------------------------------")
#         print("Type 1 : y = bx+a\nType 2 : y = ax^b\nType 3 : y = ae^(bx) ")
#         print("---------------------------------------------------------")
#         while True:
#             eq_type = int(input("Choose the type of equation(1,2 or 3) : "))
#             if eq_type in [1,2,3]:
#                 break # Break out of the loop if input is valid
#             else:
#                 print("Please! Enter a valid input. ") 

#         # Getting input elements for x and y
#         print("---------------------------------------------------------")
#         print("Enter the elements of x .")
#         print("---------------------------------------------------------")
#         array1 = [float(input(f"Enter the element {i+1} of x ")) for i in range(n)]
#         print("---------------------------------------------------------")
#         print("Enter the elements of y .")
#         print("---------------------------------------------------------")
#         array2 = [float(input(f"Enter the element {i+1} of y ")) for i in range(n)]
#         print("---------------------------------------------------------")

#         # Preparing log arrays for further computation
#         array3 = np.log(array1)
#         array4 = np.log(array2)

#         # Computing sigma values
#         sigma_x = np.sum(array1)
#         sigma_y = np.sum(array2)
#         sigma_lx = np.sum(array3)
#         sigma_ly = np.sum(array4)
#         sigma_xy = np.sum(np.multiply(array1,array2))
#         sigma_xsq = np.sum(np.square(array1))
#         sigma_lxsq = np.sum(np.square(array3))
#         sigma_lxly = np.sum(np.multiply(array3,array4))
#         sigma_xly = np.sum(np.multiply(array1,array4))

#         # Performing calculations based on the chosen equation type
#         if eq_type ==1:
#             a = (sigma_xsq*sigma_y-sigma_x*sigma_xy)/(n*sigma_xsq-sigma_x**2)
#             b = (n*sigma_xy-sigma_x*sigma_y)/(n*sigma_xsq-sigma_x**2)
#             return a,b
#         elif eq_type ==2:
#             a = np.exp((sigma_lxsq*sigma_ly-sigma_lx*sigma_lxly)/(n*sigma_lxsq-sigma_lx**2))   
#             b = (n*sigma_lxly-sigma_lx*sigma_ly)/(n*sigma_lxsq-sigma_lx**2)         
#             return a,b
#         elif eq_type==3:
#             a = np.exp((sigma_xsq*sigma_ly-sigma_x*sigma_xly)/(n*sigma_xsq-sigma_x**2))
#             b = (n*sigma_xly-sigma_x*sigma_ly)/(n*sigma_xsq-sigma_x**2)  
#             return a,b
#         else:
#             print("Error! Invalid input.")
#             return None,None
#     except ValueError:
#         print("Error! Please enter a valid integer. Try Again!!!")

# # Calling the function and printing the results if they are valid
# result = lsqf()
# if result is not None:
#     a,b = result
#     print(a,b)
# print("---------------------------------------")
# print("This code is written by Ram(2023PHY1034)")
# print("---------------------------------------")

###################################################################################################
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

n_values = [0,1,2,3,4]
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
# p0_values = legendre_polynomials[0]
# p1_values = legendre_polynomials[1]
# p2_values = legendre_polynomials[2]
# p3_values = legendre_polynomials[3]
# p4_values = legendre_polynomials[4]

# pd0_values = legendre_derivatives[0]
# pd1_values = legendre_derivatives[1]
# pd2_values = legendre_derivatives[2]

data_p_values = np.column_stack((x, p_values[0], p_values[1], p_values[2]))
np.savetxt('leg00.dat', data_p_values, header='x P0(x) P1(x) P2(x)')
# np.savetxt('leg00.dat', data_p_values, header='x P0(x) P1(x) P2(x)', fmt='%12.8f') #using fmt for formatting

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
