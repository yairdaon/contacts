
import numpy as np

# Toy data
contact_matrix = np.array([[1, 2], [3, 4]])
I = np.array([10.0, 20.0])
population = np.array([100.0, 200.0])

# Define logI
logI = np.log(I)

# Expression as written
expr1 = contact_matrix @ np.exp(logI) / population

# Equivalent explicit-parentheses interpretation A:
# (contact_matrix @ np.exp(logI)) / population
expr2 = (contact_matrix @ np.exp(logI)) / population

# Alternative interpretation B (if @ and / were same precedence, L→R):
# contact_matrix @ (np.exp(logI) / population)
expr3 = contact_matrix @ (np.exp(logI) / population)

print("expr1:", expr1)
print("expr2 (matrix product, then divide):", expr2)
print("expr3 (elementwise divide before product):", expr3)

print("expr1 == expr2 ?", np.allclose(expr1, expr2))
print("expr1 == expr3 ?", np.allclose(expr1, expr3))
