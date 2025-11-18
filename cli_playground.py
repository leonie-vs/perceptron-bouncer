from main import model as perceptron

print("Put 0 for No and 1 for Yes")
shirt = int(input("Are they wearing a 'I <3 JavaScript' T-shirt?"))
duck = int(input("Are they carrying a menacing rubber duck?"))
pennies = int(input("Are they trying to pay with pennies?"))
queue = int(input("Are they jumpping the queue?"))

input_vector = [[shirt, duck, pennies, queue]]
decision = perceptron.predict(input_vector)
print("✅ Welcome in!" if decision[0] else "🚫 They're on the list. The *bad* list.")