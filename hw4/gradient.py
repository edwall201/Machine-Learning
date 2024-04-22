def gradient_ascent(x, y, eta, iterations):
    for i in range(iterations):
        gradient_x = 1 - 400 * x * (x**2 + y**2 - 1)
        gradient_y = 1 - 400 * y * (x**2 + y**2 - 1)
        
        x += eta * gradient_x
        y += eta * gradient_y
    
    return x, y

x0 = 1
y0 = 1
eta = 0.0005
iterations = 1000

x_final, y_final = gradient_ascent(x0, y0, eta, iterations)
print("Final values after 1000 iterations:")
print(f"x = {x_final}, y = {y_final}")
print("final ans: ", x_final+y_final - 100*(x_final ** 2 + y_final**2 - 1)**2)
