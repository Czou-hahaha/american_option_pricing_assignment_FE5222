import time
from OurAmericanOptionPricing import Pricing

t, T, vol, rate, d_r, s, o_type, price = (0, 3, 0.2, 0.05, 0.05, 100, "Put", 80)
my_pricer = Pricing(n=10, m=5, l=24)
my_pricer.set_parameters(t, T, vol, rate, d_r, s, o_type, price)
tic = time.time()
print("price is", my_pricer.solver()[0, 0])
print("time used", time.time() - tic)
