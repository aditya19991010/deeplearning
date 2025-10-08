n = 32
p =1
f = 5
s=1

def calc_next_ldim(n,p,s,f):
    return ((n + 2*p -f) /s) +1

v = calc_next_ldim(n,p,s,f)
print(v)

