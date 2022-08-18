--- 
 layout: post
 category: [py] 
 title: Python Loops 
 tags: [Python]
---

# Range Function

# For Loop

```
def sum_to(n):
    """ Return the sum of 1+2+3 ... n """
    ss  = 0
    for v in range(n+1):
        ss = ss + v
    return ss

```

# While Loops

```
def sum_to(n):
    """ Return the sum of 1+2+3 ... n """
    ss  = 0
    v = 1
    while v <= n:
        ss = ss + v
        v = v + 1
    return ss

# For your test suite
test(sum_to(4) == 10)
test(sum_to(1000) == 500500)

```