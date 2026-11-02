import math

def my_log2(v: float) -> float:
    if v == 0.0:
        return 0.0
    return math.log2(v)

def binary_regret_precal(M: int) -> float:
    p = 10
    if M < 1:
        return 0.0

    s = 1.0
    b = 1.0
    bound = int(math.ceil(2.0 + math.sqrt(2.0 * M * p * math.log(10.0))))
    for i in range(1, bound + 1):
        b = (M - i + 1) * (b / M)
        s += b
    return s

def regret_precal(M: int, K: int) -> float:
    if K < 1:
        return 0.0
    elif K == 1:
        return 1.0
    else:
        s = binary_regret_precal(M)
        old_s = 1.0
        if K > 2:
            for j in range(3, K + 1):
                new_s = s + (M * old_s) / (float(j) - 2.0)
                old_s = s
                s = new_s
        return s

def regret(M: int, K: int) -> float:
    if K > 100:
        alpha = float(K) / float(M)
        ca = 0.5 + 0.5 * math.sqrt(1.0 + 4.0 / alpha)
        logReg = (
            float(M)
            * (math.log(alpha) + (alpha + 2.0) * math.log(ca) - 1.0 / ca)
            - 0.5 * math.log(ca + 2.0 / alpha)
        )
        return logReg / math.log(2.0)
    else:
        costs = regret_precal(M, K)
        if costs <= 0.0:
            return 0.0
        else:
            return my_log2(costs)
