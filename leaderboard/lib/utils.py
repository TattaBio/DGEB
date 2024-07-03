import random
import string


def rand_string(n):
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))
