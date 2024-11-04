from itertools import chain


print(
    list(chain.from_iterable(zip(list1, list2)))
)