import numpy as np
def main():
    print("Hello from irras!")


if __name__ == "__main__":
    main()


a1 = np.array([1, 2, 3, 4, 5])
a2 = np.array([6, 7, 8, 9, 10])

a = np.concatenate([a1, a2])
print(a)