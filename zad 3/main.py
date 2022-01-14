from data_generation import generate_data

if __name__ == "__main__":
    data_0 = generate_data((0, 0), 2, 200)

    data_1_1 = generate_data((-3, 0), 1, 100)
    data_1_2 = generate_data((3, 0), 1, 100)

    print(data_0)
