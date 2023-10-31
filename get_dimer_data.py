import numpy as np

#You need to have dimer_dataXX.npz files that contain simulated holograms\
# (scattering patterns) of holograms. These will be used for training and validation.

all_data = []
for i in range(1, 12):
    index = str(i).zfill(2)
    all_data.append('dimer_data{}.npz'.format(index))


def stack(all_data):
    '''Stack data into one big .npy file'''

    arr_list = []
    for file in all_data:
        print('working on file' + str(file) + '\n')

        container = np.load(file, mmap_mode='r')
        for key in container:
            arr_list.append(container[key])

    arr = np.dstack(arr_list)
    print(arr.shape)


    return arr


def save_file(all_data):
    '''Save data as numpy arrays. Separate the labels (Y) from the features (X)'''

    res = stack(all_data)

    # X (features) should have shape of nx301x301, where n is number of dimer holograms
    X = res[:, :-1, :].T
    Y = res[:4, -1, :].T

    print(X.shape, Y.shape)

    np.save('X.npy', X)
    np.save('Y.npy', Y)

if __name__ == "__main__":
    save_file(all_data)

