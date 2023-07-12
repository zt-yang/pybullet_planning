import h5py


def save_dict_to_h5(dict_data, filename):
    fh = h5py.File(filename, 'w')
    for k in dict_data:
        key_data = dict_data[k]
        if key_data is None:
            raise RuntimeError('data was not properly populated')
        # if type(key_data) is dict:
        #     key_data = json.dumps(key_data, sort_keys=True)
        try:
            fh.create_dataset(k, data=key_data)
        except TypeError as e:
            print("Failure on key", k)
            print(key_data)
            print(e)
            raise e
    fh.close()


def print_data_types(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key), type(key))
        if isinstance(value, dict):
            print_data_types(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value), type(value))