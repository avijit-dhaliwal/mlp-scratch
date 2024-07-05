import struct

def load_mnist():
    train_images = read_idx_file('data/train-images-idx3-ubyte')
    train_labels = read_idx_file('data/train-labels-idx1-ubyte')
    test_images = read_idx_file('data/t10k-images-idx3-ubyte')
    test_labels = read_idx_file('data/t10k-labels-idx1-ubyte')
    
    return normalize(train_images), one_hot_encode(train_labels), normalize(test_images), one_hot_encode(test_labels)

def read_idx_file(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        return [struct.unpack('>B', f.read(1))[0] for _ in range(shape[0] * (shape[1] if len(shape) > 1 else 1))]

def normalize(images):
    return [[pixel / 255.0 for pixel in images[i:i+784]] for i in range(0, len(images), 784)]

def one_hot_encode(labels, num_classes=10):
    return [[1.0 if i == label else 0.0 for i in range(num_classes)] for label in labels]