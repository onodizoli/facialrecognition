"""
Code is based on:
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
[4] Materials of Deep Learning and Neural Networks by prof. Aurel Lazar, Columbai University
"""
import numpy
import os
import theano
import theano.tensor as T


from loader import load_faceScrub, load_test
from layers import LogisticRegression, HiddenLayer, ConvPoolLayer, train_net


def test_conv(learning_rate=0.1, n_epochs=1000, nkerns=[16, 512], kern_shape=[9,7],
            batch_size=200, verbose=False, loadmodel=False):
    """
    learning_rate: term for the gradient 

    n_epochs: maximal number of epochs before exiting

    nkerns: number of kernels on each layer
    
    kern_shape: list of numbers with the dimensions of the kernels

    batch_szie: number of examples in minibatch.

    verbose: to print out epoch summary or not to
    
    loadmodel: load parameters from a saved .npy file

    """
    
    # Folder for saving and loading parameters
    folder='results'
    # Seed the random generator
    rng = numpy.random.RandomState(1990)

    # Load the dataset
    datasets = load_faceScrub(theano_shared=True)
    
    # Functions for saving and loading parameters
    def save(folder):
        for param in params:
            print (str(param.name))
            numpy.save(os.path.join(folder,
                       param.name + '.npy'), param.get_value())

    def load(folder):
        for param in params:
            param.set_value(numpy.load(os.path.join(folder,
                            param.name + '.npy')))


    # Accassing the train,test and validation set
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ###############
    # BUILD MODEL #
    ###############
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 1 * 100 * 100)
    # to a 4D tensor, which is expected by theano
    layer0_input = x.reshape((batch_size, 1, 100, 100))
    
    # First convolutional pooling layer
    layer0 = ConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=((batch_size, 1, 100, 100)), 
        filter_shape=((nkerns[0], 1, kern_shape[0], kern_shape[0])),
        poolsize=((2,2)),
        idx=0
    )

    # Second layer
    layer1 = ConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=((batch_size, nkerns[0], 46, 46)),
        filter_shape=((nkerns[1], nkerns[0], kern_shape[1], kern_shape[1])),
        poolsize=((2,2)),
        idx=1
    )
    
    # Flatten input for the fully connected layer
    layer2_input = layer1.output.flatten(2)

    # Fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=(nkerns[1]*20*20),
        n_out=(500),
        activation=T.tanh
    )   
    

    # Output layer
    layer3 = LogisticRegression(
         input=layer2.output,
         n_in=500,
    n_out=530)

    # Cost function
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # Calculate validation error
    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # Parameter list which needs update
    params = layer3.params + layer2.params + layer1.params + layer0.params
    
    # Load the parameters if we want
    if loadmodel == True:
        load(folder)
        

    # Gradient of costfunction w.r.t. parameters
    grads = T.grad(cost, params)

    # Gradient decent for every parameters
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    # Theano function for calculating the cost and updating the model
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


    print('... training')
    train_net(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)
    
    # Save parameters after training
    save(folder)
    
    
def test_model(learning_rate=0.1, nkerns=[16, 512], kern_shape=[9,7],
            batch_size=200, verbose=False, loadmodel=True):
    """
    learning_rate: term for the gradient 

    nkerns: number of kernels on each layer
    
    kern_shape: list of numbers with the dimensions of the kernels

    batch_szie: number of examples in minibatch.

    verbose: to print out epoch summary or not to
    
    loadmodel: load parameters from a saved .npy file

    """
    # Folder of saving and loading parameters
    folder='results'
    
    # Random seed
    rng = numpy.random.RandomState(23455)

    # Load testset
    datasets = load_test(theano_shared=False)

    # Function for loading parameters
    def load(folder):
        for param in params:
            param.set_value(numpy.load(os.path.join(folder,
                            param.name + '.npy')))


    # Splitting the data from the classes
    test_set_x, test_set_y = datasets[0]
    test_set_x = test_set_x.astype(numpy.float32)

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ###############
    # BUILD MODEL #
    ###############
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 1 * 100 * 100)
    # to a 4D tensor, which is expected by theano
    layer0_input = x.reshape((batch_size, 1, 100, 100))
    
    # First convolutional pooling layer
    layer0 = ConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=((batch_size, 1, 100, 100)), 
        filter_shape=((nkerns[0], 1, kern_shape[0], kern_shape[0])),
        poolsize=((2,2)),
        idx=0
    )

    # Second convolutional pooling layer
    layer1 = ConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=((batch_size, nkerns[0], 46, 46)),
        filter_shape=((nkerns[1], nkerns[0], kern_shape[1], kern_shape[1])),
        poolsize=((2,2)),
        idx=1
    )

    # Flatten input for fully connected layer
    layer2_input = layer1.output.flatten(2)

    # Fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=(nkerns[1]*20*20),
        n_out=(500),
        activation=T.tanh
    )   
    

    # Output layer
    layer3 = LogisticRegression(
         input=layer2.output,
         n_in=500,
    n_out=530)

    # Parameter list which needs update
    params = layer3.params + layer2.params + layer1.params + layer0.params
    
    # load parameters
    if loadmodel == True:
        load(folder)

    # function for prediction
    test_output = theano.function(
        [x],
        layer3.y_pred
    )
    
    # function for disrtribution over classes
    test_dist = theano.function(
        [x],
        layer3.p_y_given_x
    )
    
    # function for top 3 classes
    test_top = theano.function(
        [x],
        layer3.top
    )
    
    # function for top 3 classes probability
    test_topdist = theano.function(
        [x],
        layer3.topdist
    )

    return test_output(test_set_x), test_dist(test_set_x), test_top(test_set_x), test_topdist(test_set_x),
    
