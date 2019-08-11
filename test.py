
import load_data
from network_plus import Network


batch_size = 16

mini_batches, validation_data, test_data = load_data.load_data_by_batch(batch_size)

net = Network([784, 30, 10], batch_size)
net.train(mini_batches, 2000, 0.5, 10, test_data,
          monitor_evaluation_accuracy=False,
          monitor_evaluation_cost=False,
          monitor_training_cost=True)
