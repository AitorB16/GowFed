import tensorflow_federated as tff

print(tff.federated_computation(lambda: 'Hello World')())
