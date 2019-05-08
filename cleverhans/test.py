import math

input_shape=[32, 32, 3]
log_resolution = int(round(   
          math.log(input_shape[0]) / math.log(2)))
print("log_resolution:%d" %log_resolution)