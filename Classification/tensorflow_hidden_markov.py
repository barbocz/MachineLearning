# https://colab.research.google.com/drive/15Cyy2H7nT40sGR7TBN5wBvgTd57mVKay#forceEdit=true&sandboxMode=true&scrollTo=P4M6cZww4mZk
# We will model a simple weather system and try to predict the temperature on each day given the following information.
#
# Cold days are encoded by a 0 and hot days are encoded by a 1.
# The first day in our sequence has an 80% chance of being cold.
# A cold day has a 30% chance of being followed by a hot day.
# A hot day has a 20% chance of being followed by a cold day.
# On each day the temperature is normally distributed with mean and standard deviation 0 and 5 on a cold day and mean and standard deviation 15 and 10 on a hot day.
# If you're unfamiliar with standard deviation it can be put simply as the range of expected values.
#
# In this example, on a hot day the average temperature is 15 and ranges from 5 to 25.

import tensorflow_probability as tfp  # We are using a different module from tensorflow this time
import tensorflow as tf

tfd = tfp.distributions  # making a shortcut for later on
initial_distribution = tfd.Categorical(probs=[0.2, 0.8])  # Refer to point 2 above
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5],
                                                 [0.2, 0.8]])  # refer to points 3 and 4 above
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])  # refer to point 5 above

# the loc argument represents the mean and the scale is the standard devitation

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)

print(model.mean())