from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from kernels import core

@core.register_std_kernel
class Flatten(core.Kernel):

    @staticmethod
    def get_config():
        config = core.Config("Flatten", "flatten")
        config.add_input(core.Port(name="input"))
        config.add_output(core.Port(name="output"))
        return config

    def call(self, input):
        output = tf.contrib.layers.flatten(input[0])
        return dict(output=output)


@core.register_std_kernel
class Reshape(core.Kernel):

    @staticmethod
    def get_config():
        config = core.Config("Reshape", "reshape")
        config.add_input(core.Port(name="input"))
        config.add_output(core.Port(name="output"))
        config.add_attribute(
            core.Attribute(name="shape", type="array", value="[-1]"))
        return config

    def call(self, input):
        output = tf.reshape(input[0], self.shape)
        return dict(output=output)
