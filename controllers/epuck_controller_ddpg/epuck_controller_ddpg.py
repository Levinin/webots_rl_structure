# Author:   Andy Edmondson
# Email:    andrew.edmondson@gmail.com
# Date:     3 March 2023
# Purpose:  EPuck robot controller for use with RL algorithms
#
# Includes:
import time

from controller import Robot, Receiver, Emitter, GPS
import numpy as np


class Controller:
    def __init__(self, robot):

        # Robot Parameters
        self.robot = robot
        self.time_step = 32  # ms    ~30 Hz
        self.max_speed = 5  # m/s

        # Enable the robot systems
        self.enable_motors()
        self.enable_proximity_sensors()
        self.enable_light_sensors()
        self.enable_ground_sensors()
        self.enable_emitter_and_receiver()

        # List for input sensor data
        self.inputs = []

        # Fitness value (initialization fitness parameters once)
        self.fitness_values = []
        self.fitness = 0
        self.end_of_episode_fitness = 0

        # Starting position
        self.startPosition = [0, 0]

        self.__ongoing = False

    def enable_motors(self):
        # Enable Motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.velocity_left = 0
        self.velocity_right = 0

    def enable_proximity_sensors(self):
        self.proximity_sensors = []
        for i in range(8):
            sensor_name = 'ps' + str(i)
            self.proximity_sensors.append(self.robot.getDevice(sensor_name))
            self.proximity_sensors[i].enable(self.time_step)

    def enable_light_sensors(self):
        self.light_sensors = []
        for i in range(8):
            sensor_name = 'ls' + str(i)
            self.light_sensors.append(self.robot.getDevice(sensor_name))
            self.light_sensors[i].enable(self.time_step)

    def enable_ground_sensors(self):
        self.left_ir = self.robot.getDevice('gs0')
        self.left_ir.enable(self.time_step)
        self.center_ir = self.robot.getDevice('gs1')
        self.center_ir.enable(self.time_step)
        self.right_ir = self.robot.getDevice('gs2')
        self.right_ir.enable(self.time_step)

    def enable_emitter_and_receiver(self):
        self.emitter = self.robot.getDevice("emitter")
        self.receiver = self.robot.getDevice("receiver")
        self.receiver.enable(self.time_step)
        self.__previous_message = ""

    def compute_and_actuate(self):
        """Get motor drive values from the network"""
        output = [0.2, 0.2]  # This would be the output of the network, right now it will be moving forwards only

        # Multiply the motor values my the max speed
        self.left_motor.setVelocity(output[0] * self.max_speed)
        self.right_motor.setVelocity(output[1] * self.max_speed)

    def calculate_step_reward(self) -> float:
        """Calculate the step reward."""
        calculated_value = 0.1
        self.fitness_values.append(calculated_value)
        self.fitness = np.mean(self.fitness_values)
        return calculated_value

    def calculate_final_reward(self) -> float:
        """Calculate final reward based on return from supervisor and step rewards earned."""
        return 0.1

    def send_data_to_supervisor(self, data: str = ""):
        """Emit data to the supervisor."""
        self.emitter.send(data)

    def get_data_from_supervisor(self):
        """Get data from the supervisor. Expected in the format: '0 4.2' indicating do not continue and a
        final score of 4.2."""
        if self.receiver.getQueueLength() > 0:
            text = self.receiver.getString()
            self.receiver.nextPacket()
            if text == self.__previous_message:
                return
            self.__previous_message = text
            o, r = text.split()
            self.__ongoing = False if int(o) == 0 else True
            self.end_of_episode_fitness = float(r)
            if self.__ongoing:
                self.send_data_to_supervisor("running episode")

    def read_ground_sensors(self):
        """Read values from the ground sensors and place into the arrays."""
        # Set clip values to make learning easier
        min_gs = 200
        max_gs = 500

        # Read Ground Sensors
        left = np.clip(self.left_ir.getValue(), a_max=max_gs, a_min=min_gs)
        center = np.clip(self.center_ir.getValue(), a_max=max_gs, a_min=min_gs)
        right = np.clip(self.right_ir.getValue(), a_max=max_gs, a_min=min_gs)

        # Normalize the values between 0 and 1 and save data
        self.inputs.append((left - min_gs) / (max_gs - min_gs))
        self.inputs.append((center - min_gs) / (max_gs - min_gs))
        self.inputs.append((right - min_gs) / (max_gs - min_gs))

    def read_distance_sensors(self):
        """Read the distance sensor data."""
        # Set the explicit list of sensor indices to read - range is 0-7
        sensor_index = [*range(8)]

        # Set clip range
        min_ds = 50
        max_ds = 725

        for i in sensor_index:
            temp = np.clip(self.proximity_sensors[i].getValue(), a_min=min_ds, a_max=max_ds)
            # Normalize the values between 0 and 1
            self.inputs.append((temp - min_ds) / (max_ds - min_ds))
            # print("Distance Sensors - Index: {}  Value: {}".format(i,self.proximity_sensors[i].getValue()))

    def read_light_sensors(self):
        """Read the light sensors"""
        # Set clip range
        min_ls = 20
        max_ls = 4000

        for i in range(8):
            temp = np.clip(self.light_sensors[i].getValue(), a_min=min_ls, a_max=max_ls)
            # Normalise to 0
            self.inputs.append((temp - min_ls) / (max_ls - min_ls))
            # print("Light Sensors - Index: {}  Value: {}".format(i,self.light_sensors[i].getValue()))

    # -----------------------------------------------------------------------------------------------
    def run_robot(self):
        """Main loop"""

        step_scores = 0
        step_count = 0
        while self.robot.step(self.time_step) != -1:
            # print("Running robot.")
            self.get_data_from_supervisor()

            if self.__ongoing:
                # Read sensors and perform actions
                self.inputs = []
                self.read_ground_sensors()  # 0-2
                self.read_distance_sensors()  # 3-10
                self.read_light_sensors()  # 11-18
                self.compute_and_actuate()

                step_scores = self.calculate_step_reward()

                if step_count % 100 == 0:
                    print(f"Step {step_count:5}: {step_scores}")

                # Record the experience to replay memory

                # Perform a network update - in a new process?

                step_count += 1

            # If we've been told it's the end of the episode, do a couple of things and then tell
            # the supervisor we are ready for the next one
            else:
                # There will be an 'end of episode' score, so now we calculate the final reward
                # Only do these things once at the end of the episode, not while waiting for the next one.
                if step_count != 0:
                    total_episode_reward = self.calculate_final_reward()
                    print(f"Outside episode {total_episode_reward}")
                self.send_data_to_supervisor("end of update")

                step_count = 0
                step_scores = 0
                # Plot episode_reward etc. to track learning - remember this loops per step so latch outputs


if __name__ == "__main__":
    my_robot = Robot()
    controller = Controller(my_robot)
    controller.run_robot()
