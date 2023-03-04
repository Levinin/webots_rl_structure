#!/home/levinin/anaconda3/envs/masters/bin/python
from controller import Supervisor
from controller import Keyboard
from controller import Display

import numpy as np
import sys
from os.path import exists
from time import sleep


class SimpleSupervisor:
    def __init__(self):
        self.time_step = 32  # ms  ~ 30 Hz

        self.supervisor = Supervisor()

        # Get the robot
        self.robot_node = self.supervisor.getFromDef("EPUCK")
        if self.robot_node is None:
            sys.stderr.write("No DEF Controller node found in the current world file\n")
            sys.exit(1)

        # Get the robot translation and rotation
        self.trans_field = self.robot_node.getField("translation")
        self.rot_field = self.robot_node.getField("rotation")

        # Enable Receiver and Emitter
        self.emitter = self.supervisor.getDevice("emitter")
        self.receiver = self.supervisor.getDevice("receiver")
        self.receiver.enable(self.time_step)
        self.previous_message = ""

        self.__waiting_for_end = False

    def get_message_from_agent(self):
        if self.receiver.getQueueLength() > 0:
            text = self.receiver.getString()
            self.receiver.nextPacket()
            if text == self.previous_message:
                return
            self.previous_message = text
            if text == "end of update":
                self.__waiting_for_end = False

    def send_message_to_agent(self, new_message: str = ""):
        self.emitter.send(new_message)

    def reset_robot(self):
        """Set the robot at the start of an episode. In this case a static location."""
        self.trans_field.setSFVec3f([-0.8, 0, -0.8])
        self.rot_field.setSFRotation([0.423, 0.641, 0.641, -2.34])
        self.robot_node.resetPhysics()

    def run_episode(self, seconds: int = 15) -> float:
        """Run episode for passed number of seconds."""
        stop = int((seconds * 1000) / self.time_step)
        iterations = 0

        while self.supervisor.step(self.time_step) != -1:
            if stop == iterations:
                break
            iterations += 1

            # TODO in real system:
            #   Check if the agent thinks it has completed the episode and if so break
            #   Check whether the agent has completed the episode goal and break.

        end_of_episode_reward = 0.2
        return end_of_episode_reward

    def run_demo(self):
        """Run the trained policy"""
        pass

    def run_training_episodes(self, num_episodes: int = 5):
        """Runs the training episodes. The main processing is completed in the epuck."""

        for episode in range(num_episodes):
            print(f"Episode: {episode}")
            self.reset_robot()
            self.send_message_to_agent("1 0.0")     # Tell the robot a new episode has started.

            end_of_episode_reward = self.run_episode()
            self.__waiting_for_end = True
            self.send_message_to_agent(f"0 {end_of_episode_reward}")

            # Wait for signal from agent to make sure we don't interrupt an update.
            while self.__waiting_for_end:
                self.get_message_from_agent()


if __name__ == "__main__":
    # Call Supervisor function to initiate the supervisor module   

    supervisor = SimpleSupervisor()

    keyboard = Keyboard()
    keyboard.enable(50)
    print("(R|r)un Best Policy or (S|s)earch for New Policy:")

    while supervisor.supervisor.step(supervisor.time_step) != -1:
        resp = keyboard.getKey()
        if resp == 83:
            supervisor.run_training_episodes(10)
        elif resp == 82:
            supervisor.run_demo()

