#!/usr/bin/env python3
"""
Destination Commander Node
- Subscribes to /destination_command topic
- Reads destination coordinates from destination.yaml
- Sends navigation goal to Nav2
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
import yaml
from pathlib import Path


class DestinationCommanderNode(Node):
    def __init__(self):
        super().__init__('destination_commander')

        # Subscriber to /destination_command
        self.subscription = self.create_subscription(
            String,
            '/destination_command',
            self.destination_callback,
            10
        )

        # Action client for Nav2
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Load destination coordinates
        self.destinations = self.load_destinations()

        self.get_logger().info('Destination Commander Node Started')
        self.get_logger().info(f'Loaded {len(self.destinations)} destinations')
        self.get_logger().info('Waiting for /destination_command...')

    def load_destinations(self):
        """Load destination coordinates from YAML file"""
        yaml_path = Path(__file__).parent / 'destination.yaml'

        if not yaml_path.exists():
            self.get_logger().error(f'destination.yaml not found at {yaml_path}')
            return {}

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            destinations = data.get('destinations', {})
            self.get_logger().info(f'Available destinations: {list(destinations.keys())}')
            return destinations

        except Exception as e:
            self.get_logger().error(f'Error loading destination.yaml: {e}')
            return {}

    def destination_callback(self, msg):
        """Handle incoming destination command"""
        destination_name = msg.data

        self.get_logger().info(f'Received destination command: "{destination_name}"')

        if destination_name not in self.destinations:
            self.get_logger().warn(f'Unknown destination: "{destination_name}"')
            self.get_logger().info(f'Available: {list(self.destinations.keys())}')
            return

        # Get destination coordinates
        dest = self.destinations[destination_name]
        x = dest['x']
        y = dest['y']
        yaw = dest['yaw']

        self.get_logger().info(f'Navigating to {destination_name}: x={x}, y={y}, yaw={yaw}')

        # Send navigation goal
        self.send_nav_goal(x, y, yaw)

    def send_nav_goal(self, x, y, yaw):
        """Send navigation goal to Nav2"""
        # Wait for Nav2 action server
        self.get_logger().info('Waiting for Nav2 action server...')
        self.nav_client.wait_for_server()

        # Create goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Set position
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        # Set orientation (yaw to quaternion)
        from math import sin, cos
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = cos(yaw / 2.0)

        # Send goal
        self.get_logger().info(f'Sending navigation goal to Nav2...')
        send_goal_future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handle Nav2 goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Navigation goal rejected by Nav2!')
            return

        self.get_logger().info('Navigation goal accepted! Robot is moving...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        """Handle Nav2 navigation result"""
        result = future.result().result
        self.get_logger().info('Navigation completed!')

    def feedback_callback(self, feedback_msg):
        """Handle Nav2 feedback (optional)"""
        # feedback = feedback_msg.feedback
        # self.get_logger().info(f'Navigation feedback: {feedback}')
        pass


def main(args=None):
    rclpy.init(args=args)
    node = DestinationCommanderNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
