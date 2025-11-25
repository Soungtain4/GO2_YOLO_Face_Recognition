#!/usr/bin/env python3
"""
Destination HTTP Receiver
- Receives HTTP POST requests from face recognition board
- Converts to ROS2 /destination_command topic
- Acts as bridge between HTTP and ROS2
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from flask import Flask, request, jsonify
import threading

app = Flask(__name__)

class DestinationBridgeNode(Node):
    def __init__(self):
        super().__init__('destination_bridge')

        # Publisher to /destination_command
        self.publisher_ = self.create_publisher(String, '/destination_command', 10)

        self.get_logger().info('Destination Bridge Node Started')
        self.get_logger().info('Ready to receive HTTP requests and publish to ROS2')

    def publish_destination(self, destination):
        """Publish destination to ROS2 topic"""
        msg = String()
        msg.data = destination
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published destination command: "{destination}"')


# Global node reference
bridge_node = None

@app.route('/destination', methods=['POST'])
def receive_destination():
    """HTTP endpoint to receive destination from board"""
    try:
        data = request.get_json()

        if not data or 'destination' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing "destination" field in JSON'
            }), 400

        destination = data['destination']

        # Publish to ROS2
        if bridge_node:
            bridge_node.publish_destination(destination)

            return jsonify({
                'status': 'success',
                'message': f'Destination "{destination}" published to ROS2',
                'destination': destination
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': 'ROS2 node not initialized'
            }), 500

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'destination_http_receiver',
        'ros2_node': 'active' if bridge_node else 'inactive'
    }), 200


def run_flask_server():
    """Run Flask server in separate thread"""
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


def main(args=None):
    global bridge_node

    # Initialize ROS2
    rclpy.init(args=args)
    bridge_node = DestinationBridgeNode()

    # Start Flask server in separate thread
    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()

    print("\n" + "="*60)
    print("Destination HTTP Receiver Started")
    print("="*60)
    print(f"HTTP Server: http://0.0.0.0:5000")
    print(f"Endpoint: POST /destination")
    print(f"ROS2 Topic: /destination_command")
    print("="*60 + "\n")

    try:
        # Keep ROS2 node spinning
        rclpy.spin(bridge_node)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        bridge_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
