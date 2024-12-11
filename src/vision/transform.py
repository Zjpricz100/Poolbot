import rospy
import tf2_ros
import numpy as np
from geometry_msgs.msg import TransformStamped
from tf_transformations import quaternion_matrix, quaternion_from_matrix


def transform_to_matrix(transform):
    """Convert TransformStamped to a 4x4 transformation matrix."""
    q = [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]
    t = [transform.translation.x, transform.translation.y, transform.translation.z]
    matrix = quaternion_matrix(q)
    matrix[:3, 3] = t
    return matrix


def matrix_to_transform(matrix):
    """Convert 4x4 transformation matrix to TransformStamped."""
    q = quaternion_from_matrix(matrix)
    t = matrix[:3, 3]
    transform = TransformStamped()
    transform.transform.translation.x = t[0]
    transform.transform.translation.y = t[1]
    transform.transform.translation.z = t[2]
    transform.transform.rotation.x = q[0]
    transform.transform.rotation.y = q[1]
    transform.transform.rotation.z = q[2]
    transform.transform.rotation.w = q[3]
    return transform


def compute_base_to_usb(tf_buffer, tf_broadcaster):
    try:
        # Lookup transforms for AR marker relative to each camera
        transform_usb_to_ar = tf_buffer.lookup_transform("usb_cam", "ar_marker_3", rospy.Time(0))
        transform_head_to_ar = tf_buffer.lookup_transform("head_camera", "ar_marker_3", rospy.Time(0))

        # Convert transforms to matrices
        T_usb_to_ar = transform_to_matrix(transform_usb_to_ar.transform)
        T_head_to_ar = transform_to_matrix(transform_head_to_ar.transform)

        # Compute ar_marker to usb_cam
        T_ar_to_usb = np.linalg.inv(T_usb_to_ar)

        # Compute head_cam to usb_cam
        T_head_to_usb = np.dot(T_head_to_ar, T_ar_to_usb)

        # Lookup transform from base to head_cam
        base_to_head = tf_buffer.lookup_transform("base", "head_cam", rospy.Time(0))
        T_base_to_head = transform_to_matrix(base_to_head.transform)

        # Compute base to usb_cam
        T_base_to_usb = np.dot(T_base_to_head, T_head_to_usb)

        # Publish the transform
        transform_msg = matrix_to_transform(T_base_to_usb)
        transform_msg.header.stamp = rospy.Time.now()
        transform_msg.header.frame_id = "base"
        transform_msg.child_frame_id = "usb_cam"
        tf_broadcaster.sendTransform(transform_msg)

        rospy.loginfo("Published base to usb_cam transform.")

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logwarn(f"TF Exception: {e}")


def main():
    rospy.init_node('base_to_usb_transform')

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        compute_base_to_usb(tf_buffer, tf_broadcaster)
        rate.sleep()


if __name__ == "__main__":
    main()
