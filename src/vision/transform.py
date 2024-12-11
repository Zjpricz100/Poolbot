import rospy
import tf2_ros
import numpy as np
from geometry_msgs.msg import TransformStamped


def quaternion_to_matrix(q):
    """Convert a quaternion into a 4x4 transformation matrix."""
    x, y, z, w = q
    matrix = np.array([
        [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w, 0],
        [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w, 0],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2, 0],
        [0, 0, 0, 1]
    ])
    return matrix


def transform_to_matrix(transform):
    """Convert Transform to a 4x4 transformation matrix."""
    q = [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]
    t = [transform.translation.x, transform.translation.y, transform.translation.z]
    matrix = quaternion_to_matrix(q)
    matrix[:3, 3] = t
    return matrix


def matrix_to_transform(matrix):
    """Convert a 4x4 transformation matrix back to a TransformStamped."""
    # Extract translation
    t = matrix[:3, 3]
    # Compute quaternion
    m = matrix[:3, :3]
    trace = np.trace(m)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s

    # Build TransformStamped
    transform = TransformStamped()
    transform.transform.translation.x = t[0]
    transform.transform.translation.y = t[1]
    transform.transform.translation.z = t[2]
    transform.transform.rotation.x = x
    transform.transform.rotation.y = y
    transform.transform.rotation.z = z
    transform.transform.rotation.w = w
    return transform


def compute_base_to_usb(tf_buffer, tf_broadcaster):
    try:
        # Lookup transforms
        usb_to_ar = tf_buffer.lookup_transform("usb_cam", "ar_marker_3", rospy.Time(0))
        head_to_ar = tf_buffer.lookup_transform("head_camera", "ar_marker_3", rospy.Time(0))
        base_to_head = tf_buffer.lookup_transform("base", "head_cam", rospy.Time(0))

        # Convert to matrices
        T_usb_to_ar = transform_to_matrix(usb_to_ar.transform)
        T_head_to_ar = transform_to_matrix(head_to_ar.transform)
        T_base_to_head = transform_to_matrix(base_to_head.transform)

        # Compute head_cam to usb_cam
        T_ar_to_usb = np.linalg.inv(T_usb_to_ar)
        T_head_to_usb = np.dot(T_head_to_ar, T_ar_to_usb)

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
