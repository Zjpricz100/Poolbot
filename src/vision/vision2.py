import rospy
from tf import TransformListener
from tf.transformations import inverse_matrix, concatenate_matrices

def compute_camera_transform():
    rospy.init_node('camera_transform_computation', anonymous=True)
    listener = TransformListener()

    try:
        # Wait for transforms to be available
        listener.waitForTransform('/usb_cam', '/ar_marker_3', rospy.Time(0), rospy.Duration(4.0))
        listener.waitForTransform('/head_camera', '/ar_marker_3', rospy.Time(0), rospy.Duration(4.0))

        # Get transforms
        (trans_space_to_tag, rot_space_to_tag) = listener.lookupTransform('/usb_cam', '/ar_marker_3', rospy.Time(0))
        (trans_head_to_tag, rot_head_to_tag) = listener.lookupTransform('/head_camera', '/ar_marker_3', rospy.Time(0))

        # Convert to transformation matrices
        T_space_to_tag = listener.fromTranslationRotation(trans_space_to_tag, rot_space_to_tag)
        T_head_to_tag = listener.fromTranslationRotation(trans_head_to_tag, rot_head_to_tag)

        # Compute inverse of head_camera -> ar_tag
        T_tag_to_head = inverse_matrix(T_head_to_tag)

        # Compute space_camera -> head_camera
        T_space_to_head = concatenate_matrices(T_space_to_tag, T_tag_to_head)

        print("Transform from usb_camera to head_camera:")
        print(T_space_to_head)

    except Exception as e:
        rospy.logerr("Error computing transform: %s", str(e))

if __name__ == "__main__":
    compute_camera_transform()
