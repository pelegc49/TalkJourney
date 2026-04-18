using System;
using UnityEngine;

public class Follow : MonoBehaviour
{
    [Header("Target Setup")]
    [Tooltip("Drag the Main Camera (from your XR Rig) here")]
    public Transform playerTarget;

    [Tooltip("How high above the player the map camera should hover")]
    public float cameraHeight = 20f;

    [Header("Rotation Options")]
    [Tooltip("If true, the map spins as the player turns. If false, North is always up.")]
    public bool rotateWithPlayer = false;

    void LateUpdate()
    {
        if (playerTarget == null) return;

        // 1. Update Position
        // Follow the player's X and Z, but keep the camera hovering at the set height above them
        Vector3 newPosition = playerTarget.position;
        newPosition.y += cameraHeight;
        transform.position = newPosition;

        // 2. Update Rotation
        if (rotateWithPlayer)
        {
            // Follow the player's Y rotation (turning left/right), but lock X to 90 (looking down) and Z to 0
            transform.rotation = Quaternion.Euler(90f, playerTarget.eulerAngles.y, 0f);
        }
        else
        {
            // Lock the camera to always point North and look straight down
            transform.rotation = Quaternion.Euler(90f, 0f, 0f);
        }
    }
  
}