using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Interaction.Toolkit.Interactors;
using UnityEngine.XR.Interaction.Toolkit.Interactables;
using System;

public class ChestInventoryManager : MonoBehaviour
{
    [Header("Tablet Settings")]
    [Tooltip("The tablet object in the scene. It should have an XRGrabInteractable component.")]
    public GameObject tabletInstance;
    private XRGrabInteractable tabletInteractable;

    [Header("Left Hand Settings")]
    [Tooltip("Drag the LEFT Hand Interactor (Direct or Near-Far) here.")]
    public NearFarInteractor leftHandInteractor;
    [Tooltip("The trigger input action for the LEFT hand.")]
    public InputActionReference leftSpawnAction;

    [Header("Right Hand Settings")]
    [Tooltip("Drag the RIGHT Hand Interactor (Direct or Near-Far) here.")]
    public NearFarInteractor rightHandInteractor;
    [Tooltip("The trigger input action for the RIGHT hand.")]
    public InputActionReference rightSpawnAction;

    // Track the presence of each hand in the chest zone separately
    private bool isLeftHandInChest = false;
    private bool isRightHandInChest = false;

    void Start()
    {
        Debug.Log("ChestInventoryManager started. Initializing tablet and input actions.");
        if (tabletInstance != null)
        {
            tabletInteractable = tabletInstance.GetComponent<XRGrabInteractable>();
            tabletInstance.SetActive(false);
        }
        if (tabletInteractable != null)
        {
            tabletInteractable.selectExited.AddListener(OnTabletReleased);
        }
    }

    void OnEnable()
    {
        Debug.Log("ChestInventoryManager enabled. Subscribing to input actions.");
        if (leftSpawnAction != null) leftSpawnAction.action.performed += OnLeftTriggerPressed;
        if (rightSpawnAction != null) rightSpawnAction.action.performed += OnRightTriggerPressed;

        // Subscribe each hand to its specific input action

    }

    void OnDisable()
    {
        Debug.Log("ChestInventoryManager disabled. Unsubscribing from input actions.");
        if (leftSpawnAction != null) leftSpawnAction.action.performed -= OnLeftTriggerPressed;
        if (rightSpawnAction != null) rightSpawnAction.action.performed -= OnRightTriggerPressed;

        if (tabletInteractable != null)
        {
            tabletInteractable.selectExited.RemoveListener(OnTabletReleased);
        }
    }

    void OnTriggerEnter(Collider other)
    {
        Debug.Log("Trigger entered by: " + other.name);
        // Identify exactly which hand entered the chest trigger
        NearFarInteractor interactor = other.GetComponentInChildren<NearFarInteractor>(false);
        Debug.Log("Interactor found: " + (interactor != null ? interactor.name : "None"));
        if (interactor == leftHandInteractor)
        {
            isLeftHandInChest = true;
        }
        else if (interactor == rightHandInteractor)
        {
            isRightHandInChest = true;
        }
    }

     void OnTriggerExit(Collider other)
    {
        Debug.Log("Trigger exited by: " + other.name);
        // Identify exactly which hand exited the chest trigger
        NearFarInteractor interactor = other.GetComponentInChildren<NearFarInteractor>(false);

        if (interactor == leftHandInteractor)
        {
            isLeftHandInChest = false;
        }
        else if (interactor == rightHandInteractor)
        {
            isRightHandInChest = false;
        }
    }

     void OnLeftTriggerPressed(InputAction.CallbackContext context)
    {
        Debug.Log("Left trigger pressed. Left hand in chest: " + isLeftHandInChest);
        // Only spawn if the left hand is inside the chest
        if (isLeftHandInChest)
        {
            SpawnTabletOnHand(leftHandInteractor);
        }
    }

     void OnRightTriggerPressed(InputAction.CallbackContext context)
    {
        Debug.Log("Right trigger pressed. Right hand in chest: " + isRightHandInChest);
        // Only spawn if the right hand is inside the chest
        if (isRightHandInChest)
        {
            SpawnTabletOnHand(rightHandInteractor);
        }
    }

     void SpawnTabletOnHand(XRBaseInteractor targetHand)
    {
        // Make sure we only spawn the tablet if it's currently hidden
        if (!tabletInstance.activeInHierarchy && targetHand != null)
        {
            // 1. Move the tablet to the specific hand's position and rotation
            tabletInstance.transform.position = targetHand.transform.position;
            tabletInstance.transform.rotation = targetHand.transform.rotation;

            // 2. Activate the tablet
            tabletInstance.SetActive(true);

            // 3. Force the specific interactor to manually grab the tablet
            IXRSelectInteractable selectInteractable = tabletInteractable as IXRSelectInteractable;

            if (selectInteractable != null)
            {
                targetHand.StartManualInteraction(selectInteractable);
            }
        }
    }

     void OnTabletReleased(SelectExitEventArgs args)
    {
        tabletInstance.SetActive(false);
    }
    
}