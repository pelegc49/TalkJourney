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
    [Tooltip("Drag the LEFT Hand Direct Interactor here.")]
    public XRBaseInteractor leftHandDirectInteractor;
    [Tooltip("Drag the LEFT Hand Near-Far Interactor here.")]
    public XRBaseInteractor leftHandNearFarInteractor;
    [Tooltip("The trigger input action for the LEFT hand.")]
    public InputActionReference leftSpawnAction;

    [Header("Right Hand Settings")]
    [Tooltip("Drag the RIGHT Hand Direct Interactor here.")]
    public XRBaseInteractor rightHandDirectInteractor;
    [Tooltip("Drag the RIGHT Hand Near-Far Interactor here.")]
    public XRBaseInteractor rightHandNearFarInteractor;


    [Tooltip("The trigger input action for the RIGHT hand.")]
    public InputActionReference rightSpawnAction;

    // Track the presence of each hand in the chest zone separately
    private bool isLeftHandInChest = false;
    private bool isRightHandInChest = false;

    private float hideAt = float.PositiveInfinity;
    private float checkReleasedAt = float.PositiveInfinity;


    private string handTag = "Hand"; // Tag to identify hand colliders
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
        if (leftSpawnAction != null) leftSpawnAction.action.canceled += OnLeftTriggerReleased;
        if (rightSpawnAction != null) rightSpawnAction.action.performed += OnRightTriggerPressed;
        if (rightSpawnAction != null) rightSpawnAction.action.canceled += OnRightTriggerReleased;

        // Subscribe each hand to its specific input action

    }

    void OnDisable()
    {
        Debug.Log("ChestInventoryManager disabled. Unsubscribing from input actions.");
        if (leftSpawnAction != null) leftSpawnAction.action.performed -= OnLeftTriggerPressed;
        if (leftSpawnAction != null) leftSpawnAction.action.canceled -= OnLeftTriggerReleased;
        if (rightSpawnAction != null) rightSpawnAction.action.performed -= OnRightTriggerPressed;
        if (rightSpawnAction != null) rightSpawnAction.action.canceled -= OnRightTriggerReleased;
    }

    void OnTriggerEnter(Collider other)
    {
        if (!other.CompareTag(handTag)) return;
        Debug.Log("Trigger entered by: " + other.name);
        // Identify exactly which hand entered the chest trigger
        XRBaseInteractor interactor = other.GetComponentInChildren<XRDirectInteractor>(false);
        Debug.Log("Interactor found: " + (interactor != null ? interactor.name : "None"));
        if (interactor == leftHandDirectInteractor)
        {
            isLeftHandInChest = true;
        }
        else if (interactor == rightHandDirectInteractor)
        {
            isRightHandInChest = true;
        }
    }

    void OnTriggerExit(Collider other)
    {
        if (!other.CompareTag(handTag)) return;
        Debug.Log("Trigger exited by: " + other.name);
        // Identify exactly which hand exited the chest trigger
        XRBaseInteractor interactor = other.GetComponentInChildren<XRDirectInteractor>(false);

        if (interactor == leftHandDirectInteractor)
        {
            isLeftHandInChest = false;
        }
        else if (interactor == rightHandDirectInteractor)
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
            SpawnTabletOnHand(leftHandDirectInteractor);
            hideAt = float.PositiveInfinity;
        }
    }

    void OnLeftTriggerReleased(InputAction.CallbackContext context)
    {
        bool grabbing = false;
        foreach (IXRSelectInteractable inter in leftHandDirectInteractor.interactablesSelected)
        {
            if (tabletInteractable == inter)
            {
                grabbing = true;
            }
        }
        if (grabbing)
        {
            leftHandDirectInteractor.EndManualInteraction();
        }

    }
    void OnRightTriggerReleased(InputAction.CallbackContext context)
    {
        bool grabbing = false;
        foreach (IXRSelectInteractable inter in rightHandDirectInteractor.interactablesSelected)
        {
            if (tabletInteractable == inter)
            {
                grabbing = true;
            }
        }
        if (grabbing)
        {
            rightHandDirectInteractor.EndManualInteraction();
        }

    }
    void OnRightTriggerPressed(InputAction.CallbackContext context)
    {
        Debug.Log("Right trigger pressed. Right hand in chest: " + isRightHandInChest);
        // Only spawn if the right hand is inside the chest
        if (isRightHandInChest)
        {
            SpawnTabletOnHand(rightHandDirectInteractor);
            hideAt = float.PositiveInfinity;
        }
    }

    void SpawnTabletOnHand(XRBaseInteractor targetHand)
    {
        if (!float.IsPositiveInfinity(hideAt))
        {
            tabletInstance.SetActive(false);
        }
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
        checkReleasedAt = Time.time + 0.1f;
    }

    void LateUpdate()
    {
        if (Time.time >= checkReleasedAt)
        {
            if (tabletInteractable.interactorsSelecting.Count == 0)
            {
                Debug.Log("released");
                // set a time 2 seconds from now
                hideAt = Time.time + 2f;
            }
            checkReleasedAt = float.PositiveInfinity;
        }

        if (Time.time >= hideAt)
        {
            tabletInstance.SetActive(false);
            hideAt = float.PositiveInfinity;
        }
    }

}