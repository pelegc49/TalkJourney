using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.InputSystem;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Interaction.Toolkit.Interactables;
using UnityEngine.XR.Interaction.Toolkit.Locomotion;
using TalkJourney.BubbleSystem.Data;

namespace TalkJourney.BubbleSystem.Flow
{
    [DisallowMultipleComponent]
    [RequireComponent(typeof(XRSimpleInteractable))]
    public class StageEntrance : MonoBehaviour
    {
        [Header("XR Rig")]
        [Tooltip("Root transform of the XR Origin / XR Rig. This transform will move when the player sits in the stage.")]
        public Transform xrOriginRoot;

        [Tooltip("Optional camera transform under the XR Origin. If left empty, the first Camera child of xrOriginRoot will be used.")]
        public Transform xrCamera;

        [Header("Seat Pose")]
        [Tooltip("Transform that represents the head position and rotation for the stage entry.")]
        public Transform seatPose;

        [Tooltip("Additional world-space offset from the seatPose. Useful to adjust vertical/head offset.")]
        public Vector3 seatOffset = Vector3.zero;

        [Header("Stage Launch")]
        [Tooltip("Launcher that starts the BubbleSystem stage when the object is clicked.")]
        public BubbleSystemLauncher bubbleSystemLauncher;

        [Tooltip("Stage data asset to start when this entrance is clicked.")]
        public StageData stageData;

        [Tooltip("Optional runtime parent for BubbleSystemBootstrap / result presenter instances.")]
        public Transform stageRuntimeParent;

        [Header("Stage UI")]
        [Tooltip("Optional GameObject root for the stage canvas. This will be enabled when the stage starts and disabled when exiting.")]
        public GameObject stageCanvasRoot;

        [Tooltip("If true, the stageCanvasRoot is hidden on Awake before the game starts.")]
        public bool hideStageCanvasOnAwake = true;

        [Header("Exit")]
        [Tooltip("If true, the original XR rig position/rotation is restored when exiting the stage.")]
        public bool restoreOriginalRigPoseOnExit = true;

        [Tooltip("Optional exit pose transform to move the rig to when leaving the stage instead of restoring the original pose.")]
        public Transform exitPose;

        [Tooltip("If true, the BubbleSystem runtime containers created for the stage are destroyed when exiting.")]
        public bool destroyStageContainersOnExit = true;

        [Header("Events")]
        public UnityEvent onStageStarted;
        public UnityEvent onStageExited;

        [Header("Input")]
        [Tooltip("Assign the XRI Default Input Actions Activate action from the XRI Right Interaction map.")]
        public InputActionReference activateAction;

        public bool IsStageActive { get; private set; }
        public static StageEntrance ActiveStageEntrance { get; private set; }

        private XRSimpleInteractable _xrSimpleInteractable;
        private readonly List<LocomotionProvider> _disabledLocomotionProviders = new List<LocomotionProvider>();
        private Vector3 _savedRigPosition;
        private Quaternion _savedRigRotation;
        private bool _hasSavedRigPose;
        private bool _isHovered;

        private void Awake()
        {
            _xrSimpleInteractable = GetComponent<XRSimpleInteractable>();
            if (_xrSimpleInteractable == null)
            {
                Debug.LogError("StageEntrance requires an XRSimpleInteractable on the same GameObject.", this);
                enabled = false;
                return;
            }

            _xrSimpleInteractable.hoverEntered.AddListener(OnHoverEntered);
            _xrSimpleInteractable.hoverExited.AddListener(OnHoverExited);
        }

        private void OnEnable()
        {
            if (activateAction != null && activateAction.action != null)
            {
                activateAction.action.performed += OnActivatePerformed;
            }
        }

        private void Start()
        {
            if (xrOriginRoot == null)
            {
                Debug.LogError("StageEntrance.xrOriginRoot is not assigned.", this);
            }

            if (xrCamera == null && xrOriginRoot != null)
            {
                var cameraComp = xrOriginRoot.GetComponentInChildren<Camera>();
                if (cameraComp != null)
                {
                    xrCamera = cameraComp.transform;
                }
            }

            if (stageData == null)
            {
                Debug.LogWarning("StageEntrance.stageData is not assigned. The stage will not start without a StageData asset.", this);
            }

            if (bubbleSystemLauncher == null)
            {
                bubbleSystemLauncher = FindFirstObjectByType<BubbleSystemLauncher>();
                if (bubbleSystemLauncher == null)
                {
                    Debug.LogWarning("StageEntrance could not find a BubbleSystemLauncher in the scene.", this);
                }
            }

            if (stageCanvasRoot != null && hideStageCanvasOnAwake)
            {
                stageCanvasRoot.SetActive(false);
            }
        }

        private void OnDisable()
        {
            if (activateAction != null && activateAction.action != null)
            {
                activateAction.action.performed -= OnActivatePerformed;
            }

            if (_xrSimpleInteractable != null)
            {
                _xrSimpleInteractable.hoverEntered.RemoveListener(OnHoverEntered);
                _xrSimpleInteractable.hoverExited.RemoveListener(OnHoverExited);
            }
        }

        private void OnDestroy()
        {
            if (ActiveStageEntrance == this)
            {
                ActiveStageEntrance = null;
            }
        }

        private void OnActivatePerformed(InputAction.CallbackContext context)
        {
            if (!_isHovered)
            {
                return;
            }

            StartStage();
        }

        private void OnHoverEntered(HoverEnterEventArgs args)
        {
            _isHovered = true;
        }

        private void OnHoverExited(HoverExitEventArgs args)
        {
            _isHovered = false;
        }

        public void OnClicked()
        {
            StartStage();
        }



        public void StartStage()
        {
            if (IsStageActive)
            {
                return;
            }

            if (xrOriginRoot == null || xrCamera == null || seatPose == null)
            {
                Debug.LogError("StageEntrance cannot start stage because XR rig or seatPose is not configured.", this);
                return;
            }

            SaveRigPose();
            DisableLocomotionProviders();
            TeleportRigToSeat();

            if (stageCanvasRoot != null)
            {
                stageCanvasRoot.SetActive(true);
            }

            if (bubbleSystemLauncher != null && stageData != null)
            {
                bubbleSystemLauncher.StartStageFromButton(stageData, stageRuntimeParent);
            }

            IsStageActive = true;
            ActiveStageEntrance = this;
            onStageStarted?.Invoke();
        }

        public void ExitStage()
        {
            if (!IsStageActive)
            {
                return;
            }

            if (restoreOriginalRigPoseOnExit && _hasSavedRigPose)
            {
                RestoreRigPose();
            }
            else if (exitPose != null)
            {
                TeleportRigToWorldPose(exitPose.position, Quaternion.Euler(0f, exitPose.rotation.eulerAngles.y, 0f));
            }

            if (stageCanvasRoot != null)
            {
                stageCanvasRoot.SetActive(false);
            }

            if (destroyStageContainersOnExit && bubbleSystemLauncher != null)
            {
                bubbleSystemLauncher.StopAndDestroyActiveBootstrap();
            }

            RestoreLocomotionProviders();

            if (ActiveStageEntrance == this)
            {
                ActiveStageEntrance = null;
            }

            IsStageActive = false;
            onStageExited?.Invoke();
        }

        public void OnExitStageButtonClicked()
        {
            ExitStage();
        }

        private void SaveRigPose()
        {
            _savedRigPosition = xrOriginRoot.position;
            _savedRigRotation = xrOriginRoot.rotation;
            _hasSavedRigPose = true;
        }

        private void RestoreRigPose()
        {
            xrOriginRoot.SetPositionAndRotation(_savedRigPosition, _savedRigRotation);
        }

        private void TeleportRigToSeat()
        {
            var targetPosition = seatPose.position + seatPose.TransformDirection(seatOffset);
            var targetRotation = Quaternion.Euler(0f, seatPose.rotation.eulerAngles.y, 0f);
            TeleportRigToWorldPose(targetPosition, targetRotation);
        }

        private void TeleportRigToWorldPose(Vector3 targetWorldPosition, Quaternion targetWorldRotation)
        {
            if (xrOriginRoot == null || xrCamera == null)
            {
                return;
            }

            var localCameraPosition = xrOriginRoot.InverseTransformPoint(xrCamera.position);
            var localCameraRotation = Quaternion.Inverse(xrOriginRoot.rotation) * xrCamera.rotation;
            var localCameraYaw = Quaternion.Euler(0f, localCameraRotation.eulerAngles.y, 0f);
            var newRigRotation = targetWorldRotation * Quaternion.Inverse(localCameraYaw);

            xrOriginRoot.rotation = newRigRotation;
            xrOriginRoot.position = targetWorldPosition - newRigRotation * localCameraPosition;
        }

        private void DisableLocomotionProviders()
        {
            _disabledLocomotionProviders.Clear();
            if (xrOriginRoot == null)
            {
                return;
            }

            var providers = FindObjectsByType<LocomotionProvider>(FindObjectsSortMode.None);
            foreach (var provider in providers)
            {
                if (provider == null || !provider.isActiveAndEnabled)
                {
                    continue;
                }

                if (provider.transform == xrOriginRoot || provider.transform.IsChildOf(xrOriginRoot))
                {
                    provider.enabled = false;
                    _disabledLocomotionProviders.Add(provider);
                }
            }
        }

        private void RestoreLocomotionProviders()
        {
            for (int i = 0; i < _disabledLocomotionProviders.Count; i++)
            {
                var provider = _disabledLocomotionProviders[i];
                if (provider != null)
                {
                    provider.enabled = true;
                }
            }

            _disabledLocomotionProviders.Clear();
        }
    }
}
