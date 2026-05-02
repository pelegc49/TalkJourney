using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Interaction.Toolkit.Interactables;

namespace TalkJourney.BubbleSystem.Flow
{
    [DisallowMultipleComponent]
    [RequireComponent(typeof(XRSimpleInteractable))]
    public class StageExitButtonBridge : MonoBehaviour
    {
        [Header("Input")]
        [Tooltip("Assign the XRI Default Input Actions Activate action from the XRI Right Interaction map.")]
        public InputActionReference activateAction;

        private XRSimpleInteractable _xrSimpleInteractable;
        private bool _isHovered;

        private void Awake()
        {
            _xrSimpleInteractable = GetComponent<XRSimpleInteractable>();
            if (_xrSimpleInteractable != null)
            {
                _xrSimpleInteractable.hoverEntered.AddListener(OnHoverEntered);
                _xrSimpleInteractable.hoverExited.AddListener(OnHoverExited);
            }
        }

        private void OnEnable()
        {
            if (activateAction != null && activateAction.action != null)
            {
                activateAction.action.performed += OnActivatePerformed;
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

        private void OnActivatePerformed(InputAction.CallbackContext context)
        {
            if (!_isHovered)
            {
                return;
            }

            ExitActiveStage();
        }

        private void OnHoverEntered(HoverEnterEventArgs args)
        {
            _isHovered = true;
        }

        private void OnHoverExited(HoverExitEventArgs args)
        {
            _isHovered = false;
        }

        public void ExitActiveStage()
        {
            var activeEntrance = StageEntrance.ActiveStageEntrance;
            if (activeEntrance == null)
            {
                Debug.LogWarning("StageExitButtonBridge could not find an active StageEntrance.", this);
                return;
            }

            activeEntrance.ExitStage();
        }
    }
}
