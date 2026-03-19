using System;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.EventSystems;

namespace TalkJourney.BubbleSystem.Interaction
{
    [DisallowMultipleComponent]
    [RequireComponent(typeof(Collider))]
    public class VrPointerInteractable : MonoBehaviour, IVrPointerInteractable, IPointerEnterHandler, IPointerExitHandler, IPointerClickHandler
    {
        public event Action HoverEntered;
        public event Action HoverExited;
        public event Action Clicked;

        [Header("Desktop Fallback")]
        [Tooltip("Enable legacy mouse-based interaction (OnMouseEnter/Exit/Down) for non-VR desktop testing.")]
        public bool enableDesktopMouseFallback = true;

        private bool _isHovering;
        private int _lastClickFrame = -1;

        [Header("Optional Unity Events")]
        public UnityEvent onHoverEntered;
        public UnityEvent onHoverExited;
        public UnityEvent onClicked;

        public void OnPointerEnter(PointerEventData eventData)
        {
            RaiseHoverEntered();
        }

        public void OnPointerExit(PointerEventData eventData)
        {
            RaiseHoverExited();
        }

        public void OnPointerClick(PointerEventData eventData)
        {
            RaiseClicked();
        }

        private void OnMouseEnter()
        {
            if (!enableDesktopMouseFallback)
            {
                return;
            }

            RaiseHoverEntered();
        }

        private void OnMouseExit()
        {
            if (!enableDesktopMouseFallback)
            {
                return;
            }

            RaiseHoverExited();
        }

        private void OnMouseDown()
        {
            if (!enableDesktopMouseFallback)
            {
                return;
            }

            // Prevent duplicate click dispatch when both pointer and mouse callbacks fire in the same frame.
            if (_lastClickFrame == Time.frameCount)
            {
                return;
            }

            RaiseClicked();
        }

        public void InvokeHoverEntered()
        {
            RaiseHoverEntered();
        }

        public void InvokeHoverExited()
        {
            RaiseHoverExited();
        }

        public void InvokeClicked()
        {
            RaiseClicked();
        }

        private void RaiseHoverEntered()
        {
            if (_isHovering)
            {
                return;
            }

            _isHovering = true;
            HoverEntered?.Invoke();
            onHoverEntered?.Invoke();
        }

        private void RaiseHoverExited()
        {
            if (!_isHovering)
            {
                return;
            }

            _isHovering = false;
            HoverExited?.Invoke();
            onHoverExited?.Invoke();
        }

        private void RaiseClicked()
        {
            _lastClickFrame = Time.frameCount;
            Clicked?.Invoke();
            onClicked?.Invoke();
        }
    }
}
